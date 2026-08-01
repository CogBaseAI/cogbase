# Preference Profiles

CogBase apps produce generic output until they learn how *this* customer works.
Two lawyers reviewing the same contract want different things: one caps liability
at 12 months and refuses any IP assignment; the other lives with 24 months and
cares only about auto-renewal windows. Neither preference is derivable from the
documents — it has to be *collected*. This document describes how CogBase
collects, stores, and applies customer preferences, in two scoped tiers.

## Two tiers, matched to the scope boundaries

| Tier | Scope | Holds | Status |
|---|---|---|---|
| **Account profile** | account | who the customer is: industry, what they work on, which side of the table, the one deal-breaker, risk appetite, jurisdictions, regulators, who reads the output, role | **built** |
| **App preferences** | app | how *this* app should behave: playbook positions, extraction priorities, review framing, escalation thresholds | **planned** |

The account profile is stable org-wide context that should never be re-collected
per app. App preferences are specific to one app's pipeline and query behavior.
This mirrors the scoping already in the factory (`AppScope(account_id,
namespace_id, app_id)`) — see [long-term-memory.md](long-term-memory.md), where
the app is the partition boundary.

The boundary between the two is worth stating precisely, because it is easy to
get wrong. Playbook *positions* — LoL cap, indemnity direction, renewal windows —
are thresholds that vary by deal and by app, so they sit at the app level. The
account level holds what is true across every app: who the customer is, which
side they sit on, their one deal-breaker, jurisdictions, regulators, risk
appetite, role. The "one thing" deal-breaker looks like a playbook entry but
belongs to the account, because it is a refusal that holds across every matter
rather than a threshold that gets negotiated.

---

# Account profile

## Storage

A **markdown document**, not YAML — the user reads and edits it in plain English.
It lives in the system document store under
`AppScope(account_id=account_id)`, collection `"profile"`, doc_id
`"company-profile.md"` (`cogbase/core/profile.py`). Dropping `namespace_id` and
`app_id` from the scope is what makes one document shared across every namespace
and app in the account.

Edit metadata (`updated_at`, `updated_by`, `source` ∈ `interview|manual`) lives
in a `profile_records` index in `api/system_store.py`, following the
`skill_records` precedent — bytes in the document store, index in the system
store.

We considered modelling the profile as a new account-scoped memory tier instead,
to inherit long-term memory's reconcile loop. A plain document won: the memory
route costs the RBAC / multi-partition design that `long_term.py` explicitly
defers, and buys nothing the interview needs today.

## Runtime injection

`build_app` reads the profile once and hands it to the `QueryRunner`, which
renders it as a labeled system block ahead of the retrieval prompt
(`_format_profile_block`, `build_system_prompt`). Two properties matter:

- **Unconditional, not recalled.** Long-term memory recall is a vector search —
  query-dependent and free to return nothing. The profile is small, stable, and
  always relevant to framing, so it is injected every turn like a system preamble.
  A fixed slot in the prompt prefix also makes it a good prompt-cache candidate.
- **Labeled but not citable.** The block states that it is customer-supplied
  framing context, not document evidence. It carries no citation id: ids exist so
  the runner can resolve them back to `LongTermRecord`s, and the profile has no
  records behind it.

The read is best-effort — a document-store hiccup logs and leaves the app
profile-blind rather than failing app construction, which happens on six call
paths including startup provisioning.

**Not yet injected:** workflow LLM steps. House style matters for review
workflows, but `WorkflowRunner` has no prompt-preamble seam.

## API and editing

`GET`/`PUT`/`DELETE /profile`, account-scoped with no namespace segment
(`api/routers/profile.py`). `GET` returns `200` with `exists: false` when there is
no profile — a cold start is a state, not a 404. `PUT` refuses bodies over
`MAX_PROFILE_BYTES` with 413. A deployment with no system document store cannot
hold profiles at all, so the routes 503: a failed *read* should degrade quietly, a
failed *write* must not look like it succeeded.

A write **hot-patches** the account's cached app instances
(`CogBaseApp.set_account_profile`) rather than evicting them — evicting would throw
away warm memory tiers, a skill registry, and wired workflows, and charge the next
query for a rebuild, a steep price for editing a page of prose. The patch is local
to the node handling the write; other nodes serve the old profile until their
app-cache TTL expires (≤60s). That staleness is accepted, not engineered away.

The editing surface is **both** a profile card in Settings (textarea, save, reset,
delete, and a "re-run the interview" button) and the interview itself. Settings is
shown in saas mode for this reason, with the provider sections gated off.

## The interview

The interview **is** its script, and the script is a skill:
`skills/account-onboarding-interview-legal/SKILL.md` holds the questions *and* the
company-profile template they fill in. It is resolved by name from the skill
registry (`INTERVIEW_SKILL_NAME`, overridable with `COGBASE_INTERVIEW_SKILL`), so a
non-engineer can edit it and a second vertical is a second SKILL.md — never a
branch in code. With no script registered the route 503s rather than improvising an
interview.

Only the *markdown* is adopted, not skill execution: no router LLM call, no session
workdir, no subprocess sandbox. This is a text interview.

`cogbase/core/onboarding.py` holds only the plumbing — the `save_company_profile`
tool, today's date, the re-run block for an account that already has a profile, and
the frame binding them. There is no default script and no profile template in code.

**Its own surface, not the generator chat.** `POST /profile/interview/chat` and
`/chat/stream` (SSE), stateless and account-scoped like `/generate/chat`: the client
holds the message history, the server runs the agent loop. The interview does not
live inside the generator chat because a new account is auto-provisioned with a
working namespace and app (`api/provisioning.py`) — the SMB user who most needs
onboarding may never open the Build tab. The interview also runs once per account
and is re-run on demand, while the generator runs once per app.

The interview **degrades rather than 503s**: a save failure comes back as the tool
result so the model can tell the user and the conversation survives. Only an
unconfigured LLM is a hard error. On success it writes the document, records
`source="interview"`, and hot-patches live apps — which is what makes the
auto-provisioned app pick the profile up mid-session.

### Shape of the questions

The script is the source of truth for wording and order; what follows is the design
behind it.

Questions are **ordered by marginal impact on output**, so stopping at any point
leaves something usable. There is no fixed finish line — once the core questions are
in, the script says so and offers the rest as optional. Partial profiles are the
expected case, not the degraded one.

Two behaviors are worth their cost:

- **Assume the answer exists somewhere.** For anything longer than a sentence —
  playbook, escalation matrix, company description — ask for a link or paste before
  asking the user to type from memory.
- **Don't ask what a document can tell you.** (See [the learn loop](#the-learn-loop)
  for the unbuilt half of this.)

**Base questions (vertical-agnostic).** Core: who the org is + industry · what
documents it runs on, what decisions they drive, what must never be missed · risk
appetite · who reads the output. Then, if the user is still engaged: always-escalate
triggers · regulators and compliance regimes · jurisdictions. The documents question
is what yields vertical-specific content without the script naming a vertical — it is
what gives a VC firm or a research lab a useful profile when no vertical layer applies.

**Legal-vertical layer.** *Which side* (their paper / counterparty paper / both) is the
highest-leverage answer in the interview — a liability cap, an indemnity, an
auto-renewal each reads as protection on one side and exposure on the other, and
without it every review hedges both ways. Then *the one thing* that makes them refuse
to sign, *role* (which drives the work-product header and research-framed vs. verdict
output), *governing law*, and *practice setting*.

**Deliberately not asked:** team size, house style, who holds final sign-off by name,
and where executed documents live. None change what a review finds; the last returns
when there is a connector to point at.

### The profile template

Read it from `## The profile template` in the interview SKILL.md. It is not
reproduced here — this document carried a copy once and it went stale the first time
the script changed, which is the failure mode that keeping the template in the script
exists to prevent. Its shape: a dateline, then `## Who we are`, `## What we work on`,
`## How we want outputs`, `## Regulators & compliance`, `## Escalation`, and
`## Legal practice`. Empty sections are dropped at save time.

## Onboarding UI

The onboarding card fires **after the first completed answer** in the Query tab, not
at signup and not after the first ingest: the ask trades on value already delivered,
and it never shares a screen with the Ingest tab's review CTA, which is the one thing
we want clicked first. It is dismissible ("Not now", persisted per account), never
blocking — the account already has a working workspace, so the interview competes
with real work and must lose gracefully. Settings keeps a standing prompt as backstop.

`GET /profile` is read three ways in `context.jsx`: not fetched yet; fetched but
*unknown* (the route 503'd, or the deployment has no document store); and a known cold
start (`exists: false`). Only the third offers onboarding, so a deployment that cannot
hold profiles never dangles an interview that would fail on open.

---

# App preferences (planned)

App-level preferences split by *where they take effect*. This is the one place CogBase
diverges from a pure prompt-context design, because it has a compile step:

| Kind | Example | Destination |
|---|---|---|
| **Compile-time** | "I always need the auto-renewal cancel-by date" → a `cancel_by` extraction field; "flag playbook deviations" → a review workflow step | the generated `config.yaml` |
| **Runtime** | "keep redlines terse"; a house tone; a per-app deal-breaker check | the app's `config.query_prompt` |

Both are collected in the **generator chat**, which is already a conversational
interview ("document types, facts that matter, example questions") — extending it with
deal-breakers, review priorities, and escalation thresholds needs no new machinery.
First-cut routing rule: anything implying a new extractable field or a review/compare
step is compile-time; anything about tone, framing, or a single check is runtime.

Runtime preferences go to `query_prompt` rather than long-term-memory
`preference` records for the same reason the account profile is a static injection:
stable framing must apply unconditionally, and recall can return nothing.
`query_prompt` is also versioned with the config and already editable in the UI.
Long-term memory earns its keep on the *drift* half instead — see below.

**The generator is currently profile-blind.** It neither writes an account profile nor
reads one. It does not write because a stub profile saved from an offhand remark sets
`exists: true` and thereby suppresses the real onboarding it was meant to complement.
It does not read because the value, while plausible — jurisdictions, regulators, and
escalation triggers genuinely shape a config, and config decisions are sticky in a way
prompt framing is not — is unproven, and the failure mode is untested: a profile-primed
model can bolt GDPR fields onto an app about support tickets. The test that settles it
is a live one: the same generator conversation with and without a profile, asserting
the non-legal app grows no jurisdiction fields. Add the read path back behind that test.

---

# The learn loop (later)

The profile is a seed, not the last word. Two existing mechanisms keep it current:

- **Runtime preference drift** → long-term memory `reconcile` (ADD / UPDATE / DELETE /
  NOOP against accumulated belief), see [long-term-memory.md](long-term-memory.md).
- **Config-shaping gaps** → adaptive evolution (new field / step / skill mined from
  episodic signals), see [adaptive-evolution.md](adaptive-evolution.md).

The third, unbuilt piece is **learning the real preferences from seed documents**: read
the customer's actually-signed documents and compute the delta between stated positions
and what was signed — the delta is the real playbook. This is nearly free in CogBase,
since `extract-structured` already pulls the structured facts; it needs only a
`profile-refine` pass comparing interview answers against the extracted record
distribution. It is deferred because it requires documents already ingested, so it
belongs after an app exists.

The onboarding interview's job is only to avoid a cold start.
