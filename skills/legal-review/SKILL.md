---
name: legal-review
description: >-
  Review, risk-assess, or mark up a Microsoft Word .docx legal document already — a
  contract or agreement, but equally any other legal document (an assessment report, memo,
  filing, notice, or the like), clause by clause. Use this whenever the user asks to
  review, examine, audit, or mark up a legal document — e.g. "review the saas-002
  contract", "review this agreement", "mark up the MSA", or "审查 X 文档" / "依据 <法律> 审查 X
  文档" (review the X document against the cited law) — even when they do NOT name a party or
  a position, and even when the file is NOT literally called a "contract": a bare "review X
  document" request is the normal entry point, not a reason to skip this skill. The skill
  itself asks which party or side the review represents and the posture (dominant, neutral,
  or disadvantaged) before analyzing — skip that step when the document has no opposing
  side. It then analyzes every clause or section through that lens (per clause: a risk
  level with rationale, a suggested change, and any cross-clause contradictions), writes a
  review file (review.json) of suggestions, and produces a tracked-changes redline .docx of
  those suggestions that the user reviews in Word and then accepts, rejects, or refines.
  Once the user has decided, it produces the final clean .docx with the accepted changes
  baked in (no tracked-change markup). This is the document-review workflow, not general
  Q&A about the document. Works on .docx only; the document must already be in the app —
  uploaded to the session or ingested into the app.
metadata:
  requires:
    bins: []
  install:
    - type: pip
      packages:
        - python-docx
---

# legal-review

Review a `.docx` legal document already in the app **through one lens** and produce a structured,
clause-by-clause assessment: per clause a **risk** (level + rationale), an optional
**suggested change**, and any **contradictions** with other clauses. The suggestions are
written to a **review file** (`review.json`) — the durable source of truth — and, in the
same turn, projected into a **tracked-changes redline `.docx`** (all suggestions as
tracked changes) that you hand back as the primary deliverable. The user reviews the
redline in Word / the UI's document panel, then accepts / rejects / refines suggestions
over the conversation; each decision patches `review.json`. Once the user has weighed in,
you hand back the **final clean `.docx`** — the accepted changes applied directly, with no
tracked-change markup — as the finished contract.

You (the agent) do the judgment — deciding risk and drafting suggested language for the
represented party. Two bundled helpers do the mechanical, error-prone parts so your
output stays reliable:

- `segment_clauses.py` — splits the base `.docx` into clauses and gives every paragraph
  a stable id and its **verbatim** text. Anchor your suggestions to these paragraphs by
  id; never hand-copy anchor text.
- `build_ops.py` — turns your analysis into the review file (resolving those ids to
  matchable anchors) and later projects the review file into `edit-docx` operations.

Scope: **Microsoft Word `.docx` only.** This skill *analyzes and marks up*; it does not
answer general questions about the contract (use ordinary retrieval for that).

## Inputs

- **base** — the legal document to review (`base_doc_id`), a `.docx` already in the app
  (uploaded to the session or ingested into the app). A contract or agreement, or any
  other legal document (assessment, memo, filing, notice).
- **review lens** — what the analysis is written *through*. There are two shapes:
  - **party lens** (contracts/agreements with two or more sides) — the **represented
    party** the review speaks for, plus a **review position** (`dominant`, `neutral`,
    `disadvantaged`) that sets how aggressive suggestions are. Derive the candidate parties
    from the document and **confirm both with the user** before analyzing.
  - **compliance lens** (documents with no opposing side, e.g. an assessment reviewed
    "依据 民法典、民事诉讼法") — the **governing law / standard** to check against. Infer the
    applicable law automatically from the document's type and content; if the user named
    specific laws, use those. No party/position question, and no need to confirm the law
    with the user first.

If the base id, or which lens fits, is unclear, ask before proceeding.

## Workflow

1. **Fetch the contract.** Call `fetch_document` with `base_doc_id` to materialize the raw
   `.docx` to a local path (you need the binary, not just extracted text).

2. **Confirm the lens.** Use `read_document` on the base to find the document type, any
   parties, and the governing law. Then pick the lens (see **Inputs**):
   - If the document has two or more opposing sides (a contract/agreement), **ask the user**
     which party the review represents and the review position (`dominant` / `neutral` /
     `disadvantaged`), and wait for the answer — do not analyze until both are confirmed.
   - If the document has no opposing side (e.g. an assessment reviewed against cited law),
     use the **compliance lens**: determine the applicable law/standard automatically from
     the document's type and content — and honor any laws the user named (e.g. "依据
     民法典、民事诉讼法"). No party/position question, and no need to confirm the law first;
     proceed straight to analysis.
   The whole analysis is written through the chosen lens.

3. **Segment the clauses.** Run the segmenter with the `shell` tool (substitute the skill
   base directory printed above):

   ```
   python <skill base directory>/segment_clauses.py \
     --original <fetched path> --output clauses.json
   ```

   `clauses.json` is `{"clauses": [{"clause_id", "heading", "paragraphs": [{"para_id",
   "text"}]}]}`. Read it back.

4. **Analyze each clause.** For every clause, judge it through the confirmed lens — *for
   the represented party at the chosen position* (party lens), or *against the governing
   law/standard* (compliance lens) — and write a raw analysis file `analysis.json`:

   ```json
   {
     "base_doc_id": "<base_doc_id>",
     "meta": {"parties": ["Provider", "Client"], "representative_party": "Client",
              "review_position": "disadvantaged", "contract_type": "MSA",
              "governing_law": "New York"},
     "analyses": [
       {"clause_id": "c1",
        "risk": {"level": "high", "rationale": "30-day payment term strains the Client."},
        "contradicts": [],
        "suggestion": {"op": "replace", "para_id": "c1.p1",
                       "new_text": "Payment shall be due within 45 days of invoice receipt."}}
     ]
   }
   ```

   Rules:
   - `risk.level` is one of `high` / `medium` / `low` / `none`.
   - `suggestion` is `null` when the clause needs no change. Otherwise `op` is one of
     `replace`, `delete`, `insert_after`, `append` (same vocabulary as `edit-docx`):
     `replace`/`delete`/`insert_after` need a `para_id` naming the target paragraph from
     `clauses.json`; `replace`/`insert_after`/`append` need `new_text`. **Reference
     paragraphs by `para_id`** — do not write `anchor_text` yourself.
   - `contradicts` lists other `clause_id`s this clause conflicts with (e.g. a payment
     term that disagrees with a fees schedule). Include clauses you'd flag even when you
     don't suggest a change.
   - Draft `new_text` in the represented party's favor, proportionate to the position
     (party lens); or to bring the clause into compliance with the cited law/standard
     (compliance lens). Never invent obligations the parties didn't discuss; keep clause
     intent intact.
   - For the compliance lens, `meta` carries `governing_law` (the cited law/standard) and
     omits `representative_party`/`review_position` — both are optional downstream.

5. **Build the review file.** Resolve ids to anchors and default every verdict to
   `pending`:

   ```
   python <skill base directory>/build_ops.py finalize \
     --analysis analysis.json --clauses clauses.json --output review.json
   ```

   A non-existent `para_id` or a malformed op fails loudly here — fix the analysis and
   rerun rather than shipping a review that can't be applied.

6. **Persist the review as working state.** Call `save_artifact` with `review.json` and a
   descriptive `filename` (e.g. `<contract-name>-review.json`). Note the returned
   **artifact id** — it is the handle for the rest of the conversation. Reopen it later
   with `fetch_artifact`, patch verdicts or suggestions, and `save_artifact` a fresh copy.

7. **Produce and return the redline.** Don't stop at the review file — on the first review
   the redline is the primary deliverable. Project every suggestion to a tracked-changes
   `.docx` and hand it back in the same turn (see *Producing the redline and the final docx*
   below). The UI's document panel auto-reveals the latest `.docx` artifact, so the user
   reviews the marked-up contract in Word rather than reading JSON.

8. **Summarize for the user.** With the redline link in hand, report the clauses by risk
   (highest first), the suggested changes, and any contradictions, and tell the user the
   redline shows every suggestion as a tracked change — they can accept, reject, or refine
   any of them ("accept c6 and c8, reject c10, soften c11") and you'll hand back the final
   clean contract with their accepted changes applied.

## Producing the redline and the final docx

Project the review file to `edit-docx` operations and run the `edit-docx` apply helper.
There are two moments, and they differ in **which** suggestions go in and **how** they're
applied:

**First review — a tracked-changes redline of every suggestion.** Use `--all` so every
suggestion appears (verdicts are still `pending`), and apply them as tracked changes (the
default, no `--clean`) so the user can accept/reject each in Word:

```
python <skill base directory>/build_ops.py to-edit-ops \
  --review review.json --all --output edit_ops.json

python <edit-docx base directory>/apply_operations.py \
  --original <fetched path> --ops edit_ops.json --output redline.docx --author "legal-review"
```

`save_artifact` the `redline.docx` (e.g. `<contract-name>-redline.docx`) and include the
download link, noting it is a tracked-changes redline reviewable in Word.

**After the user accepts/rejects — the final clean docx.** Drop `--all` so `to-edit-ops`
includes only clauses whose `verdict` is `accepted`, and pass `--clean` so the accepted
changes are baked in directly, with no tracked-change markup — this is the finished
contract, not another redline:

```
python <skill base directory>/build_ops.py to-edit-ops \
  --review review.json --output edit_ops.json

python <edit-docx base directory>/apply_operations.py \
  --original <fetched path> --ops edit_ops.json --output final.docx --clean
```

`save_artifact` the `final.docx` (e.g. `<contract-name>-final.docx`) and hand back its link
as the finished contract. The anchors were taken verbatim from the base during
segmentation, so they match; still check `apply_operations`' `unmatched` count and surface
any misses in either pass.

## Refining across turns

The review file is the single source of truth for the review. When the user accepts,
rejects, or asks to reword suggestions ("accept clause 1, reject clause 3, soften clause
5"), don't hand-edit the JSON — apply the changes deterministically with `build_ops.py
patch` so a mistyped `clause_id` or verdict fails loudly instead of corrupting the file:

1. `fetch_artifact` the current review file to a local path.
2. Apply the changes:

   ```
   # Verdicts go on the command line; rewording/dropping a suggestion goes in a patch file.
   python <skill base directory>/build_ops.py patch \
     --review review.json --accept c1 --reject c3 \
     --patch changes.json --output review.json
   ```

   `changes.json` is optional and only needed to reword or drop a suggestion:

   ```json
   {"suggestions": {
     "c5": {"new_text": "Payment shall be due within 30 days of invoice receipt."},
     "c8": null
   }}
   ```

   A `null` drops the clause's suggestion so it's never applied; a fields dict is merged
   into the existing suggestion (keeping the baked `anchor_text`, so a reworded change
   still matches the base). Verdicts may also be given in the patch file under `"verdicts"`
   instead of the flags.

3. `save_artifact` the updated review file, then produce the **final clean docx** of the
   accepted changes (see *After the user accepts/rejects* above) and hand back its link. If
   the user is still weighing options and asks to see the effect first, you can regenerate a
   redline of the accepted-only suggestions instead (drop `--all`, keep tracked changes).

When the review is complete, `delete_artifact` the review file (and any stale redline
previews) to clean up the working state.

## Helper reference

- `segment_clauses.py --original in.docx [--output clauses.json]` — emit clauses with
  per-paragraph verbatim anchors.
- `build_ops.py finalize --analysis a.json --clauses c.json [--output review.json]` —
  resolve `para_id`s to anchors, default verdicts to `pending`, validate.
- `build_ops.py patch --review review.json [--accept ID...] [--reject ID...]
  [--pending ID...] [--patch changes.json] [--output review.json]` — apply
  accept/reject/refine decisions; validates every `clause_id`, verdict, and resulting op.
- `build_ops.py to-edit-ops --review review.json [--all] [--output edit_ops.json]` —
  project to `edit-docx` operations (accepted-only unless `--all`).
- `<edit-docx>/apply_operations.py --original in.docx --ops edit_ops.json --output out.docx
  [--clean]` — apply the ops to the base; tracked-changes redline by default, or a clean
  final docx with `--clean`.

## Limitations

Segmentation splits on headings and section numbers (`ARTICLE`/`SECTION`, `4`, `4.2`,
roman numerals) and ALL-CAPS title lines; unusual numbering may under-segment, though
anchors still resolve at the paragraph level. Redlining inherits `edit-docx`'s scope —
body paragraphs only, not tables/headers/footers, and a `replace` swaps a whole paragraph
rather than a word-level diff. The analysis is decision support, not legal advice.
