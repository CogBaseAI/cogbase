"""Account onboarding — the interview that writes the company profile.

The company profile is stable org-wide context (who the org is, jurisdictions,
regulators, risk appetite, house style).  It is collected once per account by a
short conversational interview and read forever after by the query runner — see
docs/preference-profiles.md.

This module owns the *plumbing* for that interview, deliberately apart from both
its transport (``POST /profile/interview/chat``, in ``api/routers/profile.py``)
and its content.

The interview does *not* live in the app-generator chat because a freshly-minted
account is already provisioned with a namespace and an app (``api/provisioning.py``);
that user has no reason to open the Build tab, so an interview hosted there would
never run for the accounts it was written for.  The generator is out of the
profile business entirely — it neither collects one nor reads one.

**The script is a skill, and it is the whole script.**  In CogBase a ``Skill`` is
a SKILL.md whose markdown is injected as prompt context, so shipping the interview
as a skill makes it a versioned, uploadable document a non-engineer can edit —
see :func:`resolve_interview_script`.  What we do not use is the skill *execution*
machinery (the router LLM call, session workdir, subprocess sandbox): this is a
text interview, and it needs the markdown, not the sandbox.

So this module holds only what is true of **every** interview regardless of who
is being interviewed: the save tool, today's date, and the instruction to call
the tool with the template the script supplies.  The questions *and* the shape of
the profile document both belong to a SKILL.md; ``skills/`` ships the legal one.
There is deliberately no fallback script here — a vertical's interview is content,
and content that lives in the framework becomes a code change per vertical.  A
deployment with no interview skill registered has no interview, and the route says
so rather than onboarding everyone with a generic one.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import date

from cogbase.llms.base import ToolDefinition
from cogbase.skills.skill import ONBOARDING_SURFACE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

SAVE_COMPANY_PROFILE_TOOL_NAME = "save_company_profile"

SAVE_COMPANY_PROFILE_TOOL: ToolDefinition = {
    "name": SAVE_COMPANY_PROFILE_TOOL_NAME,
    "description": (
        "Save the account's company profile. Pass the filled-in company-profile markdown. "
        "The profile is account-scoped: every app in this account reads it, and these "
        "questions are never asked again. Saving does not end the conversation — narrate "
        "the result and carry on."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "markdown": {
                "type": "string",
                "description": (
                    "The completed company-profile markdown, following the template given "
                    "in the interview script. Prose the user can read and edit — never "
                    "YAML. Omit sections you have no answers for rather than leaving "
                    "placeholders."
                ),
            },
        },
        "required": ["markdown"],
        "additionalProperties": False,
    },
}

#: Tools available to the interview loop. The interview has exactly one job, so
#: it has exactly one tool — and it is the only surface that writes a profile
#: through an LLM.
INTERVIEW_TOOLS: list[ToolDefinition] = [SAVE_COMPANY_PROFILE_TOOL]


# ---------------------------------------------------------------------------
# Prompt frame
# ---------------------------------------------------------------------------

#: Wraps the script with the parts the transport guarantees: what product this
#: is, today's date, and the tool contract. Everything else — the questions, the
#: profile template they fill in, the closing — is the script's, so a new vertical
#: is a new SKILL.md rather than an edit here.
_INTERVIEW_FRAME = """\
You are running the account onboarding interview for CogBase, a platform where a \
customer's documents are ingested, extracted, and queried by AI applications they \
configure. Today is {TODAY}.

{SCRIPT}

## Saving

When you have enough — or the user wants to stop — call `save_company_profile` \
with the profile template from the script above, filled in. Drop any section you \
have no answer for; never leave a bracketed placeholder in the saved text. Saving \
does not end the conversation: narrate the result, then continue.

If the user later corrects something, call `save_company_profile` again with the \
full updated document — it replaces the previous version rather than appending to it.\
"""

#: Appended when the account already has a profile, which makes a re-run ("my
#: playbook changed", the Settings re-run button) the same code path as a first
#: run rather than a second mode to maintain.
#:
#: A re-run has two quite different reasons behind it, and the interview cannot
#: know which one it is looking at until it reads what is saved. *Something
#: changed* is the obvious one. *The profile was never finished* is now the common
#: one: the script has no fixed finish line — it asks in descending order of value
#: and the user leaves when they've had enough, at which point the UI's exit save
#: writes whatever was collected. So a first run ending early is the designed
#: outcome, not a failure. Asking "what changed?" about a document with four empty
#: sections is the wrong opening question, so the model is told to work out which
#: case it is in and let the user pick.
_RERUN_PROMPT = """\

## This account already has a profile

It is below. This is a **re-run**, so do not start from scratch.

First read it against the template and questions in the script above, and work out \
what is *missing* — sections that were dropped, and lines carrying no real answer. \
A partial profile is normal here: the interview has no fixed finish line, so one the \
user left early saved exactly what it had at that point. So open by telling them in a \
line or two what you already have and what is still blank, then offer the choice:

- **finish it** — you ask only the questions the profile has no answer for, in the \
script's order, and leave everything already answered alone.
- **update it** — they say what changed, and you touch only that.

If nothing is missing, do not invent gaps to have something to offer: say it looks \
complete and ask what changed. Either way, ask only what the chosen path needs, and \
when you save, pass the full updated document — the old one is replaced, so anything \
you leave out is deleted.

--- BEGIN CURRENT PROFILE ---
{PROFILE}
--- END CURRENT PROFILE ---\
"""


# ---------------------------------------------------------------------------
# Script resolution
# ---------------------------------------------------------------------------

#: The skill whose markdown *is* the interview. Ships as a builtin under
#: ``skills/``. The default names the legal variant because that is what CogBase
#: ships; a deployment serving another vertical registers its own SKILL.md and
#: points ``COGBASE_INTERVIEW_SKILL`` at that name, rather than patching code.
#: (Uploading a skill under *this* name does not shadow the builtin — the registry
#: enforces name uniqueness across an account's visible set, so a second script
#: needs a second name.)
INTERVIEW_SKILL_NAME = os.environ.get(
    "COGBASE_INTERVIEW_SKILL", "account-onboarding-interview-legal"
)

_FRONT_MATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)


def _strip_front_matter(markdown: str) -> str:
    """Drop a SKILL.md's YAML front-matter, keeping the prose below it.

    The front-matter is registry bookkeeping (name, description used for skill
    *selection*), and this script is chosen by name rather than selected — so in
    the prompt it is noise.
    """
    return _FRONT_MATTER_RE.sub("", markdown, count=1).strip()


def resolve_interview_script(
    registry, account_id: str | None = None, *, name: str = INTERVIEW_SKILL_NAME
) -> str | None:
    """Return the interview script for *account_id*, or ``None`` if there is none.

    Resolution is by **name**, not through the skill router:
    ``SkillRegistry.get_by_name`` already scopes lookup to "this account's own
    skills plus global builtins", so the shipped script is a ``skills_dir``
    builtin and an account-uploaded skill under a name of its own is reachable by
    passing that *name*.

    ``None`` means the deployment has no interview to run — no registry, no such
    skill, or an empty one. The caller surfaces that as a 503 rather than
    substituting a generic script, because a made-up interview would ask the wrong
    customer the wrong questions and save the answers forever.

    The script must be a ``skills_dir`` **builtin**, i.e. operator-written. An
    uploaded skill is owned by one account, so pointing ``COGBASE_INTERVIEW_SKILL``
    at one would resolve for that account and leave every other account without an
    interview; refusing it turns that into a startup-visible misconfiguration
    instead of a per-account mystery, and keeps one tenant's markdown out of
    another's account-level system prompt.
    """
    if registry is None:
        return None
    try:
        skill = registry.get_by_name(name, account_id)
    except KeyError:
        logger.warning("no interview skill named '%s' is registered", name)
        return None
    except Exception:
        logger.warning("interview skill lookup failed name=%s", name, exc_info=True)
        return None

    if not getattr(skill, "builtin", False):
        logger.warning(
            "interview skill '%s' is an uploaded skill, not a skills_dir builtin — "
            "refusing to use it as the interview script",
            name,
        )
        return None

    # Advisory only: the surface governs listing and app assignment, not trust, so a
    # deployment that writes its own interview and forgets the metadata line still
    # gets a working interview — plus a log line telling it how to tidy up.
    if getattr(skill, "surface", None) != ONBOARDING_SURFACE:
        logger.warning(
            "interview skill '%s' does not declare 'metadata.surface: %s' — it works, "
            "but it will also be listed and assignable as an application skill",
            name,
            ONBOARDING_SURFACE,
        )

    script = _strip_front_matter(skill.raw_markdown or "")
    if not script:
        logger.warning("interview skill '%s' has no body", name)
        return None
    logger.info("interview script from skill '%s' (id=%s)", name, skill.id)
    return script


def build_interview_system_prompt(
    script: str,
    *,
    today: str | None = None,
    existing_profile: str | None = None,
) -> str:
    """Compose the interview's system prompt around *script*.

    *script* is the skill-supplied conversation guide, which also carries the
    profile template; *existing_profile* turns the turn into a re-run over what is
    already saved.
    """
    prompt = (
        _INTERVIEW_FRAME
        .replace("{TODAY}", today or date.today().isoformat())
        .replace("{SCRIPT}", script.strip())
    )
    if existing_profile and existing_profile.strip():
        prompt += _RERUN_PROMPT.replace("{PROFILE}", existing_profile.strip())
    return prompt
