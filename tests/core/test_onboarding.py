"""Tests for cogbase/core/onboarding.py — the interview's frame and resolver.

The module is deliberately content-free: the questions and the profile template
both live in a SKILL.md, so what is tested here is the plumbing around them and
the refusal to invent an interview when no skill is registered.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cogbase.core.onboarding import (
    INTERVIEW_SKILL_NAME,
    INTERVIEW_TOOLS,
    SAVE_COMPANY_PROFILE_TOOL_NAME,
    build_interview_system_prompt,
    resolve_interview_script,
)
from cogbase.skills.registry import SkillRegistry
from cogbase.skills.skill import APPLICATION_SURFACE, ONBOARDING_SURFACE, Skill

SKILL_BODY = "# Custom interview\n\nAsk about their fleet of submarines.\n"


def _skill(
    markdown: str,
    *,
    name: str = INTERVIEW_SKILL_NAME,
    skill_id: str = "s1",
    builtin: bool = True,
    surface: str = ONBOARDING_SURFACE,
) -> Skill:
    """An interview skill as ``skills_dir`` would register one: a builtin on the
    onboarding surface. Both defaults are overridable — the resolver's rules about
    them are what several tests below are checking."""
    return Skill(
        name=name,
        description="d",
        raw_markdown=markdown,
        id=skill_id,
        builtin=builtin,
        surface=surface,
    )


class TestBuildInterviewSystemPrompt:
    def test_the_script_is_the_body_of_the_prompt(self):
        prompt = build_interview_system_prompt(SKILL_BODY, today="2026-07-30")

        assert "fleet of submarines" in prompt
        assert "2026-07-30" in prompt
        assert "{TODAY}" not in prompt
        assert "{SCRIPT}" not in prompt

    def test_the_tool_contract_survives_any_script(self):
        """A swapped script describes the conversation; the frame keeps the plumbing."""
        prompt = build_interview_system_prompt(SKILL_BODY)

        assert SAVE_COMPANY_PROFILE_TOOL_NAME in prompt
        assert "profile template from the script above" in prompt

    def test_the_frame_carries_no_profile_template_of_its_own(self):
        """The document's shape belongs to the vertical, not to the framework."""
        prompt = build_interview_system_prompt(SKILL_BODY)

        assert "# Company Profile" not in prompt
        assert "Risk appetite" not in prompt

    def test_no_rerun_block_without_an_existing_profile(self):
        assert "BEGIN CURRENT PROFILE" not in build_interview_system_prompt(SKILL_BODY)

    def test_existing_profile_makes_it_a_rerun(self):
        prompt = build_interview_system_prompt(
            SKILL_BODY, existing_profile="# Us\n\nWe do widgets."
        )

        assert "BEGIN CURRENT PROFILE" in prompt
        assert "We do widgets." in prompt
        assert "re-run" in prompt
        assert "{PROFILE}" not in prompt

    @pytest.mark.parametrize("blank", ["", "   \n  "])
    def test_blank_existing_profile_is_treated_as_absent(self, blank):
        assert "BEGIN CURRENT PROFILE" not in build_interview_system_prompt(
            SKILL_BODY, existing_profile=blank
        )

    def test_only_the_save_tool_is_offered(self):
        """The interview has one job, so it gets one tool."""
        assert [t["name"] for t in INTERVIEW_TOOLS] == [SAVE_COMPANY_PROFILE_TOOL_NAME]


class TestResolveInterviewScript:
    """No script is a state, not a reason to improvise one.

    A generic fallback in the framework would ask the wrong customer the wrong
    questions and then save the answers forever, and would make every new vertical
    a code change.
    """

    def test_no_registry_yields_no_script(self):
        assert resolve_interview_script(None) is None

    def test_missing_skill_yields_no_script(self):
        assert resolve_interview_script(SkillRegistry(), "acme") is None

    def test_a_registered_skill_supplies_the_script(self):
        registry = SkillRegistry()
        registry.register(_skill(SKILL_BODY))

        assert "fleet of submarines" in resolve_interview_script(registry, "acme")

    def test_front_matter_is_stripped(self):
        """Front-matter is registry bookkeeping; this script is chosen by name."""
        registry = SkillRegistry()
        registry.register(_skill(
            f"---\nname: {INTERVIEW_SKILL_NAME}\ndescription: noise\n---\n{SKILL_BODY}"
        ))

        script = resolve_interview_script(registry, "acme")

        assert "description: noise" not in script
        assert script.startswith("# Custom interview")

    def test_a_global_builtin_is_visible_to_every_account(self):
        registry = SkillRegistry()
        registry.register(_skill(SKILL_BODY), account_id=None)

        assert "fleet of submarines" in resolve_interview_script(registry, "any-account")

    def test_another_accounts_skill_is_not_used(self):
        registry = SkillRegistry()
        registry.register(_skill(SKILL_BODY), account_id="other")

        assert resolve_interview_script(registry, "acme") is None

    def test_an_empty_skill_body_yields_no_script(self):
        registry = SkillRegistry()
        registry.register(_skill("---\nname: x\n---\n"))

        assert resolve_interview_script(registry, "acme") is None

    def test_a_broken_registry_does_not_raise(self):
        registry = MagicMock()
        registry.get_by_name.side_effect = RuntimeError("registry down")

        assert resolve_interview_script(registry, "acme") is None

    def test_an_uploaded_skill_is_not_used_as_the_script(self):
        """Trust comes from the load path: only skills_dir (operator-written) may
        supply the script. An uploaded skill is owned by one account, so honouring
        one would put a tenant's markdown in an account-level system prompt — and
        would leave every *other* account with no interview at all."""
        registry = SkillRegistry()
        registry.register(_skill(SKILL_BODY, builtin=False), account_id="acme")

        assert resolve_interview_script(registry, "acme") is None

    def test_a_builtin_without_the_surface_still_works(self):
        """A self-hoster who writes their own interview and forgets the metadata
        line gets a working interview (plus a log line), not a silent 503. The
        surface governs listing and assignment, not trust."""
        registry = SkillRegistry()
        registry.register(_skill(SKILL_BODY, surface=APPLICATION_SURFACE))

        assert "fleet of submarines" in resolve_interview_script(registry, "acme")

    def test_another_vertical_is_another_name(self):
        """The seam for a second vertical: a skill under its own name."""
        registry = SkillRegistry()
        registry.register(_skill(SKILL_BODY, name="account-onboarding-interview-vc"))

        assert resolve_interview_script(registry, "acme") is None
        assert "submarines" in resolve_interview_script(
            registry, "acme", name="account-onboarding-interview-vc"
        )


class TestShippedSkill:
    """``skills/account-onboarding-interview-legal`` is the whole legal interview."""

    def test_it_carries_the_questions_and_the_profile_template(self):
        script = self._shipped_script()

        assert "legal layer" in script.lower()
        assert "## Legal practice" in script
        assert "**Practice setting:**" in script
        assert "# Company Profile" in script          # the template to fill in
        assert "name: account-onboarding-interview" not in script  # front-matter gone

    def test_the_prompt_built_from_it_is_a_complete_interview(self):
        prompt = build_interview_system_prompt(self._shipped_script())

        assert "Quick or full?" in prompt             # the questions
        assert "# Company Profile" in prompt          # the document shape
        assert SAVE_COMPANY_PROFILE_TOOL_NAME in prompt  # the plumbing

    @staticmethod
    def _shipped_script() -> str:
        import pathlib

        skills_dir = pathlib.Path(__file__).resolve().parents[2] / "skills"
        registry = SkillRegistry()
        registry.load_from_dir(skills_dir, skill_names=["account-onboarding-interview-legal"])
        return resolve_interview_script(
            registry, "acme", name="account-onboarding-interview-legal"
        )
