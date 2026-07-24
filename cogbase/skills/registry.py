"""Registry for looking up skills by id."""

from __future__ import annotations

from pathlib import Path

from cogbase.skills.skill import Skill, load_skills


class SkillRegistry:
    """Maps skill ids to loaded ``Skill`` instances, scoped by owning account.

    Skills are keyed by their stable *id* (a UUID for uploaded skills, the
    directory name for dev-time ``skills_dir`` loads); ids are globally unique, so
    id lookup is account-agnostic. Uploaded skills are owned by an *account_id* and
    only visible to that account; ``skills_dir`` builtins are registered with
    ``account_id=None`` and are visible to **every** account (platform-level).
    Names are for display and are unique only within an account's visible set.
    """

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}
        # skill_id -> owning account_id; ``None`` marks a global (builtin) skill.
        self._accounts: dict[str, str | None] = {}

    def _visible(self, account_id: str | None) -> list[str]:
        """Ids visible to *account_id*: its own skills plus global builtins."""
        return [
            sid for sid, owner in self._accounts.items()
            if owner is None or owner == account_id
        ]

    def register(
        self, skill: Skill, *, account_id: str | None = None, replace: bool = False
    ) -> None:
        """Register *skill* by its id under *account_id* (``None`` = global builtin).

        Raises ``ValueError`` if the id is missing, if it is already taken and
        *replace* is False, or if another skill **visible to the same account**
        already uses the same name.
        """
        if not skill.id:
            raise ValueError("Cannot register a skill without an id.")
        if skill.id in self._skills and not replace:
            raise ValueError(
                f"A skill with id '{skill.id}' is already registered. "
                "Pass replace=True to overwrite, or unregister it first."
            )
        # Name uniqueness is scoped to what the owning account can see (its own
        # skills + globals), so two accounts may reuse a name without colliding.
        for existing_id in self._visible(account_id):
            existing = self._skills[existing_id]
            if existing.name == skill.name and existing_id != skill.id:
                raise ValueError(
                    f"A skill with name '{skill.name}' is already registered under id '{existing_id}'."
                )
        self._skills[skill.id] = skill
        self._accounts[skill.id] = account_id

    def unregister(self, skill_id: str) -> None:
        """Remove the skill with *skill_id*. No-op if it is not registered."""
        self._skills.pop(skill_id, None)
        self._accounts.pop(skill_id, None)

    def get(self, skill_id: str) -> Skill:
        """Return the skill for *skill_id* (account-agnostic). Raises ``KeyError``."""
        if skill_id not in self._skills:
            known = ", ".join(sorted(self._skills)) or "(none)"
            raise KeyError(f"No skill with id '{skill_id}'. Known ids: {known}")
        return self._skills[skill_id]

    def get_by_name(self, name: str, account_id: str | None = None) -> Skill:
        """Return the skill *name* visible to *account_id*. Raises ``KeyError``."""
        for skill_id in self._visible(account_id):
            if self._skills[skill_id].name == name:
                return self._skills[skill_id]
        raise KeyError(f"No skill with name '{name}'")

    def all_skills(self, account_id: str | None = None) -> list[Skill]:
        """Return skills visible to *account_id* (its own plus global builtins)."""
        return [self._skills[sid] for sid in self._visible(account_id)]

    def load_from_dir(self, skills_dir: str | Path, skill_names: list[str] | None = None) -> None:
        """Scan *skills_dir* for SKILL.md files and register them as global builtins.

        When *skill_names* is given, only those subdirectories are loaded.
        Otherwise every subdirectory with a SKILL.md is loaded. Each skill is
        registered under id = its directory name (dev-time convenience path) with
        ``account_id=None`` so it is visible to every account.
        """
        skills_dir = Path(skills_dir)
        if skill_names is None:
            skill_names = [p.name for p in skills_dir.iterdir() if p.is_dir()]
        for skill in load_skills(skill_names, skills_dir):
            if skill.id not in self._skills:
                self._skills[skill.id] = skill
                self._accounts[skill.id] = None
