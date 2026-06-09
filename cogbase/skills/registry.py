"""Registry for looking up skills by id."""

from __future__ import annotations

from pathlib import Path

from cogbase.skills.skill import Skill, load_skills


class SkillRegistry:
    """Maps skill ids to loaded ``Skill`` instances.

    Skills are keyed by their stable *id* (a UUID for uploaded skills, the
    directory name for dev-time ``skills_dir`` loads). Names are for display only
    and may change without affecting application references.
    """

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill, *, replace: bool = False) -> None:
        """Register *skill* by its id.

        Raises ``ValueError`` if the id is missing, if it is already taken and
        *replace* is False, or if another skill already uses the same name.
        """
        if not skill.id:
            raise ValueError("Cannot register a skill without an id.")
        if skill.id in self._skills and not replace:
            raise ValueError(
                f"A skill with id '{skill.id}' is already registered. "
                "Pass replace=True to overwrite, or unregister it first."
            )
        for existing_id, existing in self._skills.items():
            if existing.name == skill.name and existing_id != skill.id:
                raise ValueError(
                    f"A skill with name '{skill.name}' is already registered under id '{existing_id}'."
                )
        self._skills[skill.id] = skill

    def unregister(self, skill_id: str) -> None:
        """Remove the skill with *skill_id*. No-op if it is not registered."""
        self._skills.pop(skill_id, None)

    def get(self, skill_id: str) -> Skill:
        """Return the skill for *skill_id*. Raises ``KeyError`` if not found."""
        if skill_id not in self._skills:
            known = ", ".join(sorted(self._skills)) or "(none)"
            raise KeyError(f"No skill with id '{skill_id}'. Known ids: {known}")
        return self._skills[skill_id]

    def get_by_name(self, name: str) -> Skill:
        """Return the skill with *name*. Raises ``KeyError`` if not found."""
        for skill in self._skills.values():
            if skill.name == name:
                return skill
        raise KeyError(f"No skill with name '{name}'")

    def all_skills(self) -> list[Skill]:
        """Return all registered skills."""
        return list(self._skills.values())

    def load_from_dir(self, skills_dir: str | Path, skill_names: list[str] | None = None) -> None:
        """Scan *skills_dir* for SKILL.md files and register the results.

        When *skill_names* is given, only those subdirectories are loaded.
        Otherwise every subdirectory with a SKILL.md is loaded. Each skill is
        registered under id = its directory name (dev-time convenience path).
        """
        skills_dir = Path(skills_dir)
        if skill_names is None:
            skill_names = [p.name for p in skills_dir.iterdir() if p.is_dir()]
        for skill in load_skills(skill_names, skills_dir):
            if skill.id not in self._skills:
                self._skills[skill.id] = skill
