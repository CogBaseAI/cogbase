"""Registry for looking up skills by name."""

from __future__ import annotations

from pathlib import Path

from cogbase.skills.skill import Skill, load_skills


class SkillRegistry:
    """Maps skill names to loaded Skill instances."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        """Register a skill. Raises ``ValueError`` if the name is already taken."""
        if skill.name in self._skills:
            raise ValueError(
                f"A skill named '{skill.name}' is already registered. "
                "Use a unique name or deregister the existing skill first."
            )
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill:
        """Return the skill for ``name``. Raises ``KeyError`` if not found."""
        if name not in self._skills:
            known = ", ".join(sorted(self._skills)) or "(none)"
            raise KeyError(f"No skill named '{name}'. Known skills: {known}")
        return self._skills[name]

    def all_skills(self) -> list[Skill]:
        """Return all registered skills."""
        return list(self._skills.values())

    def load_from_dir(self, skills_dir: str | Path, skill_names: list[str] | None = None) -> None:
        """Scan *skills_dir* for SKILL.md files and register the results.

        When *skill_names* is given, only those subdirectories are loaded.
        Otherwise every subdirectory with a SKILL.md is loaded.
        """
        skills_dir = Path(skills_dir)
        if skill_names is None:
            skill_names = [p.name for p in skills_dir.iterdir() if p.is_dir()]
        for skill in load_skills(skill_names, skills_dir):
            if skill.name not in self._skills:
                self._skills[skill.name] = skill
