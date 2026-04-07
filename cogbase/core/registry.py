"""Registry for looking up skills by name."""

from cogbase.core.skill import Skill


class SkillRegistry:
    """Maps skill names to skill classes.

    Skills are registered by class (not instance) because skills are stateless.
    The ``CogBase`` root object owns the registry instance.
    """

    def __init__(self) -> None:
        self._skills: dict[str, type[Skill]] = {}

    def register(self, skill_cls: type[Skill]) -> None:
        """Register a skill class. Raises ``ValueError`` if the name is already taken."""
        if skill_cls.name in self._skills:
            raise ValueError(
                f"A skill named '{skill_cls.name}' is already registered. "
                "Use a unique name or deregister the existing skill first."
            )
        self._skills[skill_cls.name] = skill_cls

    def get(self, name: str) -> type[Skill]:
        """Return the skill class for ``name``. Raises ``KeyError`` if not found."""
        if name not in self._skills:
            known = ", ".join(sorted(self._skills)) or "(none)"
            raise KeyError(f"No skill named '{name}'. Known skills: {known}")
        return self._skills[name]

    def all_skills(self) -> list[type[Skill]]:
        """Return all registered skill classes."""
        return list(self._skills.values())
