import textwrap
from pathlib import Path

import pytest

from cogbase.skills.skill import Skill
from cogbase.skills.registry import SkillRegistry


def _make_skill(name: str, description: str = "A skill.") -> Skill:
    return Skill(name=name, description=description, raw_markdown=f"# {name}\n")


def test_register_and_get():
    registry = SkillRegistry()
    skill = _make_skill("echo")
    registry.register(skill)
    assert registry.get("echo") is skill


def test_duplicate_registration_raises():
    registry = SkillRegistry()
    registry.register(_make_skill("echo"))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(_make_skill("echo"))


def test_get_unknown_raises():
    registry = SkillRegistry()
    with pytest.raises(KeyError, match="nonexistent"):
        registry.get("nonexistent")


def test_all_skills():
    registry = SkillRegistry()
    a, b = _make_skill("alpha"), _make_skill("beta")
    registry.register(a)
    registry.register(b)
    assert {s.name for s in registry.all_skills()} == {"alpha", "beta"}


def test_load_from_dir(tmp_path: Path):
    skill_dir = tmp_path / "weather"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        textwrap.dedent("""\
            ---
            name: weather
            description: Get weather.
            ---
            # Weather Skill
        """)
    )
    registry = SkillRegistry()
    registry.load_from_dir(tmp_path)
    skill = registry.get("weather")
    assert skill.name == "weather"


def test_load_from_dir_with_names(tmp_path: Path):
    for name in ("alpha", "beta"):
        d = tmp_path / name
        d.mkdir()
        (d / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: skill {name}.\n---\n# {name}\n"
        )
    registry = SkillRegistry()
    registry.load_from_dir(tmp_path, skill_names=["alpha"])
    registry.get("alpha")
    with pytest.raises(KeyError):
        registry.get("beta")
