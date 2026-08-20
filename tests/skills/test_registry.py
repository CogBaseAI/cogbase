import textwrap
from pathlib import Path

import pytest

from cogbase.skills.skill import Skill
from cogbase.skills.registry import SkillRegistry


def _make_skill(name: str, description: str = "A skill.", skill_id: str | None = None) -> Skill:
    return Skill(name=name, description=description, raw_markdown=f"# {name}\n", id=skill_id or name)


def test_register_and_get():
    registry = SkillRegistry()
    skill = _make_skill("echo")
    registry.register(skill)
    assert registry.get("echo") is skill


def test_register_without_id_raises():
    registry = SkillRegistry()
    with pytest.raises(ValueError, match="without an id"):
        registry.register(Skill(name="echo", description="d", raw_markdown="# echo"))


def test_duplicate_registration_raises():
    registry = SkillRegistry()
    registry.register(_make_skill("echo"))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(_make_skill("echo"))


def test_register_replace_overwrites():
    registry = SkillRegistry()
    registry.register(_make_skill("echo", description="v1"))
    registry.register(_make_skill("echo", description="v2"), replace=True)
    assert registry.get("echo").description == "v2"


def test_unregister():
    registry = SkillRegistry()
    registry.register(_make_skill("echo"))
    registry.unregister("echo")
    with pytest.raises(KeyError):
        registry.get("echo")
    registry.unregister("echo")  # idempotent — no raise


def test_get_by_id_not_name():
    registry = SkillRegistry()
    registry.register(_make_skill("display-name", skill_id="uuid-123"))
    assert registry.get("uuid-123").name == "display-name"
    with pytest.raises(KeyError):
        registry.get("display-name")


def test_get_unknown_raises():
    registry = SkillRegistry()
    with pytest.raises(KeyError, match="nonexistent"):
        registry.get("nonexistent")


def test_get_without_account_id_is_account_agnostic():
    registry = SkillRegistry()
    registry.register(_make_skill("echo"), account_id="account-a")
    # No account_id passed: today's behavior for every existing caller.
    assert registry.get("echo").name == "echo"


def test_get_with_owning_account_id_succeeds():
    registry = SkillRegistry()
    registry.register(_make_skill("echo"), account_id="account-a")
    assert registry.get("echo", "account-a").name == "echo"


def test_get_with_global_builtin_succeeds_for_any_account():
    registry = SkillRegistry()
    registry.register(_make_skill("echo"))  # account_id=None -> global builtin
    assert registry.get("echo", "some-account").name == "echo"


def test_get_with_a_different_accounts_id_raises():
    # This is the cross-tenant hole: an id is globally unique, so knowing/
    # guessing another account's skill id must not resolve it when the caller
    # passes its own account_id.
    registry = SkillRegistry()
    registry.register(_make_skill("echo"), account_id="account-a")
    with pytest.raises(KeyError):
        registry.get("echo", "account-b")


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
