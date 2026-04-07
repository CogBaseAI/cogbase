import pytest

from cogbase.core.registry import SkillRegistry
from cogbase.core.session import Session
from cogbase.core.skill import Skill


class EchoSkill(Skill):
    name = "echo"
    description = "Returns the input unchanged. Use when you need to pass data through without modification."

    def run(self, input: dict, session: Session) -> dict:
        return input


class SumSkill(Skill):
    name = "sum"
    description = "Sums two numbers. Use when you need to add numeric values together."

    def run(self, input: dict, session: Session) -> dict:
        return {"result": input["a"] + input["b"]}


def test_register_and_get():
    registry = SkillRegistry()
    registry.register(EchoSkill)
    assert registry.get("echo") is EchoSkill


def test_duplicate_registration_raises():
    registry = SkillRegistry()
    registry.register(EchoSkill)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(EchoSkill)


def test_get_unknown_raises():
    registry = SkillRegistry()
    with pytest.raises(KeyError, match="nonexistent"):
        registry.get("nonexistent")


def test_all_skills():
    registry = SkillRegistry()
    registry.register(EchoSkill)
    registry.register(SumSkill)
    assert set(registry.all_skills()) == {EchoSkill, SumSkill}


def test_name_validation_rejects_uppercase():
    with pytest.raises(TypeError, match="invalid"):
        class BadSkill(Skill):
            name = "EchoSkill"
            description = "Bad name."
            def run(self, input: dict, session: Session) -> dict: return {}


def test_name_validation_rejects_consecutive_hyphens():
    with pytest.raises(TypeError, match="invalid"):
        class BadSkill(Skill):
            name = "echo--skill"
            description = "Bad name."
            def run(self, input: dict, session: Session) -> dict: return {}


def test_description_too_long_raises():
    with pytest.raises(TypeError, match="1024"):
        class BadSkill(Skill):
            name = "bad-skill"
            description = "x" * 1025
            def run(self, input: dict, session: Session) -> dict: return {}


def test_optional_fields_not_required():
    class MinimalSkill(Skill):
        name = "minimal"
        description = "A minimal skill with only required fields."
        def run(self, input: dict, session: Session) -> dict: return {}

    assert not hasattr(MinimalSkill, "compatibility")
    assert not hasattr(MinimalSkill, "metadata")
    assert not hasattr(MinimalSkill, "allowed_tools")
