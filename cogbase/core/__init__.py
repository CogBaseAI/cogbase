from cogbase.core.models import Chunk, Contradiction, Event, Fact
from cogbase.core.registry import SkillRegistry
from cogbase.core.session import Session
from cogbase.core.skill import Skill

__all__ = [
    "Application",
    "Chunk",
    "Contradiction",
    "Event",
    "Fact",
    "Session",
    "Skill",
    "SkillRegistry",
    "StructuredCollection",
    "VectorCollection",
]


def __getattr__(name: str):
    if name in ("Application", "StructuredCollection", "VectorCollection"):
        from cogbase.core.application import Application, StructuredCollection, VectorCollection  # noqa: F401
        return {"Application": Application, "StructuredCollection": StructuredCollection, "VectorCollection": VectorCollection}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
