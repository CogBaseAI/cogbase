"""Abstract base class for all CogBase skills.

Aligned with the AgentSkills specification (https://agentskills.io/specification).
"""

import abc
from typing import ClassVar

from cogbase.core.session import Session


class Skill(abc.ABC):
    """All skills share this interface.

    Required class variables (AgentSkills spec):
    - ``name``: unique identifier, max 64 chars, lowercase alphanumeric + hyphens
    - ``description``: what the skill does and when to use it, max 1024 chars

    Optional class variables (AgentSkills spec):
    - ``compatibility``: environment requirements (e.g. "Requires Python 3.11+")
    - ``metadata``: arbitrary str→str key-value pairs for additional context
    - ``allowed_tools``: tools this skill is permitted to invoke

    Document expected inputs and outputs in the class docstring or a SKILL.md
    file alongside the implementation.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    compatibility: ClassVar[str]
    metadata: ClassVar[dict[str, str]]
    allowed_tools: ClassVar[list[str]]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        # Skip the check for abstract intermediaries.
        if abc.ABC in cls.__bases__:
            return
        missing = [attr for attr in ("name", "description") if not hasattr(cls, attr)]
        if missing:
            raise TypeError(
                f"{cls.__name__} must define class variables: {', '.join(missing)}"
            )
        _validate_name(cls.name, cls.__name__)
        _validate_description(cls.description, cls.__name__)

    @abc.abstractmethod
    def run(self, input: dict, session: Session) -> dict:
        """Execute the skill and return a result dict."""


def _validate_name(name: str, cls_name: str) -> None:
    import re
    if not name:
        raise TypeError(f"{cls_name}.name must not be empty")
    if len(name) > 64:
        raise TypeError(f"{cls_name}.name exceeds 64 characters")
    if not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", name):
        raise TypeError(
            f"{cls_name}.name '{name}' is invalid: use lowercase alphanumeric and "
            "hyphens only, no leading/trailing/consecutive hyphens"
        )


def _validate_description(description: str, cls_name: str) -> None:
    if not description:
        raise TypeError(f"{cls_name}.description must not be empty")
    if len(description) > 1024:
        raise TypeError(f"{cls_name}.description exceeds 1024 characters")
