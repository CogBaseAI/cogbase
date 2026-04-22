"""Abstract base class for all CogBase tools."""

import abc
from typing import ClassVar

from cogbase.core.session import Session


class Tool(abc.ABC):
    """Pipeline-level operation that transforms or stores document data.

    Tools are the composable units of the ingestion layer — each tool performs
    one focused operation (chunk+embed+upsert, extract, summarise, …).  Unlike
    Skills, tools are async (pipeline I/O is always async) and are instantiated
    with their dependencies via constructor injection.

    Required class variables:
    - ``name``: unique identifier, max 64 chars, lowercase alphanumeric + hyphens
    - ``description``: what the tool does, max 1024 chars
    """

    name: ClassVar[str]
    description: ClassVar[str]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
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
    async def run(self, input: dict, session: Session) -> dict:
        """Execute the tool and return a result dict.

        Args:
            input:   Tool-specific input parameters (see each tool's docstring).
            session: Active session for logging and correlation.

        Returns:
            Tool-specific result dict (see each tool's docstring).
        """


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
