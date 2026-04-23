"""Registry for looking up tools by name and listing built-in tools."""

from __future__ import annotations

import re
from dataclasses import dataclass

from cogbase.llms.base import SystemTool


@dataclass(frozen=True)
class ToolInfo:
    """Descriptor returned by the list API."""

    name: str
    description: str
    builtin: bool


class ToolRegistry:
    """Maps tool names to ``SystemTool`` instances.

    Tools are registered as instances because they carry injected dependencies
    (stores, embedders, chunkers).  Built-in tools are flagged so callers can
    distinguish framework-provided tools from user-registered ones.
    """

    def __init__(self) -> None:
        self._tools: dict[str, SystemTool] = {}
        self._builtin: set[str] = set()

    def register(self, tool: SystemTool, *, builtin: bool = False) -> None:
        """Register a tool instance.

        Args:
            tool:    ``SystemTool`` to register.
            builtin: When ``True`` the tool appears in ``list_builtin_tools()``.

        Raises:
            ValueError: If a tool with the same name is already registered, or
                        if the name/description fail validation.
        """
        _validate_name(tool.name)
        _validate_description(tool.definition["description"])
        if tool.name in self._tools:
            raise ValueError(
                f"A tool named '{tool.name}' is already registered. "
                "Use a unique name or deregister the existing tool first."
            )
        self._tools[tool.name] = tool
        if builtin:
            self._builtin.add(tool.name)

    def deregister(self, name: str) -> None:
        """Remove a tool by name. No-op if not registered."""
        self._tools.pop(name, None)
        self._builtin.discard(name)

    def get(self, name: str) -> SystemTool:
        """Return the tool for *name*.

        Raises:
            KeyError: If no tool with that name is registered.
        """
        if name not in self._tools:
            known = ", ".join(sorted(self._tools)) or "(none)"
            raise KeyError(f"No tool named '{name}'. Known tools: {known}")
        return self._tools[name]

    def list_tools(self) -> list[ToolInfo]:
        """Return descriptors for every registered tool, sorted by name."""
        return sorted(
            (
                ToolInfo(
                    name=t.name,
                    description=t.definition["description"],
                    builtin=t.name in self._builtin,
                )
                for t in self._tools.values()
            ),
            key=lambda ti: ti.name,
        )

    def list_builtin_tools(self) -> list[ToolInfo]:
        """Return descriptors for built-in tools only, sorted by name."""
        return [ti for ti in self.list_tools() if ti.builtin]


def _validate_name(name: str) -> None:
    if not name:
        raise ValueError("tool name must not be empty")
    if len(name) > 64:
        raise ValueError(f"tool name '{name}' exceeds 64 characters")
    if not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", name):
        raise ValueError(
            f"tool name '{name}' is invalid: use lowercase alphanumeric and "
            "hyphens only, no leading/trailing/consecutive hyphens"
        )


def _validate_description(description: str) -> None:
    if not description:
        raise ValueError("tool description must not be empty")
    if len(description) > 1024:
        raise ValueError("tool description exceeds 1024 characters")
