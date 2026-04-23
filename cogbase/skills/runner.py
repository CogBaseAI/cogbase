"""SkillRunner — selects a skill and drives the LLM agent loop."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import tempfile
from collections.abc import AsyncGenerator

from cogbase.llms.base import ChatMessage, LLMBase, ToolDefinition
from cogbase.skills.skill import Skill

logger = logging.getLogger(__name__)

BASE_TOOLS: list[ToolDefinition] = [
    {
        "name": "python",
        "description": (
            "Execute inline Python code and return stdout/stderr. "
            "Use for computation, data processing, or logic that does not need a separate script file."
        ),
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python source code to execute"}},
            "required": ["code"],
            "additionalProperties": False,
        },
    },
    {
        "name": "shell",
        "description": (
            "Run a bash command and return stdout/stderr. "
            "Use whenever the active skill instructs you to run a command, especially "
            "lines like 'python <script_path> ...' — those are shell commands, not inline code."
        ),
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "A bash command to execute"}},
            "required": ["command"],
            "additionalProperties": False,
        },
    },
]

_TOOL_TIMEOUT = 30  # seconds


class SkillRunner:
    """Drives the select → prompt → tool-call loop for a skill.

    Args:
        llm:       LLM backend used for skill selection and execution.
        max_calls: Maximum tool-call iterations before giving up.
    """

    def __init__(self, llm: LLMBase, max_calls: int = 10) -> None:
        self._llm = llm
        self._max_calls = max_calls

    async def select(
        self,
        skills: list[Skill],
        user_input: str,
        history: list[ChatMessage] | None = None,
    ) -> Skill | None:
        """Ask the LLM to pick the best skill for *user_input*; returns None if none apply."""
        if not skills:
            return None

        skill_list = "\n".join(
            f"{i + 1}. name={s.name!r}  description={s.description!r}"
            for i, s in enumerate(skills)
        )
        history_text = "\n".join(
            f"[{m['role']}] {m.get('content', '')}" for m in (history or [])
        )

        messages: list[ChatMessage] = [
            {
                "role": "system",
                "content": (
                    "You are a skill router. Given the conversation history, current user question, "
                    "and available skills, return the name of the single most relevant skill, "
                    "or 'none' if no skill applies. "
                    "Output only the skill name or 'none' — no explanation, no punctuation."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Conversation history:\n{history_text}\n\n"
                    f"Current user question:\n{user_input}\n\n"
                    f"Available skills:\n{skill_list}"
                ),
            },
        ]

        result = await self._llm.complete(messages)
        chosen = (result["content"] or "").lower().strip("'\"")

        if chosen == "none":
            logger.info("[skills] no skill selected for: %s", user_input[:100])
            return None

        for skill in skills:
            if skill.name.lower() == chosen:
                logger.info("[skills] selected skill '%s'", skill.name)
                return skill

        logger.error("[skills] router returned unknown skill '%s', ignoring", chosen)
        return None

    def build_system_prompt(
        self,
        base_prompt: str,
        skill: Skill,
        runtime_context: dict | None = None,
    ) -> str:
        """Merge *base_prompt* with the skill's markdown and optional *runtime_context* key-value pairs."""
        base_dir = str(skill.source_path.parent) if skill.source_path else ""
        metadata_block = ""
        if skill.metadata:
            metadata_block = (
                "Skill metadata:\n"
                f"```json\n{json.dumps(skill.metadata, ensure_ascii=False, indent=2)}\n```\n\n"
            )

        context_block = ""
        if runtime_context:
            lines = "\n".join(f"{k}: `{v}`" for k, v in runtime_context.items())
            context_block = f"\n\n## Runtime Context\n\n{lines}\n"

        return (
            f"{base_prompt}\n\n"
            f"## Active Skill: {skill.name}\n\n"
            + (f"Skill base directory: `{base_dir}`\n\n" if base_dir else "")
            + metadata_block
            + "Follow the skill's instructions below to complete the user's request. "
            "Use the `shell` tool to run any commands it suggests.\n\n"
            + skill.raw_markdown
            + context_block
        )

    async def run(
        self,
        skill: Skill,
        user_input: str,
        history: list[ChatMessage] | None = None,
        base_prompt: str = "You are a helpful assistant.",
        runtime_context: dict | None = None,
    ) -> AsyncGenerator[str, None]:
        """Drive the tool-call loop for *skill* and yield text chunks.

        Yields status strings during tool execution and the final LLM response.
        """
        system_prompt = self.build_system_prompt(base_prompt, skill, runtime_context)
        messages: list[ChatMessage] = [{"role": "system", "content": system_prompt}]
        messages.extend(history or [])
        messages.append({"role": "user", "content": user_input})

        call_count = 0
        while call_count < self._max_calls:
            result = await self._llm.complete(messages, tools=BASE_TOOLS)

            tool_calls = result.get("tool_calls")
            if not tool_calls:
                yield result.get("content") or ""
                return

            tool_names = ", ".join(tc["name"] for tc in tool_calls)
            logger.info("[skills] tool_calls: %s", tool_names)
            yield f"Executing: {tool_names}..."

            messages.append({
                "role": "assistant",
                "content": result.get("content"),
                "tool_calls": tool_calls,
            })

            call_count += 1
            for tc in tool_calls:
                inputs: dict = {}
                try:
                    inputs = json.loads(tc["arguments"])
                except json.JSONDecodeError:
                    pass

                logger.info("[skills] execute_tool %s(%s)", tc["name"], json.dumps(inputs)[:300])
                output = await self._execute_tool(tc["name"], inputs, skill)
                logger.info("[skills] execute_tool done: %s", output[:300])

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": output,
                })

        logger.error("[skills] max tool calls (%d) reached. skill=%s", self._max_calls, skill.name)
        yield (
            "I was unable to complete your request within the allowed number of steps. "
            "Please try a simpler or more specific request."
        )

    async def _execute_tool(self, name: str, inputs: dict, skill: Skill | None = None) -> str:
        env = self._tool_env(skill)

        if name == "python":
            return await self._run_python(inputs.get("code", ""), env)
        if name == "shell":
            return await self._run_shell(inputs.get("command", ""), env)
        return f"Unknown tool: {name}"

    async def _run_python(self, code: str, env: dict) -> str:
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                tmp = f.name
            try:
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, tmp,
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                return await self._read_proc(proc)
            finally:
                os.unlink(tmp)
        except Exception as e:
            return f"Python error: {e}"

    async def _run_shell(self, command: str, env: dict) -> str:
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            return await self._read_proc(proc)
        except Exception as e:
            return f"Shell error: {e}"

    @staticmethod
    async def _read_proc(proc: asyncio.subprocess.Process) -> str:
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=_TOOL_TIMEOUT)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return "Process timed out"
        return stdout.decode().strip() or stderr.decode().strip() or "(no output)"

    @staticmethod
    def _tool_env(skill: Skill | None) -> dict:
        env = os.environ.copy()
        if skill and skill.site_packages:
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"{skill.site_packages}:{existing}" if existing else skill.site_packages
        return env

    async def compact_messages(
        self,
        system_prompt: str,
        messages: list[ChatMessage],
    ) -> list[ChatMessage]:
        """Summarise *messages* into a minimal list to recover from context overflow."""
        transcript = "\n".join(
            f"[{m['role']}] {str(m.get('content', ''))[:500]}"
            for m in messages
        )
        result = await self._llm.complete([
            {
                "role": "user",
                "content": (
                    "Compress the following conversation transcript into a concise bullet-point "
                    "summary preserving all key decisions, tool outputs, and conclusions. Be terse.\n\n"
                    + transcript
                ),
            }
        ])
        summary = result.get("content") or "(empty summary)"
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Compacted context:\n\n{summary}\n\nContinue from this point."},
        ]
