"""Shared REST client and interactive-loop helpers for CogBase example demos."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Awaitable, Callable

import httpx

try:
    import readline as _readline
    _READLINE_AVAILABLE = True
except ImportError:
    _READLINE_AVAILABLE = False

class GeneratorClient:
    """REST client for the /generate endpoints.

    Owns the full conversation history. The server is stateless — history is
    sent on every call. The LLM drives the conversation and embeds config
    proposals in its responses; this client extracts and tracks them.

    Lifecycle:
      chat(message)*  →  (display_text, config_yaml | None)
      deploy()        →  {name, status, error}
    """

    _CONFIG_START = "---CONFIG---"
    _CONFIG_END = "---END CONFIG---"

    def __init__(self, api_base: str, http_client: httpx.AsyncClient) -> None:
        self.api_base = api_base.rstrip("/")
        self._http = http_client
        self.messages: list[dict] = []   # full history sent to the server each turn
        self.config_yaml: str | None = None

    async def chat(self, user_message: str) -> tuple[str, str | None]:
        """Send a user message and return (display_text, config_yaml | None).

        ``display_text`` has CONFIG markers stripped — safe to print directly.
        The full server response (markers included) is stored in ``self.messages``
        so the LLM retains context of its previous proposals.
        """
        resp = await self._http.post(
            f"{self.api_base}/generate/chat",
            json={"text": user_message, "history": self.messages},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        full_content: str = data["content"]
        config_yaml: str | None = data.get("config_yaml")

        # Store the full history so the LLM sees its own proposals on the next turn.
        self.messages.append({"role": "user", "content": user_message})
        self.messages.append({"role": "assistant", "content": full_content})
        if config_yaml:
            self.config_yaml = config_yaml

        display_text = self._strip_config_block(full_content)
        return display_text, config_yaml

    async def deploy(self) -> dict:
        """Deploy the current config_yaml as a new application."""
        if not self.config_yaml:
            raise RuntimeError("No config to deploy — keep chatting until the LLM proposes one")
        resp = await self._http.post(
            f"{self.api_base}/generate/deploy",
            json={"config_yaml": self.config_yaml},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()

    def reset(self) -> None:
        self.messages = []
        self.config_yaml = None

    @classmethod
    def _strip_config_block(cls, text: str) -> str:
        if cls._CONFIG_START not in text:
            return text
        before, rest = text.split(cls._CONFIG_START, 1)
        after = rest.split(cls._CONFIG_END, 1)[1] if cls._CONFIG_END in rest else ""
        return (before.strip() + ("\n\n" + after.strip() if after.strip() else "")).strip()


_BUILTIN_COMMANDS = [
    "/q", "/quit", "/exit",
    "/list",
    "/create",
    "/delete",
    "/reset",
    "/list_collections",
    "/query_structured",
    "/clear",
]


def _install_completer(commands: list[str]) -> None:
    if not _READLINE_AVAILABLE:
        return

    def _completer(text: str, state: int) -> str | None:
        matches = [c for c in commands if c.startswith(text)]
        return matches[state] if state < len(matches) else None

    _readline.set_completer(_completer)
    # Keep / as a non-delimiter so the completer receives the full /cmd token.
    _readline.set_completer_delims(" \t\n")
    # libedit (macOS default) uses a different binding syntax than GNU readline.
    if "libedit" in getattr(_readline, "__doc__", ""):
        _readline.parse_and_bind("bind ^I rl_complete")
    else:
        _readline.parse_and_bind("tab: complete")

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s"


def configure_logging() -> None:
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=_LOG_FORMAT)


class CogBaseClient:
    def __init__(
        self,
        app_name: str,
        api_base: str,
        http_client: httpx.AsyncClient,
    ) -> None:
        self.app_name = app_name
        self.api_base = api_base.rstrip("/")
        self._http = http_client
        self._history: list[dict] = []

    async def list_apps(self) -> list[dict]:
        resp = await self._http.get(f"{self.api_base}/applications", timeout=10)
        resp.raise_for_status()
        return resp.json()["applications"]

    async def get_app(self) -> dict | None:
        resp = await self._http.get(
            f"{self.api_base}/applications/{self.app_name}", timeout=10
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    async def create_app(self, bundle: bytes) -> dict:
        resp = await self._http.post(
            f"{self.api_base}/applications",
            files={"bundle": ("bundle.zip", bundle, "application/zip")},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    async def delete_app(self, name: str | None = None) -> None:
        target = name or self.app_name
        resp = await self._http.delete(
            f"{self.api_base}/applications/{target}", timeout=10
        )
        if resp.status_code not in (204, 404):
            resp.raise_for_status()

    async def ingest_documents(
        self, documents: list[dict], timeout: float = 120
    ) -> list[dict]:
        resp = await self._http.post(
            f"{self.api_base}/applications/{self.app_name}/ingest_documents",
            json={"documents": documents, "concurrency": 3},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["results"]

    def clear_history(self) -> None:
        """Reset the chat history."""
        self._history = []

    async def query_stream(self, text: str) -> None:
        """Stream a query using accumulated chat history, printing tokens as they arrive."""
        answer_parts: list[str] = []
        async with self._http.stream(
            "POST",
            f"{self.api_base}/applications/{self.app_name}/query/stream",
            json={"text": text, "history": self._history},
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            print("Answer:\n")
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if "token" in data:
                    print(data["token"], end="", flush=True)
                    answer_parts.append(data["token"])
                elif "result" in data:
                    result = data["result"]
                    if result.get("passthrough") and result.get("structured_records"):
                        formatted = json.dumps(result["structured_records"], indent=2)
                        print(formatted)
                        answer_parts.append(formatted)
                elif "error" in data:
                    print(f"\n  ERROR: {data['error']}")
            print()
        answer = "".join(answer_parts)
        self._history.append({"role": "user", "content": text})
        self._history.append({"role": "assistant", "content": answer})

    async def list_collections(self) -> dict:
        """Returns {"structured": [...], "vector": [...]}."""
        resp = await self._http.get(
            f"{self.api_base}/applications/{self.app_name}/collections", timeout=10
        )
        resp.raise_for_status()
        return resp.json()

    async def query_structured(
        self,
        collection: str,
        filters: list[dict] | None = None,
    ) -> list[dict]:
        resp = await self._http.post(
            f"{self.api_base}/applications/{self.app_name}/collections/{collection}/query",
            json={"filters": filters or [], "fields": None},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["records"]


# ---------------------------------------------------------------------------
# Interactive-loop command helpers
# ---------------------------------------------------------------------------


async def cmd_startup(client: CogBaseClient, bundle: bytes) -> dict | None:
    """Get or create the app. Returns app_info, or None if creation failed."""
    app_info = await client.get_app()
    if app_info is None:
        print(f"Creating application '{client.app_name}'...")
        try:
            app_info = await client.create_app(bundle)
        except httpx.HTTPStatusError as exc:
            print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
            return None
        print(f"  status: {app_info['status']}")
        if app_info.get("error"):
            print(f"  error:  {app_info['error']}")
    else:
        print(f"Application '{client.app_name}' already exists (status: {app_info['status']})")
    return app_info


async def cmd_list(client: CogBaseClient) -> None:
    try:
        apps = await client.list_apps()
    except httpx.HTTPStatusError as exc:
        print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
        return
    if not apps:
        print("  No applications found.")
    else:
        for app in apps:
            print(f"  {app['name']:<28}  status: {app['status']}")


async def cmd_create(client: CogBaseClient, bundle: bytes) -> None:
    existing = await client.get_app()
    if existing is not None:
        print(f"  Application '{client.app_name}' already exists (status: {existing['status']})")
        return
    print(f"Creating application '{client.app_name}'...")
    try:
        result = await client.create_app(bundle)
    except httpx.HTTPStatusError as exc:
        print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
        return
    print(f"  status: {result['status']}")
    if result.get("error"):
        print(f"  error:  {result['error']}")


async def cmd_delete(client: CogBaseClient, name: str) -> None:
    if not name:
        print("  Usage: /delete <name>")
        return
    confirm = input(f"  Delete application '{name}' and all its data? [y/N] ").strip().lower()
    if confirm == "y":
        try:
            await client.delete_app(name)
        except httpx.HTTPStatusError as exc:
            print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
            return
        print(f"  Application '{name}' deleted.")


async def cmd_reset(client: CogBaseClient) -> bool:
    """Returns True if the app was deleted (caller should break the loop)."""
    confirm = input("  Delete application and all data? [y/N] ").strip().lower()
    if confirm == "y":
        await client.delete_app()
        print("  Application deleted. Restart the demo to start fresh.")
        return True
    return False


async def cmd_list_collections(client: CogBaseClient) -> None:
    try:
        cols = await client.list_collections()
    except httpx.HTTPStatusError as exc:
        print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
        return
    print(f"  structured: {cols.get('structured', [])}")
    print(f"  vector:     {cols.get('vector', [])}")


async def cmd_query_structured(
    client: CogBaseClient,
    collection: str,
    filters: list[dict] | None = None,
) -> None:
    print(f"Querying structured collection '{collection}'...")
    try:
        records = await client.query_structured(collection, filters)
    except httpx.HTTPStatusError as exc:
        print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
        return
    if not records:
        print("  No records found.")
    else:
        print(json.dumps(records, indent=2))


async def run_interactive_loop(
    client: CogBaseClient,
    build_bundle: Callable[[], bytes],
    *,
    default_collection: str = "",
    handler: Callable[[str, str], Awaitable[bool]] | None = None,
    extra_commands: list[str] | None = None,
) -> None:
    """Run the standard interactive command loop.

    All commands use a ``/`` prefix. Anything without a leading ``/`` is sent
    as a natural-language query. Pass *handler* for demo-specific commands; it
    receives ``(raw, lower)`` and returns ``True`` if it handled the input or
    ``False`` to fall through to a query.

    Common commands: /q /quit /exit, /list, /create, /delete <name>, /reset,
    /list_collections, /query_structured [<name>], /clear.
    """
    _install_completer(_BUILTIN_COMMANDS + (extra_commands or []))
    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not raw:
            continue

        lower = raw.lower()

        if lower in {"/q", "/quit", "/exit"}:
            print("Goodbye!")
            break

        if lower == "/list":
            await cmd_list(client)
            continue

        if lower == "/create":
            await cmd_create(client, build_bundle())
            continue

        if lower.startswith("/delete"):
            await cmd_delete(client, raw[len("/delete"):].strip())
            continue

        if lower == "/reset":
            if await cmd_reset(client):
                break
            continue

        if lower == "/list_collections":
            await cmd_list_collections(client)
            continue

        if lower == "/query_structured" or lower.startswith("/query_structured "):
            collection = (
                raw[len("/query_structured "):].strip()
                if lower.startswith("/query_structured ")
                else default_collection
            )
            if not collection:
                print("  Usage: /query_structured <collection>")
                continue
            await cmd_query_structured(client, collection)
            continue

        if lower == "/clear":
            client.clear_history()
            print("  Chat history cleared.")
            continue

        if handler is not None and await handler(raw, lower):
            continue

        print("Thinking...")
        try:
            await client.query_stream(raw)
        except httpx.HTTPStatusError as exc:
            print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
