"""Shared REST client and interactive-loop helpers for CogBase example demos."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
import sys
from collections.abc import Awaitable, Callable

import httpx

API_BASE = os.environ.get("COGBASE_API_URL", "http://localhost:8000")

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
      chat_stream(message)*  →  (display_text, config_yaml | None)
      chat(message)*         →  compatibility wrapper around chat_stream()
      deploy()        →  {name, status, error}
    """

    _CONFIG_START = "---CONFIG---"
    _CONFIG_END = "---END CONFIG---"

    def __init__(self, api_base: str, http_client: httpx.AsyncClient) -> None:
        self.api_base = api_base.rstrip("/")
        self._http = http_client
        self.messages: list[dict] = []   # full history sent to the server each turn
        self.config_yaml: str | None = None

    async def chat_stream(self, user_message: str) -> tuple[str, str | None]:
        """Send a user message via the streaming endpoint.

        ``display_text`` has CONFIG markers stripped — safe to print directly.
        The full server response (markers included) is stored in ``self.messages``
        so the LLM retains context of its previous proposals.
        """
        full_content = ""
        config_yaml: str | None = None

        async with self._http.stream(
            "POST",
            f"{self.api_base}/generate/chat/stream",
            json={"text": user_message, "history": self.messages},
            timeout=120,
        ) as resp:
            resp.raise_for_status()
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
                    full_content += data["token"]
                elif "result" in data:
                    result = data["result"]
                    full_content = result.get("content") or full_content
                    config_yaml = result.get("config_yaml")
                elif "error" in data:
                    raise RuntimeError(data["error"])

        print()

        # Store the full history so the LLM sees its own proposals on the next turn.
        self.messages.append({"role": "user", "content": user_message})
        self.messages.append({"role": "assistant", "content": full_content})
        if config_yaml:
            self.config_yaml = config_yaml

        display_text = self._strip_config_block(full_content)
        return display_text, config_yaml

    async def chat(self, user_message: str) -> tuple[str, str | None]:
        """Compatibility wrapper around ``chat_stream``."""
        return await self.chat_stream(user_message)

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
    "/list_apps",
    "/create_app",
    "/delete_app",
    "/use_app",
    "/ingest_file",
    "/query_app",
    "/clear_query_history",
    "/list_collections",
    "/query_structured_collection",
    "/q", "/quit", "/exit",
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
    def __init__(self, api_base: str = API_BASE) -> None:
        self.api_base = api_base.rstrip("/")
        self._http = httpx.AsyncClient()
        self.app_name = ""
        self._history: list[dict] = []

    async def __aenter__(self) -> "CogBaseClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self._http.aclose()

    def use_app(self, app_name: str) -> None:
        self.app_name = app_name
        self._history = []

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

    def clear_query_history(self) -> None:
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

    async def upload_documents(
        self,
        file_paths: list[pathlib.Path],
        metadata: dict | None = None,
        timeout: float = 120,
    ) -> list[dict]:
        """Upload files to the server for ingestion via POST /upload_documents.

        Files are parsed to markdown server-side.  *metadata* is applied to
        every file in the batch — use separate calls for per-file metadata.
        """
        files = [
            ("files", (p.name, p.read_bytes(), "application/octet-stream"))
            for p in file_paths
        ]
        resp = await self._http.post(
            f"{self.api_base}/applications/{self.app_name}/upload_documents",
            files=files,
            data={"metadata": json.dumps(metadata or {})},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["results"]

    async def query_structured_collection(
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


async def cmd_list_apps(client: CogBaseClient) -> None:
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


async def cmd_create_app(client: CogBaseClient, bundle: bytes) -> None:
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


async def cmd_delete_app(client: CogBaseClient, name: str) -> None:
    if not name:
        print("  Usage: /delete_app <name>")
        return
    confirm = input(f"  Delete application '{name}' and all its data? [y/N] ").strip().lower()
    if confirm == "y":
        try:
            await client.delete_app(name)
        except httpx.HTTPStatusError as exc:
            print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
            return
        print(f"  Application '{name}' deleted.")


async def cmd_list_collections(client: CogBaseClient) -> None:
    try:
        cols = await client.list_collections()
    except httpx.HTTPStatusError as exc:
        print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
        return
    print(f"  structured: {cols.get('structured', [])}")
    print(f"  vector:     {cols.get('vector', [])}")


async def cmd_query_structured_collection(
    client: CogBaseClient,
    collection: str,
    filters: list[dict] | None = None,
) -> None:
    print(f"Querying structured collection '{collection}'...")
    try:
        records = await client.query_structured_collection(collection, filters)
    except httpx.HTTPStatusError as exc:
        print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
        return
    if not records:
        print("  No records found.")
    else:
        print(json.dumps(records, indent=2))


def cmd_select(client: CogBaseClient, name: str) -> None:
    client.use_app(name)
    print(f"  Using '{client.app_name}'")


async def cmd_ingest_file(
    client: CogBaseClient,
    file_paths: list[pathlib.Path],
    metadata: dict | None = None,
    timeout: float = 120,
) -> None:
    """Upload and ingest one or more files via POST /upload_documents."""
    print(f"Uploading {len(file_paths)} file(s)...")
    try:
        results = await client.upload_documents(file_paths, metadata=metadata, timeout=timeout)
    except httpx.HTTPStatusError as exc:
        print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
        return
    for r in results:
        if r["success"]:
            print(f"  {r['doc_id']:<30}  OK  ({r['records_extracted']} records extracted)")
        else:
            print(f"  {r['doc_id']:<30}  FAILED: {r['error']}")


async def run_interactive_loop(
    client: CogBaseClient,
    build_bundle: Callable[[], bytes] | None = None,
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

    Common commands:
      /list_apps, /create_app, /delete_app <name>, /use_app <name>,
      /ingest_file <path>, /query_app <text>, /clear_query_history,
      /list_collections, /query_structured_collection [<name>], /q /quit /exit.
    """
    commands = _BUILTIN_COMMANDS + (extra_commands or [])
    print("Common commands:")
    for command in _BUILTIN_COMMANDS:
        print(f"  {command}")
    print("Extra commands:")
    for command in extra_commands:
        print(f"  {command}")
    print()

    _install_completer(commands)
    while True:
        try:
            raw = input(f"[{client.app_name}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not raw:
            continue

        lower = raw.lower()

        if lower in {"/q", "/quit", "/exit"}:
            print("Goodbye!")
            break

        if lower == "/use_app" or lower.startswith("/use_app "):
            name = raw[len("/use_app"):].strip()
            if not name:
                print("  Usage: /use_app <app-name>")
            else:
                cmd_use_app(client, name)
            continue

        if lower == "/list_apps":
            await cmd_list_apps(client)
            continue

        if lower == "/create_app":
            if build_bundle is None:
                print("  /create_app is not available in standalone mode.")
            else:
                await cmd_create_app(client, build_bundle())
            continue

        if lower.startswith("/delete_app"):
            await cmd_delete_app(client, raw[len("/delete_app"):].strip())
            continue

        if lower == "/list_collections":
            await cmd_list_collections(client)
            continue

        if lower == "/query_structured_collection" or lower.startswith("/query_structured_collection "):
            collection = (
                raw[len("/query_structured_collection "):].strip()
                if lower.startswith("/query_structured_collection ")
                else default_collection
            )
            if not collection:
                print("  Usage: /query_structured_collection <collection>")
                continue
            await cmd_query_structured_collection(client, collection)
            continue

        if lower.startswith("/ingest_file"):
            rest = raw[len("/ingest_file"):].strip()
            if not rest:
                print("  Usage: /ingest_file <path> [<path2> ...]")
                continue
            paths = [pathlib.Path(p).expanduser() for p in rest.split()]
            paths = [p if p.is_absolute() else pathlib.Path.cwd() / p for p in paths]
            missing = [p for p in paths if not p.exists()]
            for p in missing:
                print(f"  File not found: {p}")
            if missing:
                continue
            await cmd_ingest_file(client, paths)
            continue

        if lower == "/clear_query_history":
            client.clear_query_history()
            print("  Chat history cleared.")
            continue

        if lower.startswith("/query_app"):
            text = raw[len("/query_app"):].strip()
            if not text:
                print("  Usage: /query_app <text>")
                continue
            print("Thinking...")
            try:
                await client.query_stream(text)
            except httpx.HTTPStatusError as exc:
                print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
            continue

        if handler is not None and await handler(raw, lower):
            continue

        if not raw.startswith("/"):
            print("  Use /query_app <text> to query, or /q to quit.")


async def _standalone_main() -> None:
    configure_logging()
    print()
    print("CogBase Interactive Client")
    print("=" * 40)
    print(f"  api: {API_BASE}")
    print()

    async with CogBaseClient() as client:
        try:
            apps = await client.list_apps()
        except Exception as exc:
            print(f"  Could not reach server: {exc}")
            return

        app_name = ""
        if apps:
            print("Available applications:")
            for i, app in enumerate(apps, 1):
                print(f"  {i}. {app['name']:<28}  status: {app['status']}")
            choice = input("Select app (number or name, blank to skip): ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(apps):
                    app_name = apps[idx]["name"]
            elif any(a["name"] == choice for a in apps):
                app_name = choice
        else:
            print("  No applications found.")

        if not app_name:
            app_name = input("Enter app name (blank to exit): ").strip()
        if not app_name:
            return

        client.use_app(app_name)
        print(f"\nConnected to '{app_name}'. Use /query_app <text> to query, /use_app to switch, /q to quit.")
        print()
        await run_interactive_loop(client)


if __name__ == "__main__":
    asyncio.run(_standalone_main())
