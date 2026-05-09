"""CogBase App Generator — conversational, LLM-driven app creation.

The LLM asks what it needs, proposes config sections as they become clear,
and refines on feedback. Just describe what you want and keep chatting.

Usage
-----
    # Start the API server first:
    uvicorn api.main:app --reload

    # Then run the generator:
    cd /path/to/cogbase
    python examples/app_generator.py

Set COGBASE_API_URL to override the default http://localhost:8000.

Commands
--------
    /preview    print the current config.yaml (available once the LLM proposes one)
    /save       save config.yaml to disk  (/save <path> for a custom path)
    /deploy     deploy the current config as a live application
    /restart    clear history and start a new conversation
    /q          quit
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).parent.parent.resolve()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import httpx  # noqa: E402

from examples.cogbase_client import (  # noqa: E402
    CogBaseClient,
    GeneratorClient,
    configure_logging,
    run_interactive_loop,
)

configure_logging()

_API_BASE = os.environ.get("COGBASE_API_URL", "http://localhost:8000")
_COMMANDS = ["/preview", "/save", "/deploy", "/restart", "/q", "/quit", "/exit"]

# ---------------------------------------------------------------------------
# readline tab completion (best-effort)
# ---------------------------------------------------------------------------

try:
    import readline as _readline
    _READLINE = True
except ImportError:
    _READLINE = False


def _setup_completion() -> None:
    if not _READLINE:
        return

    def _completer(text: str, state: int) -> str | None:
        matches = [c for c in _COMMANDS if c.startswith(text)]
        return matches[state] if state < len(matches) else None

    _readline.set_completer(_completer)
    _readline.set_completer_delims(" \t\n")
    if "libedit" in getattr(_readline, "__doc__", ""):
        _readline.parse_and_bind("bind ^I rl_complete")
    else:
        _readline.parse_and_bind("tab: complete")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    print()
    print("CogBase App Generator")
    print("=" * 40)
    print(f"  api: {_API_BASE}")
    print()
    print("Describe the app you want to build and the LLM will guide you from there.")
    print("Commands: /preview | /save [path] | /deploy | /restart | /q")
    print()

    _setup_completion()

    async with httpx.AsyncClient() as http:
        gen = GeneratorClient(_API_BASE, http)

        while True:
            try:
                raw = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                return

            if not raw:
                continue

            lower = raw.lower()

            if lower in {"/q", "/quit", "/exit"}:
                print("Goodbye!")
                return

            if lower == "/restart":
                gen.reset()
                print()
                print("Conversation cleared. Describe the app you want to build.")
                print()
                continue

            if lower == "/preview":
                if not gen.config_yaml:
                    print("  No config yet — keep chatting until the LLM proposes one.")
                else:
                    print()
                    print(gen.config_yaml)
                    print()
                continue

            if lower == "/save" or lower.startswith("/save "):
                if not gen.config_yaml:
                    print("  No config yet — keep chatting until the LLM proposes one.")
                    continue
                path_str = raw[len("/save"):].strip()
                if not path_str:
                    # derive a filename from the app name in the config
                    import yaml as _yaml
                    try:
                        name = (_yaml.safe_load(gen.config_yaml) or {}).get("name", "app")
                    except Exception:
                        name = "app"
                    path_str = f"{name}-config.yaml"
                path = pathlib.Path(path_str).expanduser()
                path.write_text(gen.config_yaml, encoding="utf-8")
                print(f"  Saved to {path}")
                continue

            if lower == "/deploy":
                if not gen.config_yaml:
                    print("  No config yet — keep chatting until the LLM proposes one.")
                    continue
                import yaml as _yaml
                try:
                    app_name = (_yaml.safe_load(gen.config_yaml) or {}).get("name", "app")
                except Exception:
                    app_name = "app"
                print(f"\nDeploying {app_name}...")
                try:
                    result = await gen.deploy()
                except httpx.HTTPStatusError as exc:
                    print(f"  ERROR {exc.response.status_code}: {exc.response.text}")
                    continue
                print(f"  status: {result['status']}")
                if result.get("error"):
                    print(f"  error:  {result['error']}")
                    continue

                deployed_name: str = result["name"]
                print()
                try:
                    launch = input("  Launch query loop? [y/N] ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print()
                    return
                if launch == "y":
                    print()
                    client = CogBaseClient(deployed_name, _API_BASE, http)
                    print("Commands: /list | /ingest <file> | /list_collections | /query_structured | /clear | /reset | /q")
                    print()
                    await run_interactive_loop(client, lambda: b"")
                return

            # Natural-language message — send to the LLM
            try:
                display_text, config_yaml = await gen.chat(raw)
            except httpx.HTTPStatusError as exc:
                print(f"  ERROR {exc.response.status_code}: {exc.response.text}")
                continue

            print()
            print(display_text)
            if config_yaml:
                print("\n  (config updated — /preview to inspect, /deploy when ready)")
            print()


if __name__ == "__main__":
    asyncio.run(main())
