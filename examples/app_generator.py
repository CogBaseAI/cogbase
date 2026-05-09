"""CogBase App Generator — describe, refine, and deploy an app interactively.

Usage
-----
    # Start the API server first:
    uvicorn api.main:app --reload

    # Then run the generator:
    cd /path/to/cogbase
    python examples/app_generator.py

Set COGBASE_API_URL to override the default http://localhost:8000.

Phases
------
    1. Describe — type a plain-language description of the app you want
    2. Revise   — refine the generated config with natural-language feedback
    3. Deploy   — deploy the config as a live application

Commands (revision phase)
-------------------------
    /preview            Print the full generated config.yaml
    /save [path]        Save config.yaml to disk (default: <app-name>-config.yaml)
    /deploy             Deploy the current config
    /restart            Start over with a new description
    /q /quit /exit      Exit
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
_REVISION_COMMANDS = ["/preview", "/save", "/deploy", "/restart", "/q", "/quit", "/exit"]

# ---------------------------------------------------------------------------
# readline tab completion (best-effort)
# ---------------------------------------------------------------------------

try:
    import readline as _readline
    _READLINE = True
except ImportError:
    _READLINE = False


def _setup_completion(commands: list[str]) -> None:
    if not _READLINE:
        return

    def _completer(text: str, state: int) -> str | None:
        matches = [c for c in commands if c.startswith(text)]
        return matches[state] if state < len(matches) else None

    _readline.set_completer(_completer)
    _readline.set_completer_delims(" \t\n")
    if "libedit" in getattr(_readline, "__doc__", ""):
        _readline.parse_and_bind("bind ^I rl_complete")
    else:
        _readline.parse_and_bind("tab: complete")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

_SEP = "─" * 62


def _print_summary(summary: dict) -> None:
    name = summary.get("name", "?")
    print(f"\nGenerated: {name}")
    print(_SEP)
    vc = summary.get("vector_collections", [])
    if vc:
        print(f"  vector collections  : {', '.join(vc)}")
    for sc in summary.get("structured_collections", []):
        fields: list[str] = sc.get("fields", [])
        shown = fields[:6]
        suffix = ", ..." if len(fields) > 6 else ""
        print(f"  structured schema   : {sc['name']}")
        if shown:
            print(f"    fields: {', '.join(shown)}{suffix}")
    steps: list[str] = summary.get("pipeline_steps", [])
    if steps:
        print(f"  pipeline steps      : {' → '.join(steps)}")
    print(_SEP)
    print()


def _print_changes(changes: list[str]) -> None:
    if not changes:
        return
    print()
    for c in changes:
        print(f"  {c}")
    print()


def _print_revision_help() -> None:
    print("Type feedback to revise, or a command:")
    print("  /preview    show the full generated config.yaml")
    print("  /save       save config.yaml to disk  (/save <path> for a custom path)")
    print("  /deploy     deploy this app")
    print("  /restart    describe a new app")
    print("  /q          quit")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    print()
    print("CogBase App Generator")
    print("=" * 40)
    print(f"  api: {_API_BASE}")
    print()

    async with httpx.AsyncClient() as http:
        gen = GeneratorClient(_API_BASE, http)

        while True:
            # ------------------------------------------------------------------
            # Phase 1: describe
            # ------------------------------------------------------------------
            print("Describe the app you want to build.")
            print("Include: document types, facts to extract, and example questions.\n")

            try:
                description = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                return

            if not description:
                continue
            if description.lower() in {"/q", "/quit", "/exit"}:
                print("Goodbye!")
                return

            print("\nGenerating...")
            try:
                gen_data = await gen.generate(description)
            except httpx.HTTPStatusError as exc:
                print(f"  ERROR {exc.response.status_code}: {exc.response.text}")
                continue

            current_summary: dict = gen_data["summary"]
            _print_summary(current_summary)
            _print_revision_help()

            # ------------------------------------------------------------------
            # Phase 2: revise
            # ------------------------------------------------------------------
            _setup_completion(_REVISION_COMMANDS)
            deployed_name: str | None = None
            restart = False

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
                    restart = True
                    print()
                    break

                if lower == "/preview":
                    print()
                    print(gen.config_yaml or "")
                    print()
                    continue

                if lower == "/save" or lower.startswith("/save "):
                    path_str = raw[len("/save"):].strip()
                    if not path_str:
                        path_str = f"{current_summary.get('name', 'app')}-config.yaml"
                    path = pathlib.Path(path_str).expanduser()
                    path.write_text(gen.config_yaml or "", encoding="utf-8")
                    print(f"  Saved to {path}")
                    continue

                if lower == "/deploy":
                    app_name = current_summary.get("name", "app")
                    print(f"\nDeploying {app_name}...")
                    try:
                        deploy_data = await gen.deploy()
                    except httpx.HTTPStatusError as exc:
                        print(f"  ERROR {exc.response.status_code}: {exc.response.text}")
                        continue
                    print(f"  status: {deploy_data['status']}")
                    if deploy_data.get("error"):
                        print(f"  error:  {deploy_data['error']}")
                    else:
                        deployed_name = deploy_data["name"]
                    break

                # Natural-language revision
                print("\nRevising...")
                try:
                    revise_data = await gen.revise(raw)
                except httpx.HTTPStatusError as exc:
                    print(f"  ERROR {exc.response.status_code}: {exc.response.text}")
                    continue
                current_summary = revise_data["summary"]
                _print_changes(revise_data.get("changes", []))

            if restart:
                continue

            if deployed_name is None:
                return

            # ------------------------------------------------------------------
            # Phase 3: optional query loop
            # ------------------------------------------------------------------
            print()
            try:
                launch = input("  Launch query loop? [y/N] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                return

            if launch == "y":
                print()
                client = CogBaseClient(deployed_name, _API_BASE, http)
                print(f"Commands: /list | /ingest <file> | /list_collections | /query_structured | /clear | /reset | /q")
                print()
                await run_interactive_loop(client, lambda: b"")
            return


if __name__ == "__main__":
    asyncio.run(main())
