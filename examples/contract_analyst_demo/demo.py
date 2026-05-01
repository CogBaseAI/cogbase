"""Contract Analyst Demo — drive CogBase via the REST API.

Usage
-----
    # Start the API server first:
    uvicorn api.main:app --reload

    # Then run the demo:
    cd /path/to/cogbase
    python examples/contract_analyst_demo/demo.py

Requires OPENAI_API_KEY in a .env file at the repo root (or in the environment).
Set COGBASE_API_URL to override the default http://localhost:8000.

Commands (interactive loop)
---------------------------
    list                        List all applications
    create                      Create the contract-analyst application
    delete <name>               Delete an application by name
    ingest saas                 Ingest the built-in 5 SaaS contract fixtures
    ingest <path>               Ingest a plain-text contract file from disk
    list collections            List all structured collections for the application
    query structured            Query the default contracts collection (all records)
    query structured <name>     Query a named structured collection (all records)
    reset                       Delete the application and start fresh
    q / quit / exit             Exit
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import pathlib
import sys
import zipfile

_format = "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=_format)

# ---------------------------------------------------------------------------
# Repo root on the Python path
# ---------------------------------------------------------------------------

_DEMO_DIR = pathlib.Path(__file__).parent.resolve()
_REPO_ROOT = _DEMO_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Imports (after sys.path is configured)
# ---------------------------------------------------------------------------

import httpx  # noqa: E402

from examples.contract_analyst_demo.schema import (  # noqa: E402
    CONTRACTS_SYSTEM_PROMPT_PREFIX,
    ContractExtraction,
)
from examples.contract_analyst_demo.saas_contracts import CONTRACTS  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_APP_NAME = "contract-analyst"
_CHAT_MODEL = "gpt-5.4-mini"
_EMBED_MODEL = "text-embedding-3-small"
_EMBED_DIM = 1536
_API_BASE = os.environ.get("COGBASE_API_URL", "http://localhost:8000").rstrip("/")

_CONTRACTS_COLLECTION = "contracts"

# ---------------------------------------------------------------------------
# ZIP bundle
# ---------------------------------------------------------------------------

_CONFIG_YAML = f"""\
name: {_APP_NAME}
llm:
  provider: openai
  model: {_CHAT_MODEL}
embedding:
  provider: openai
  model: {_EMBED_MODEL}
  dimensions: {_EMBED_DIM}
chunk_collections:
  - name: document_chunks
    chunker:
      type: fixed
      chunk_size: 512
      overlap: 64
structured_collections:
  - name: {_CONTRACTS_COLLECTION}
    schema: contracts_schema.json
    extractor:
      type: llm
      prompt: contracts_prompt.txt
pipeline:
  steps:
    - tool: chunk-embed-upsert
      collection: document_chunks
    - tool: extract-structured
      collection: {_CONTRACTS_COLLECTION}
"""


def _build_bundle() -> bytes:
    """Build an in-memory ZIP bundle: config.yaml + schema + prompt."""
    schema_json = json.dumps(ContractExtraction.model_json_schema(), indent=2)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("config.yaml", _CONFIG_YAML)
        zf.writestr("contracts_schema.json", schema_json)
        zf.writestr("contracts_prompt.txt", CONTRACTS_SYSTEM_PROMPT_PREFIX)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# REST helpers
# ---------------------------------------------------------------------------

async def _list_apps(client: httpx.AsyncClient) -> list[dict]:
    resp = await client.get(f"{_API_BASE}/applications", timeout=10)
    resp.raise_for_status()
    return resp.json()["applications"]


async def _get_app(client: httpx.AsyncClient) -> dict | None:
    resp = await client.get(f"{_API_BASE}/applications/{_APP_NAME}", timeout=10)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


async def _create_app(client: httpx.AsyncClient) -> dict:
    bundle = _build_bundle()
    resp = await client.post(
        f"{_API_BASE}/applications",
        files={"bundle": ("bundle.zip", bundle, "application/zip")},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


async def _delete_app(client: httpx.AsyncClient, name: str = _APP_NAME) -> None:
    resp = await client.delete(f"{_API_BASE}/applications/{name}", timeout=10)
    if resp.status_code not in (204, 404):
        resp.raise_for_status()


async def _ingest_documents(
    client: httpx.AsyncClient,
    documents: list[dict],
) -> list[dict]:
    resp = await client.post(
        f"{_API_BASE}/applications/{_APP_NAME}/ingest_documents",
        json={"documents": documents, "concurrency": 3},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["results"]


async def _query_stream(client: httpx.AsyncClient, text: str) -> None:
    """POST to /query/stream and print SSE tokens as they arrive."""
    async with client.stream(
        "POST",
        f"{_API_BASE}/applications/{_APP_NAME}/query/stream",
        json={"text": text},
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
            elif "result" in data:
                result = data["result"]
                if result.get("passthrough") and result.get("structured_records"):
                    print(json.dumps(result["structured_records"], indent=2))
            elif "error" in data:
                print(f"\n  ERROR: {data['error']}")
        print()


async def _list_collections(client: httpx.AsyncClient) -> list[str]:
    resp = await client.get(f"{_API_BASE}/applications/{_APP_NAME}/collections", timeout=10)
    resp.raise_for_status()
    return resp.json()["collections"]


async def _query_structured(client: httpx.AsyncClient, collection: str) -> list[dict]:
    resp = await client.post(
        f"{_API_BASE}/applications/{_APP_NAME}/collections/{collection}/query",
        json={"filters": [], "fields": None},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["records"]


# ---------------------------------------------------------------------------
# Main async loop
# ---------------------------------------------------------------------------


async def main() -> None:
    print()
    print("Contract Analyst Demo (REST API)")
    print("=" * 40)
    print(f"  model:  {_CHAT_MODEL}")
    print(f"  embed:  {_EMBED_MODEL}")
    print(f"  api:    {_API_BASE}")
    print()

    async with httpx.AsyncClient() as client:
        app_info = await _get_app(client)
        if app_info is None:
            print(f"Creating application '{_APP_NAME}'...")
            try:
                app_info = await _create_app(client)
            except httpx.HTTPStatusError as exc:
                print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
                return
            print(f"  status: {app_info['status']}")
            if app_info.get("error"):
                print(f"  error:  {app_info['error']}")
        else:
            print(f"Application '{_APP_NAME}' already exists (status: {app_info['status']})")
        print()

        print("Commands: list | create | delete <name> | ingest saas | ingest <file> | list collections | query structured [<name>] | reset | q")
        print()

        while True:
            try:
                raw = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not raw:
                continue

            lower = raw.lower()

            if lower in {"q", "quit", "exit"}:
                print("Goodbye!")
                break

            # ---- list ---------------------------------------------------
            if lower == "list":
                try:
                    apps = await _list_apps(client)
                except httpx.HTTPStatusError as exc:
                    print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
                    continue
                if not apps:
                    print("  No applications found.")
                else:
                    for app in apps:
                        print(f"  {app['name']:<24}  status: {app['status']}")
                continue

            # ---- create -------------------------------------------------
            if lower == "create":
                existing = await _get_app(client)
                if existing is not None:
                    print(f"  Application '{_APP_NAME}' already exists (status: {existing['status']})")
                    continue
                print(f"Creating application '{_APP_NAME}'...")
                try:
                    result = await _create_app(client)
                except httpx.HTTPStatusError as exc:
                    print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
                    continue
                print(f"  status: {result['status']}")
                if result.get("error"):
                    print(f"  error:  {result['error']}")
                continue

            # ---- delete <name> ------------------------------------------
            if lower.startswith("delete "):
                name = raw[len("delete "):].strip()
                if not name:
                    print("  Usage: delete <name>")
                    continue
                confirm = input(f"  Delete application '{name}' and all its data? [y/N] ").strip().lower()
                if confirm == "y":
                    try:
                        await _delete_app(client, name)
                    except httpx.HTTPStatusError as exc:
                        print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
                        continue
                    print(f"  Application '{name}' deleted.")
                continue

            # ---- reset --------------------------------------------------
            if lower == "reset":
                confirm = input("  Delete application and all data? [y/N] ").strip().lower()
                if confirm == "y":
                    await _delete_app(client)
                    print("  Application deleted. Restart the demo to start fresh.")
                    break
                continue

            # ---- ingest saas --------------------------------------------
            if lower == "ingest saas":
                print(f"Ingesting {len(CONTRACTS)} built-in SaaS contracts...")
                documents = [
                    {"doc_id": doc_id, "text": text}
                    for doc_id, text in CONTRACTS.items()
                ]
                try:
                    results = await _ingest_documents(client, documents)
                except httpx.HTTPStatusError as exc:
                    print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
                    continue
                for r in results:
                    if r["success"]:
                        print(f"  {r['doc_id']:<12}  OK  ({r['records_extracted']} record extracted)")
                    else:
                        print(f"  {r['doc_id']:<12}  FAILED: {r['error']}")
                continue

            # ---- list collections ---------------------------------------
            if lower == "list collections":
                try:
                    cols = await _list_collections(client)
                except httpx.HTTPStatusError as exc:
                    print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
                    continue
                if not cols:
                    print("  No structured collections found.")
                else:
                    for c in cols:
                        print(f"  {c}")
                continue

            # ---- query structured [<collection>] ------------------------
            if lower == "query structured" or lower.startswith("query structured "):
                collection = (
                    raw[len("query structured "):].strip()
                    if lower.startswith("query structured ")
                    else _CONTRACTS_COLLECTION
                )
                print(f"Querying structured collection '{collection}'...")
                try:
                    records = await _query_structured(client, collection)
                except httpx.HTTPStatusError as exc:
                    print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
                    continue
                if not records:
                    print("  No records found.")
                else:
                    print(json.dumps(records, indent=2))
                continue

            # ---- ingest <file> ------------------------------------------
            if lower.startswith("ingest "):
                rest = raw[len("ingest "):].strip()
                file_path = pathlib.Path(rest).expanduser()
                if not file_path.is_absolute():
                    file_path = pathlib.Path.cwd() / file_path
                if not file_path.exists():
                    print(f"  File not found: {file_path}")
                    continue
                doc_id = file_path.stem
                text = file_path.read_text(errors="replace")
                print(f"Ingesting {file_path.name} as doc_id={doc_id!r}...")
                try:
                    results = await _ingest_documents(client, [{"doc_id": doc_id, "text": text}])
                except httpx.HTTPStatusError as exc:
                    print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
                    continue
                r = results[0]
                if r["success"]:
                    print(f"  {doc_id}  OK")
                else:
                    print(f"  {doc_id}  FAILED: {r['error']}")
                continue

            # ---- question / anything else -------------------------------
            print("Thinking...")
            try:
                await _query_stream(client, raw)
            except httpx.HTTPStatusError as exc:
                print(f"  ERROR: {exc.response.status_code} {exc.response.text}")


if __name__ == "__main__":
    asyncio.run(main())
