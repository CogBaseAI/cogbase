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
    ingest saas          Ingest the built-in 5 SaaS contract fixtures
    ingest <path>        Ingest a plain-text contract file from disk
    reset                Delete the application and start fresh
    q / quit / exit      Exit
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
_CHAT_MODEL = "gpt-5-mini"
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
vector_store:
  type: faiss
  dim: {_EMBED_DIM}
structured_store:
  type: memory
vector_collections:
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


async def _delete_app(client: httpx.AsyncClient) -> None:
    resp = await client.delete(f"{_API_BASE}/applications/{_APP_NAME}", timeout=10)
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

        print("Commands: ingest saas | ingest <file> | reset | q")
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
