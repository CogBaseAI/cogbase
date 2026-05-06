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
import os
import pathlib
import readline  # noqa: F401 — enables arrow-key line editing in input()
import sys
import zipfile

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

from examples.cogbase_client import (  # noqa: E402
    CogBaseClient,
    cmd_create,
    cmd_delete,
    cmd_list,
    cmd_list_collections,
    cmd_query_structured,
    cmd_reset,
    cmd_startup,
    configure_logging,
)
from examples.contract_analyst_demo.schema import (  # noqa: E402
    CONTRACTS_SYSTEM_PROMPT_PREFIX,
    ContractExtraction,
)
from examples.contract_analyst_demo.saas_contracts import CONTRACTS  # noqa: E402

configure_logging()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_APP_NAME = "contract-analyst"
_API_BASE = os.environ.get("COGBASE_API_URL", "http://localhost:8000")

_CONTRACTS_COLLECTION = "contracts"

# ---------------------------------------------------------------------------
# ZIP bundle
# ---------------------------------------------------------------------------

_CONFIG_YAML = f"""\
name: {_APP_NAME}
vector_collections:
  - name: document_chunks
    description: >-
      Full-text document chunks for detailed retrieval.
structured_collections:
  - name: {_CONTRACTS_COLLECTION}
    description: >-
      Extracted contract facts and entities for exact lookup.
    schema: contracts_schema.json
pipeline:
  steps:
    - tool: chunk-embed-upsert
      collection: document_chunks
      chunker:
        type: fixed
        chunk_size: 512
        overlap: 64
    - tool: extract-structured
      collection: {_CONTRACTS_COLLECTION}
      extractor:
        type: llm
        prompt: contracts_prompt.txt
"""


def _build_bundle() -> bytes:
    schema_json = json.dumps(ContractExtraction.model_json_schema(), indent=2)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("config.yaml", _CONFIG_YAML)
        zf.writestr("contracts_schema.json", schema_json)
        zf.writestr("contracts_prompt.txt", CONTRACTS_SYSTEM_PROMPT_PREFIX)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Main async loop
# ---------------------------------------------------------------------------


async def main() -> None:
    print()
    print("Contract Analyst Demo (REST API)")
    print("=" * 40)
    print(f"  api:    {_API_BASE}")
    print()

    async with httpx.AsyncClient() as http:
        client = CogBaseClient(_APP_NAME, _API_BASE, http)

        app_info = await cmd_startup(client, _build_bundle())
        if app_info is None:
            return
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

            if lower == "list":
                await cmd_list(client)
                continue

            if lower == "create":
                await cmd_create(client, _build_bundle())
                continue

            if lower.startswith("delete "):
                await cmd_delete(client, raw)
                continue

            if lower == "reset":
                if await cmd_reset(client):
                    break
                continue

            if lower == "ingest saas":
                print(f"Ingesting {len(CONTRACTS)} built-in SaaS contracts...")
                documents = [
                    {"doc_id": doc_id, "text": text}
                    for doc_id, text in CONTRACTS.items()
                ]
                try:
                    results = await client.ingest_documents(documents)
                except httpx.HTTPStatusError as exc:
                    print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
                    continue
                for r in results:
                    if r["success"]:
                        print(f"  {r['doc_id']:<12}  OK  ({r['records_extracted']} record extracted)")
                    else:
                        print(f"  {r['doc_id']:<12}  FAILED: {r['error']}")
                continue

            if lower == "list collections":
                await cmd_list_collections(client)
                continue

            if lower == "query structured" or lower.startswith("query structured "):
                collection = (
                    raw[len("query structured "):].strip()
                    if lower.startswith("query structured ")
                    else _CONTRACTS_COLLECTION
                )
                await cmd_query_structured(client, collection)
                continue

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
                    results = await client.ingest_documents([{"doc_id": doc_id, "text": text}])
                except httpx.HTTPStatusError as exc:
                    print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
                    continue
                r = results[0]
                if r["success"]:
                    print(f"  {doc_id}  OK")
                else:
                    print(f"  {doc_id}  FAILED: {r['error']}")
                continue

            print("Thinking...")
            try:
                await client.query_stream(raw)
            except httpx.HTTPStatusError as exc:
                print(f"  ERROR: {exc.response.status_code} {exc.response.text}")


if __name__ == "__main__":
    asyncio.run(main())
