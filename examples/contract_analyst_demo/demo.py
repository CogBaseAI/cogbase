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
    /list                       List all applications
    /create                     Create the contract-analyst application
    /delete <name>              Delete an application by name
    /ingest_saas                Ingest the built-in 5 SaaS contract fixtures
    /ingest <path>              Ingest a plain-text contract file from disk
    /list_collections           List all structured collections for the application
    /query_structured           Query the default contracts collection (all records)
    /query_structured <name>    Query a named structured collection (all records)
    /clear                      Clear chat history
    /reset                      Delete the application and start fresh
    /q /quit /exit              Exit
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import pathlib
import sys
import zipfile

_DEMO_DIR = pathlib.Path(__file__).parent.resolve()
_REPO_ROOT = _DEMO_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import httpx  # noqa: E402

from examples.cogbase_client import (  # noqa: E402
    CogBaseClient,
    cmd_startup,
    configure_logging,
    run_interactive_loop,
)
from examples.contract_analyst_demo.schema import (  # noqa: E402
    ContractExtraction,
    ContractExtractionRecord,
)
from examples.contract_analyst_demo.saas_contracts import CONTRACTS  # noqa: E402

configure_logging()

_APP_NAME = "contract-analyst"
_API_BASE = os.environ.get("COGBASE_API_URL", "http://localhost:8000")
_CONTRACTS_COLLECTION = "contracts"


def _build_bundle() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(_DEMO_DIR / "config.yaml", "config.yaml")
        zf.writestr("contracts_record_schema.json", json.dumps(ContractExtractionRecord.model_json_schema(), indent=2))
        zf.writestr("contracts_extraction_schema.json", json.dumps(ContractExtraction.model_json_schema(), indent=2))
        zf.write(_DEMO_DIR / "contracts_prompt.txt", "contracts_prompt.txt")
    return buf.getvalue()


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

        print("Commands: /list | /create | /delete <name> | /ingest_saas | /ingest <file> | /list_collections | /query_structured [<name>] | /clear | /reset | /q")
        print()

        async def handler(raw: str, lower: str) -> bool:
            if lower == "/ingest_saas":
                print(f"Ingesting {len(CONTRACTS)} built-in SaaS contracts...")
                documents = [{"doc_id": doc_id, "text": text} for doc_id, text in CONTRACTS.items()]
                try:
                    results = await client.ingest_documents(documents)
                except httpx.HTTPStatusError as exc:
                    print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
                    return True
                for r in results:
                    if r["success"]:
                        print(f"  {r['doc_id']:<12}  OK  ({r['records_extracted']} record extracted)")
                    else:
                        print(f"  {r['doc_id']:<12}  FAILED: {r['error']}")
                return True

            if lower.startswith("/ingest "):
                rest = raw[len("/ingest "):].strip()
                file_path = pathlib.Path(rest).expanduser()
                if not file_path.is_absolute():
                    file_path = pathlib.Path.cwd() / file_path
                if not file_path.exists():
                    print(f"  File not found: {file_path}")
                    return True
                doc_id = file_path.stem
                text = file_path.read_text(errors="replace")
                print(f"Ingesting {file_path.name} as doc_id={doc_id!r}...")
                try:
                    results = await client.ingest_documents([{"doc_id": doc_id, "text": text}])
                except httpx.HTTPStatusError as exc:
                    print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
                    return True
                r = results[0]
                if r["success"]:
                    print(f"  {doc_id}  OK")
                else:
                    print(f"  {doc_id}  FAILED: {r['error']}")
                return True

            return False

        await run_interactive_loop(
            client, _build_bundle,
            default_collection=_CONTRACTS_COLLECTION,
            handler=handler,
            extra_commands=["/ingest_saas", "/ingest"],
        )


if __name__ == "__main__":
    asyncio.run(main())
