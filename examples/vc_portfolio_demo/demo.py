"""VC Portfolio Intelligence Demo — drive CogBase via the REST API.

Usage
-----
    # Start the API server first:
    uvicorn api.main:app --reload

    # Then run the demo:
    cd /path/to/cogbase
    python examples/vc_portfolio_demo/demo.py

Requires OPENAI_API_KEY in a .env file at the repo root (or in the environment).
Set COGBASE_API_URL to override the default http://localhost:8000.

Commands (interactive loop)
---------------------------
    /list                       List all applications
    /create                     Create the vc-portfolio application
    /delete <name>              Delete an application by name
    /ingest_all                 Ingest all built-in synthetic board updates + memos
    /ingest_board               Ingest only board updates
    /ingest_memos               Ingest only investment memos
    /list_collections           List all structured and vector collections
    /query_structured           Dump all portfolio_kpis records
    /query_structured <name>    Query a named structured collection
    /clear                      Clear chat history
    /reset                      Delete the application and start fresh
    /q /quit /exit              Exit

Then type any natural-language question to run a query, e.g.:
    Which companies are burning more than $500K per month?
    What was Nova Analytics' ARR in Q3 2024?
    Are there any contradictions in how Helix reported ARR?
    Which companies have runway below 12 months?
    What are the key risks across the portfolio?
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
from examples.vc_portfolio_demo.portfolio_data import BOARD_UPDATES, DEAL_MEMOS  # noqa: E402
from examples.vc_portfolio_demo.schema import PortfolioKPIExtraction, PortfolioKPIRecord  # noqa: E402

configure_logging()

_APP_NAME = "vc-portfolio"
_API_BASE = os.environ.get("COGBASE_API_URL", "http://localhost:8000")
_KPI_COLLECTION = "portfolio_kpis"

_DEMO_DIR = pathlib.Path(__file__).parent.resolve()


def _build_bundle() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(_DEMO_DIR / "config.yaml", "config.yaml")
        zf.writestr("kpi_record_schema.json", json.dumps(PortfolioKPIRecord.model_json_schema(), indent=2))
        zf.writestr("kpi_extraction_schema.json", json.dumps(PortfolioKPIExtraction.model_json_schema(), indent=2))
        zf.write(_DEMO_DIR / "kpi_extraction_prompt.txt", "kpi_extraction_prompt.txt")
    return buf.getvalue()


async def _ingest_batch(client: CogBaseClient, batch: dict[str, dict], label: str) -> None:
    documents = [
        {"doc_id": doc_id, "text": entry["text"], "metadata": entry["metadata"]}
        for doc_id, entry in batch.items()
    ]
    print(f"Ingesting {len(documents)} {label}...")
    try:
        results = await client.ingest_documents(documents, timeout=300)
    except httpx.HTTPStatusError as exc:
        print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
        return
    for r in results:
        extracted = r.get("records_extracted", 0)
        if r["success"]:
            print(f"  {r['doc_id']:<30}  OK  ({extracted} record extracted)")
        else:
            print(f"  {r['doc_id']:<30}  FAILED: {r['error']}")


async def main() -> None:
    print()
    print("VC Portfolio Intelligence Demo (REST API)")
    print("=" * 45)
    print(f"  api:  {_API_BASE}")
    print()
    print("Suggested queries after ingestion:")
    print("  Which companies are burning more than $500K per month?")
    print("  What was Nova Analytics' ARR growth across all quarters?")
    print("  Are there any contradictions in how Helix reported ARR in Q3 2024?")
    print("  Which portfolio companies have runway below 12 months?")
    print("  What are the biggest risks across the portfolio?")
    print("  What was the investment thesis for Helix Biotech?")
    print()

    async with httpx.AsyncClient() as http:
        client = CogBaseClient(_APP_NAME, _API_BASE, http)

        app_info = await cmd_startup(client, _build_bundle())
        if app_info is None:
            return
        print()

        print("Commands: /list | /create | /delete <name> | /ingest_all | /ingest_board | /ingest_memos | /list_collections | /query_structured [<name>] | /clear | /reset | /q")
        print()

        async def handler(raw: str, lower: str) -> bool:
            if lower == "/ingest_all":
                await _ingest_batch(client, BOARD_UPDATES, "board updates + LP updates")
                await _ingest_batch(client, DEAL_MEMOS, "investment memos")
                return True
            if lower == "/ingest_board":
                await _ingest_batch(client, BOARD_UPDATES, "board updates + LP updates")
                return True
            if lower == "/ingest_memos":
                await _ingest_batch(client, DEAL_MEMOS, "investment memos")
                return True
            return False

        await run_interactive_loop(
            client, _build_bundle,
            default_collection=_KPI_COLLECTION,
            handler=handler,
            extra_commands=["/ingest_all", "/ingest_board", "/ingest_memos"],
        )


if __name__ == "__main__":
    asyncio.run(main())
