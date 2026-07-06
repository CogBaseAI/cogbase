"""Contract Analyst Demo — drive CogBase via the REST API.

Usage
-----
    # Start the API server first (Docker, no build required — see server/README.md):
    ./server/docker_hub_demo.sh pull
    ./server/docker_hub_demo.sh run

    # Then run the demo:
    cd /path/to/cogbase
    python examples/contract_analyst_demo/demo.py

The API server runs at http://localhost:8000. After it starts, configure your LLM
and embedding provider (including API key) via the UI Settings tab.

Commands (interactive loop)
---------------------------
    /ingest_demo_contracts                Ingest the 6-contract starter set (fast first impression)
    /ingest_demo_contracts all            Ingest all 30 built-in contract fixtures (full showcase)
    /ingest_demo_contract <doc_id>        Ingest a single built-in contract (e.g. saas-001)
"""

from __future__ import annotations

import asyncio
import io
import json
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
from examples.contract_analyst_demo.contracts import (  # noqa: E402
    CONTRACTS,
    STARTER_CONTRACTS,
)

configure_logging()

_APP_NAME = "contract-analyst"
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

    async with CogBaseClient() as client:
        client.use_app(_APP_NAME)
        print(f"  api:    {client.api_base}")
        print()

        app_info = await cmd_startup(client, _build_bundle())
        if app_info is None:
            return
        print()

        async def ingest(documents: list[dict]) -> None:
            # The fixtures live as plain text in contracts.py; each is rendered
            # to a Word document on the fly and uploaded (parsed to markdown
            # server-side) so nothing needs to be committed to git.
            try:
                results = await client.upload_docx_documents(documents)
            except httpx.HTTPStatusError as exc:
                print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
                return
            for r in results:
                if r["success"]:
                    print(f"  {r['doc_id']:<20}  OK")
                else:
                    print(f"  {r['doc_id']:<20}  FAILED: {r['error']}")

        async def handler(raw: str, lower: str) -> bool:
            if lower == "/ingest_demo_contracts" or lower.startswith("/ingest_demo_contracts "):
                arg = raw[len("/ingest_demo_contracts"):].strip().lower()
                full = arg in ("all", "full")
                corpus = CONTRACTS if full else STARTER_CONTRACTS
                label = "all" if full else "starter-set"
                print(f"Ingesting {len(corpus)} built-in contracts ({label}) as .docx...")
                if not full:
                    print("  (run '/ingest_demo_contracts all' for the full 30-contract showcase)")
                documents = [{"doc_id": doc_id, "text": text} for doc_id, text in corpus.items()]
                await ingest(documents)
                return True

            if lower == "/ingest_demo_contract" or lower.startswith("/ingest_demo_contract "):
                doc_id = raw[len("/ingest_demo_contract"):].strip()
                if not doc_id:
                    print("  Usage: /ingest_demo_contract <doc_id>  (e.g. saas-001)")
                    print(f"  Available: {', '.join(CONTRACTS)}")
                    return True
                text = CONTRACTS.get(doc_id)
                if text is None:
                    print(f"  Unknown contract: {doc_id}")
                    print(f"  Available: {', '.join(CONTRACTS)}")
                    return True
                print(f"Ingesting built-in contract {doc_id} as .docx...")
                await ingest([{"doc_id": doc_id, "text": text}])
                return True

            return False

        await run_interactive_loop(
            client, _build_bundle,
            default_collection=_CONTRACTS_COLLECTION,
            handler=handler,
            extra_commands=["/ingest_demo_contracts", "/ingest_demo_contract"],
        )


if __name__ == "__main__":
    asyncio.run(main())
