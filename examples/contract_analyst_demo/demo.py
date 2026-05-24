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
    /ingest_demo_contracts                Ingest the built-in 5 SaaS contract fixtures
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
from examples.contract_analyst_demo.saas_contracts import CONTRACTS  # noqa: E402

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

        async def handler(raw: str, lower: str) -> bool:
            if lower == "/ingest_demo_contracts":
                print(f"Ingesting {len(CONTRACTS)} built-in SaaS contracts...")
                documents = [{"doc_id": doc_id, "text": text} for doc_id, text in CONTRACTS.items()]
                try:
                    results = await client.upload_text_documents(documents)
                except httpx.HTTPStatusError as exc:
                    print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
                    return True
                for r in results:
                    if r["success"]:
                        print(f"  {r['doc_id']:<12}  OK")
                    else:
                        print(f"  {r['doc_id']:<12}  FAILED: {r['error']}")
                return True

            return False

        await run_interactive_loop(
            client, _build_bundle,
            default_collection=_CONTRACTS_COLLECTION,
            handler=handler,
            extra_commands=["/ingest_demo_contracts"],
        )


if __name__ == "__main__":
    asyncio.run(main())
