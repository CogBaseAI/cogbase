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
    /ingest_demo_contracts                Ingest the built-in SaaS contract fixtures
"""

from __future__ import annotations

import asyncio
import io
import json
import pathlib
import sys
import tempfile
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
from examples.contract_analyst_demo.docx_render import write_docx  # noqa: E402

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
                print(f"Ingesting {len(CONTRACTS)} built-in SaaS contracts as .docx...")
                # The fixtures live as plain text in saas_contracts.py; convert each
                # to a Word document on the fly and upload it (parsed to markdown
                # server-side) so nothing needs to be committed to git.
                try:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        paths = []
                        for doc_id, text in CONTRACTS.items():
                            path = pathlib.Path(tmpdir) / f"{doc_id}.docx"
                            write_docx(text, path)
                            paths.append(path)
                        results = await client.upload_documents(paths)
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
