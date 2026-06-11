"""Contract Compliance Demo — drive CogBase via the REST API.

Usage
-----
    # Start the API server first (Docker, no build required — see server/README.md):
    ./server/docker_hub_demo.sh pull
    ./server/docker_hub_demo.sh run

    # Then run the demo (from the repo root):
    python examples/contract_compliance_demo/demo.py

The API server runs at http://localhost:8000. After it starts, configure your LLM
and embedding provider (including API key) via the UI Settings tab.

NOTE: The check, report, and alerts commands require persistent store backends.
The Docker demo image ships with SQLite + FAISS + local file storage, so they work
out of the box; pass a host data directory to docker_hub_demo.sh run to persist
data across container restarts (see server/README.md).

Commands (interactive loop)
---------------------------
    /ingest_demo_rules          Ingest the built-in demo company rules documents
    /ingest_rules <path>        Ingest a rules document from disk
    /ingest_demo_contracts      Ingest the built-in demo sample contracts (3 contracts)
    /ingest_contract <path>     Ingest a contract from disk
    /check <doc_id>             Run clause-by-clause compliance check
    /report <doc_id>            Print stored compliance report for one contract
    /alerts                     List high and critical findings across all contracts

Any other input is sent as a natural-language query to the contract-compliance app.
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
from examples.contract_compliance_demo.contracts_data import CONTRACTS_DOCUMENTS  # noqa: E402
from examples.contract_compliance_demo.rules_data import RULES_DOCUMENTS  # noqa: E402
from examples.contract_compliance_demo.schema import (  # noqa: E402
    ClauseComplianceFinding,
    ContractClause,
    ContractClauseRecord,
    ContractMetadata,
    ContractMetadataRecord,
)

configure_logging()

_APP_NAME = "contract-compliance"
_DEFAULT_STRUCTURED_COLLECTION = "contract_metadata"


def _build_bundle() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(_DEMO_DIR / "config.yaml", "config.yaml")
        zf.writestr("contract_metadata_record_schema.json", json.dumps(ContractMetadataRecord.model_json_schema(), indent=2))
        zf.writestr("contract_metadata_extraction_schema.json", json.dumps(ContractMetadata.model_json_schema(), indent=2))
        zf.write(_DEMO_DIR / "contract_metadata_prompt.txt", "contract_metadata_prompt.txt")
        zf.writestr("contract_clause_record_schema.json", json.dumps(ContractClauseRecord.model_json_schema(), indent=2))
        zf.writestr("contract_clause_extraction_schema.json", json.dumps(ContractClause.model_json_schema(), indent=2))
        zf.write(_DEMO_DIR / "contract_clauses_prompt.txt", "contract_clauses_prompt.txt")
        zf.writestr("clause_compliance_findings_schema.json", json.dumps(ClauseComplianceFinding.model_json_schema(), indent=2))
        zf.write(_DEMO_DIR / "compliance_judge_prompt.txt", "compliance_judge_prompt.txt")
    return buf.getvalue()


async def main() -> None:
    print()
    print("Contract Compliance Demo (REST API)")
    print("=" * 42)

    async with CogBaseClient() as client:
        client.use_app(_APP_NAME)
        print(f"  api:     {client.api_base}")
        print()

        app_info = await cmd_startup(client, _build_bundle())
        if app_info is None:
            return
        print()

        async def handler(raw: str, lower: str) -> bool:
            if lower == "/ingest_demo_rules" or lower.startswith("/ingest_demo_rules "):
                documents = [
                    {"doc_id": doc.doc_id, "text": doc.text, "metadata": dict(doc.metadata)}
                    for doc in RULES_DOCUMENTS
                ]
                print(f"Ingesting {len(documents)} built-in rule documents...")
                try:
                    results = await client.upload_text_documents(documents, timeout=180)
                except httpx.HTTPStatusError as exc:
                    print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
                    return True
                for r in results:
                    if r["success"]:
                        print(f"  {r['doc_id']:<14}  OK  (rule chunks indexed)")
                    else:
                        print(f"  {r['doc_id']:<14}  FAILED: {r['error']}")
                return True

            if lower.startswith("/ingest_rules "):
                rest = raw[len("/ingest_rules "):].strip()
                file_path = pathlib.Path(rest).expanduser()
                if not file_path.is_absolute():
                    file_path = pathlib.Path.cwd() / file_path
                if not file_path.exists():
                    print(f"  File not found: {file_path}")
                    return True
                print(f"Ingesting {file_path.name}...")
                try:
                    results = await client.upload_documents(
                        [file_path], metadata={"doc_type": "rules"}, timeout=180
                    )
                except httpx.HTTPStatusError as exc:
                    print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
                    return True
                for r in results:
                    if r["success"]:
                        print(f"  {r['doc_id']:<14}  OK  (rule chunks indexed)")
                    else:
                        print(f"  {r['doc_id']:<14}  FAILED: {r['error']}")
                return True

            if lower == "/ingest_demo_contracts":
                print(f"Ingesting {len(CONTRACTS_DOCUMENTS)} built-in demo contracts...")
                documents = [
                    {"doc_id": doc.doc_id, "text": doc.text, "metadata": dict(doc.metadata)}
                    for doc in CONTRACTS_DOCUMENTS
                ]
                try:
                    results = await client.upload_text_documents(documents, timeout=180)
                except httpx.HTTPStatusError as exc:
                    print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
                    return True
                for r in results:
                    if r["success"]:
                        print(f"  {r['doc_id']:<14}  OK")
                    else:
                        print(f"  {r['doc_id']:<14}  FAILED: {r['error']}")
                return True

            if lower.startswith("/ingest_contract "):
                rest = raw[len("/ingest_contract "):].strip()
                file_path = pathlib.Path(rest).expanduser()
                if not file_path.is_absolute():
                    file_path = pathlib.Path.cwd() / file_path
                if not file_path.exists():
                    print(f"  File not found: {file_path}")
                    return True
                print(f"Ingesting {file_path.name}...")
                try:
                    results = await client.upload_documents(
                        [file_path], metadata={"doc_type": "contract"}, timeout=180
                    )
                except httpx.HTTPStatusError as exc:
                    print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
                    return True
                r = results[0]
                if r["success"]:
                    print(f"  {r['doc_id']}  OK")
                else:
                    print(f"  {r['doc_id']}  FAILED: {r['error']}")
                return True

            if lower.startswith("/check"):
                doc_id = raw[len("/check"):].strip()
                if not doc_id:
                    print("  Usage: /check <doc_id>")
                    return True
                print(f"Checking compliance for {doc_id!r}...")
                count = non_compliant = needs_review = 0
                try:
                    async with client._http.stream(
                        "POST",
                        f"{client.api_base}/applications/{client.app_name}/workflows/check-contract-compliance/stream",
                        json={"doc_id": doc_id},
                        timeout=300,
                    ) as resp:
                        resp.raise_for_status()
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
                            if "error" in data:
                                print(f"\n  ERROR: {data['error']}")
                                continue
                            finding = data.get("record", data)
                            count += 1
                            status_val = finding.get("status", "")
                            print(json.dumps(finding, indent=2))
                            if status_val == "non_compliant":
                                non_compliant += 1
                            elif status_val == "needs_review":
                                needs_review += 1
                except httpx.HTTPStatusError as exc:
                    print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
                    return True
                if count == 0:
                    print(f"  No clauses found for {doc_id!r}. Run '/ingest_demo_contracts' first.")
                else:
                    compliant = count - non_compliant - needs_review
                    print(f"\n  {count} findings saved.  "
                          f"non-compliant: {non_compliant}  "
                          f"needs-review: {needs_review}  "
                          f"compliant: {compliant}")
                return True

            if lower.startswith("/report"):
                doc_id = raw[len("/report"):].strip()
                if not doc_id:
                    print("  Usage: /report <doc_id>")
                    return True
                try:
                    findings = await client.query_structured(
                        "clause_compliance_findings",
                        filters=[{"field": "doc_id", "op": "=", "value": doc_id}],
                    )
                except httpx.HTTPStatusError as exc:
                    print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
                    return True
                if not findings:
                    print(f"  No findings for {doc_id!r}. Run '/check {doc_id}' first.")
                    return True
                print(f"\nCompliance report for {doc_id}")
                print("-" * 60)
                by_status: dict[str, list] = {}
                for f in findings:
                    by_status.setdefault(f.get("status", "unknown"), []).append(f)
                for status, group in by_status.items():
                    print(f"\n[{status.upper()}] — {len(group)} clause(s)")
                    for f in group:
                        print(json.dumps(f, indent=2))
                print()
                return True

            if lower == "/alerts":
                try:
                    findings = await client.query_structured(
                        "clause_compliance_findings",
                        filters=[
                            {"field": "status", "op": "=", "value": "non_compliant"},
                            {"field": "severity", "op": "in", "value": ["high", "critical"]},
                        ],
                    )
                except httpx.HTTPStatusError as exc:
                    print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
                    return True
                if not findings:
                    print("  No high/critical non-compliant findings.")
                    return True
                print(f"\nHigh / Critical non-compliant findings ({len(findings)})")
                print("-" * 70)
                for f in sorted(
                    findings,
                    key=lambda x: (
                        {"critical": 0, "high": 1}.get(x.get("severity", ""), 2),
                        x.get("doc_id", ""),
                    ),
                ):
                    sev = (f.get("severity") or "").upper()
                    doc = f.get("doc_id", "")
                    cid = f.get("clause_id", "")
                    summary = f.get("summary", "")
                    print(f"  {sev:<8}  {doc:<14}  {cid:<28}  {summary}")
                print()
                return True

            return False

        await run_interactive_loop(
            client, _build_bundle,
            default_collection=_DEFAULT_STRUCTURED_COLLECTION,
            handler=handler,
            extra_commands=[
                "/ingest_demo_rules", "/ingest_rules",
                "/ingest_demo_contracts", "/ingest_contract",
                "/check", "/report", "/alerts",
            ],
        )


if __name__ == "__main__":
    asyncio.run(main())
