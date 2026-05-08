"""Contract Compliance Demo — drive CogBase via the REST API.

Usage
-----
    # Start the API server first:
    uvicorn api.main:app --reload

    # Then run the demo (from the repo root):
    python examples/contract_compliance_demo/demo.py

Requires OPENAI_API_KEY in a .env file at the repo root (or in the environment).
Set COGBASE_API_URL to override the default http://localhost:8000.

NOTE: The check, report, and alerts commands require persistent store backends
(SQLite + FAISS).  Configure cogbase_system.yaml with structured_store.type=sqlite
and vector_store.type=faiss, or set COGBASE_CONFIG to point to your system config.

Commands (interactive loop)
---------------------------
    create                      Create the contract-compliance application
    ingest rules                Ingest the built-in company rules documents
    ingest rules <path>         Ingest a rules document from disk
    ingest contracts            Ingest the built-in sample contracts (3 contracts)
    ingest contract <path>      Ingest a contract from disk
    check <doc_id>              Run clause-by-clause compliance check
    report <doc_id>             Print stored compliance report for one contract
    alerts                      List high and critical findings across all contracts
    list                        List all applications
    list collections            List all collections for the compliance app
    query structured            Dump the contract_metadata collection
    query structured <name>     Dump a named collection
    delete <name>               Delete an application by name
    reset                       Delete the application and all demo data
    q / quit / exit             Exit

Any other input is sent as a natural-language query to the contract-compliance app.
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
from examples.contract_compliance_demo.contracts_data import (  # noqa: E402
    CONTRACTS_DOCUMENTS,
)
from examples.contract_compliance_demo.rules_data import RULES_DOCUMENTS  # noqa: E402
from examples.contract_compliance_demo.schema import (  # noqa: E402
    ClauseComplianceFinding,
    ContractClause,
    ContractClauseRecord,
    ContractMetadata,
    ContractMetadataRecord,
)

configure_logging()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_APP_NAME = "contract-compliance"
_API_BASE = os.environ.get("COGBASE_API_URL", "http://localhost:8000")

_DEFAULT_STRUCTURED_COLLECTION = "contract_metadata"

_JUDGE_SYSTEM_PROMPT = """\
You are a contract compliance reviewer. Determine whether a contract clause complies
with the company's internal policies, using ONLY the company policy excerpts provided.

Rules:
- Ground every finding exclusively in the provided policy excerpts.
- Do not invent policy or apply general legal knowledge not present in the excerpts.
- If the excerpts are insufficient to determine compliance, set status=needs_review.
- Every non_compliant finding MUST include at least one matched_rule_quote.
- Populate recommended_redline with revised clause language for non_compliant findings; null otherwise.
- Return ONLY valid JSON — no markdown fences, no explanation.
"""

# ---------------------------------------------------------------------------
# App bundle — config.yaml + metadata extraction schema + prompt
# ---------------------------------------------------------------------------

_CONTRACT_METADATA_SYSTEM_PROMPT = (
    "You are a legal contract analyst. Extract key contract facts from the provided contract.\n\n"
    "Rules:\n"
    "- Do not invent information not present in the contract.\n"
    "- Use null for any field not found in the contract.\n"
    "- Return ONLY the JSON object — no explanation, no markdown fences.\n"
    "- Format dates as YYYY-MM-DD.\n"
    "- For parties, return an array where each element has 'name' (legal name) "
    "and 'role' (role in the agreement, e.g. vendor, customer) keys.\n\n"
    "Return a single JSON object with these fields:\n\n"
)

_CONTRACT_CLAUSES_SYSTEM_PROMPT = (
    "You are a legal contract analyst. Extract every distinct clause from the provided contract.\n\n"
    "Rules:\n"
    "- Copy all clause text verbatim — do not paraphrase or summarise.\n"
    "- Do not invent clauses not present in the contract.\n"
    "- Assign clause_type from: liability, indemnification, termination, payment, "
    "privacy, confidentiality, ip, governing_law, other. Use null when unclear.\n"
    "- Return ONLY the JSON object — no explanation, no markdown fences.\n\n"
)

_CONFIG_YAML = f"""\
name: {_APP_NAME}
vector_collections:
  - name: rule_chunks
    description: >-
      Company policy and vendor contract standard passages. Use to retrieve rules,
      standards, and fallback positions relevant to a clause type or compliance topic.
  - name: contract_chunks
    description: >-
      Contract text passage chunks. Use for detailed questions about specific
      contract terms, wording, or clauses.
structured_collections:
  - name: contract_metadata
    description: >-
      Key facts per contract: parties, dates, value, governing law, termination
      notice period. One record per contract document.
    schema: contract_metadata_record_schema.json
    primary_fields: [doc_id]
  - name: contract_clauses
    description: >-
      Individual clauses extracted from contracts. Each record is one clause with
      its type and verbatim text. Filter by doc_id to retrieve all clauses for a
      contract, or filter by clause_type to find clauses of a specific category.
    schema: contract_clause_record_schema.json
    primary_fields: [clause_id]
  - name: clause_compliance_findings
    description: >-
      Clause-level compliance findings. Each record captures whether a contract
      clause complies with company policy, with severity, summary, and redline.
    schema: clause_compliance_findings_schema.json
    primary_fields: [clause_id]
pipelines:
  - name: rules
    match:
      metadata:
        doc_type: rules
    steps:
      - tool: chunk-embed-upsert
        collection: rule_chunks
        chunker:
          type: langchain
  - name: contracts
    match:
      metadata:
        doc_type: contract
    steps:
      - tool: chunk-embed-upsert
        collection: contract_chunks
        chunker:
          type: langchain
      - tool: extract-structured
        collection: contract_metadata
        extractor:
          type: llm
          extraction_schema: contract_metadata_extraction_schema.json
          prompt: contract_metadata_prompt.txt
      - tool: extract-structured
        collection: contract_clauses
        extractor:
          type: llm
          extraction_schema: contract_clause_extraction_schema.json
          record_mode: many
          response_field: clauses
          id_field: clause_id
          id_template: "{{doc_id}}__{{index:04d}}"
          prompt: contract_clauses_prompt.txt
workflows:
  - name: check-contract-compliance
    trigger:
      type: manual
    input_schema:
      doc_id: string
    steps:
      - id: load_clauses
        tool: structured-query
        collection: contract_clauses
        filters:
          doc_id: "{{{{ input.doc_id }}}}"
      - id: review_each_clause
        foreach: "{{{{ steps.load_clauses.records }}}}"
        steps:
          - id: retrieve_rules
            tool: vector-search
            collection: rule_chunks
            query: "{{{{ item.clause_type }}}}\n{{{{ item.text }}}}"
            top_k: 5
          - id: judge
            tool: llm-structured
            prompt: compliance_judge_prompt.txt
            input:
              clause: "{{{{ item }}}}"
              rules: "{{{{ steps.retrieve_rules.chunks }}}}"
            output_schema: clause_compliance_findings_schema.json
          - id: save_finding
            tool: structured-save
            collection: clause_compliance_findings
            records:
              - "{{{{ steps.judge.output }}}}"
"""


def _build_bundle() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("config.yaml", _CONFIG_YAML)
        zf.writestr("contract_metadata_record_schema.json", json.dumps(ContractMetadataRecord.model_json_schema(), indent=2))
        zf.writestr("contract_metadata_extraction_schema.json", json.dumps(ContractMetadata.model_json_schema(), indent=2))
        zf.writestr("contract_metadata_prompt.txt", _CONTRACT_METADATA_SYSTEM_PROMPT)
        zf.writestr("contract_clause_record_schema.json", json.dumps(ContractClauseRecord.model_json_schema(), indent=2))
        zf.writestr("contract_clause_extraction_schema.json", json.dumps(ContractClause.model_json_schema(), indent=2))
        zf.writestr("contract_clauses_prompt.txt", _CONTRACT_CLAUSES_SYSTEM_PROMPT)
        zf.writestr("clause_compliance_findings_schema.json", json.dumps(ClauseComplianceFinding.model_json_schema(), indent=2))
        zf.writestr("compliance_judge_prompt.txt", _JUDGE_SYSTEM_PROMPT)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Main async loop
# ---------------------------------------------------------------------------


async def main() -> None:
    print()
    print("Contract Compliance Demo (REST API)")
    print("=" * 42)
    print(f"  api:     {_API_BASE}")
    print()

    async with httpx.AsyncClient() as http:
        client = CogBaseClient(_APP_NAME, _API_BASE, http)

        app_info = await cmd_startup(client, _build_bundle())
        if app_info is None:
            return
        print()

        print(
            "Commands: create | ingest rules [<file>] | ingest contracts | "
            "ingest contract <file> | check <doc_id> | report <doc_id> | alerts | "
            "list | list collections | query structured [<name>] | "
            "reset | delete <name> | q"
        )
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

            if lower == "list collections":
                await cmd_list_collections(client)
                continue

            if lower == "query structured" or lower.startswith("query structured "):
                collection = (
                    raw[len("query structured "):].strip()
                    if lower.startswith("query structured ")
                    else _DEFAULT_STRUCTURED_COLLECTION
                )
                await cmd_query_structured(client, collection)
                continue

            if lower == "ingest rules" or lower.startswith("ingest rules "):
                rest = raw[len("ingest rules"):].strip()
                if rest:
                    file_path = pathlib.Path(rest).expanduser()
                    if not file_path.is_absolute():
                        file_path = pathlib.Path.cwd() / file_path
                    if not file_path.exists():
                        print(f"  File not found: {file_path}")
                        continue
                    doc_id = file_path.stem
                    text = file_path.read_text(errors="replace")
                    documents = [{"doc_id": doc_id, "text": text, "metadata": {"doc_type": "rules"}}]
                    print(f"Ingesting {file_path.name} as doc_id={doc_id!r}...")
                else:
                    documents = [
                        {"doc_id": doc.doc_id, "text": doc.text, "metadata": dict(doc.metadata)}
                        for doc in RULES_DOCUMENTS
                    ]
                    print(f"Ingesting {len(documents)} built-in rule documents...")
                try:
                    results = await client.ingest_documents(documents, timeout=180)
                except httpx.HTTPStatusError as exc:
                    print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
                    continue
                for r in results:
                    if r["success"]:
                        print(f"  {r['doc_id']:<14}  OK  (rule chunks indexed)")
                    else:
                        print(f"  {r['doc_id']:<14}  FAILED: {r['error']}")
                continue

            if lower == "ingest contracts":
                print(f"Ingesting {len(CONTRACTS_DOCUMENTS)} built-in contracts...")
                documents = [
                    {"doc_id": doc.doc_id, "text": doc.text, "metadata": dict(doc.metadata)}
                    for doc in CONTRACTS_DOCUMENTS
                ]
                try:
                    results = await client.ingest_documents(documents, timeout=180)
                except httpx.HTTPStatusError as exc:
                    print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
                    continue
                for r in results:
                    if r["success"]:
                        print(f"  {r['doc_id']:<14}  OK  ({r['records_extracted']} records extracted)")
                    else:
                        print(f"  {r['doc_id']:<14}  FAILED: {r['error']}")
                continue

            if lower.startswith("ingest contract "):
                rest = raw[len("ingest contract "):].strip()
                file_path = pathlib.Path(rest).expanduser()
                if not file_path.is_absolute():
                    file_path = pathlib.Path.cwd() / file_path
                if not file_path.exists():
                    print(f"  File not found: {file_path}")
                    continue
                doc_id = file_path.stem
                text = file_path.read_text(errors="replace")
                documents = [{"doc_id": doc_id, "text": text, "metadata": {"doc_type": "contract"}}]
                print(f"Ingesting {file_path.name} as doc_id={doc_id!r}...")
                try:
                    results = await client.ingest_documents(documents, timeout=180)
                except httpx.HTTPStatusError as exc:
                    print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
                    continue
                r = results[0]
                if r["success"]:
                    print(f"  {doc_id}  OK  ({r['records_extracted']} records extracted)")
                else:
                    print(f"  {doc_id}  FAILED: {r['error']}")
                continue

            if lower.startswith("check "):
                doc_id = raw[len("check "):].strip()
                if not doc_id:
                    print("  Usage: check <doc_id>")
                    continue

                print(f"Checking compliance for {doc_id!r}...")
                count = 0
                non_compliant = 0
                needs_review = 0

                try:
                    async with http.stream(
                        "POST",
                        f"{client.api_base}/applications/{client.app_name}/workflows/check-contract-compliance/stream",
                        json={"params": {"doc_id": doc_id}},
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
                    continue

                if count == 0:
                    print(f"  No clauses found for {doc_id!r}. Run 'ingest contracts' first.")
                else:
                    compliant = count - non_compliant - needs_review
                    print(f"\n  {count} findings saved.  "
                          f"non-compliant: {non_compliant}  "
                          f"needs-review: {needs_review}  "
                          f"compliant: {compliant}")
                continue

            if lower.startswith("report "):
                doc_id = raw[len("report "):].strip()
                if not doc_id:
                    print("  Usage: report <doc_id>")
                    continue

                try:
                    findings = await client.query_structured(
                        "clause_compliance_findings",
                        filters=[{"field": "doc_id", "op": "=", "value": doc_id}],
                    )
                except httpx.HTTPStatusError as exc:
                    print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
                    continue
                if not findings:
                    print(f"  No findings for {doc_id!r}. Run 'check {doc_id}' first.")
                    continue

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
                continue

            if lower == "alerts":
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
                    continue
                if not findings:
                    print("  No high/critical non-compliant findings.")
                    continue

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
                continue

            print("Thinking...")
            try:
                await client.query_stream(raw)
            except httpx.HTTPStatusError as exc:
                print(f"  ERROR: {exc.response.status_code} {exc.response.text}")


if __name__ == "__main__":
    asyncio.run(main())
