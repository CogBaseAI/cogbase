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

from examples.contract_compliance_demo.contracts_data import (  # noqa: E402
    CONTRACTS_DOCUMENTS,
)
from examples.contract_compliance_demo.rules_data import RULES_DOCUMENTS  # noqa: E402
from examples.contract_compliance_demo.schema import (  # noqa: E402
    ClauseComplianceFinding,
    ContractClause,
    ContractMetadata,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_APP_NAME = "contract-compliance"
_CHAT_MODEL = "gpt-5.4-mini"
_EMBED_MODEL = "text-embedding-3-small"
_EMBED_DIM = 1536
_API_BASE = os.environ.get("COGBASE_API_URL", "http://localhost:8000").rstrip("/")

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
llm:
  provider: openai
  model: {_CHAT_MODEL}
embedding:
  provider: openai
  model: {_EMBED_MODEL}
  dimensions: {_EMBED_DIM}
chunk_collections:
  - name: rule_chunks
    description: >-
      Company policy and vendor contract standard passages. Use to retrieve rules,
      standards, and fallback positions relevant to a clause type or compliance topic.
    chunker:
      type: fixed
      chunk_size: 512
      overlap: 64
  - name: contract_chunks
    description: >-
      Contract text passage chunks. Use for detailed questions about specific
      contract terms, wording, or clauses.
    chunker:
      type: fixed
      chunk_size: 512
      overlap: 64
structured_collections:
  - name: contract_metadata
    description: >-
      Key facts per contract: parties, dates, value, governing law, termination
      notice period. One record per contract document.
    schema: contract_metadata_schema.json
    extractor:
      type: llm
      prompt: contract_metadata_prompt.txt
  - name: contract_clauses
    description: >-
      Individual clauses extracted from contracts. Each record is one clause with
      its type and verbatim text. Filter by doc_id to retrieve all clauses for a
      contract, or filter by clause_type to find clauses of a specific category.
    schema: contract_clauses_schema.json
    extractor:
      type: llm
      extract_as_list: true
      list_field: clauses
      item_id_field: clause_id
      prompt: contract_clauses_prompt.txt
pipeline:
  steps:
    - tool: chunk-embed-upsert
      collection: rule_chunks
      when:
        metadata:
          doc_type: rules
    - tool: chunk-embed-upsert
      collection: contract_chunks
      when:
        metadata:
          doc_type: contract
    - tool: extract-structured
      collection: contract_metadata
      when:
        metadata:
          doc_type: contract
    - tool: extract-structured
      collection: contract_clauses
      when:
        metadata:
          doc_type: contract
workflows:
  - name: check-contract-compliance
    trigger:
      type: manual
    input_schema:
      doc_id: string
    output_collections:
      - name: clause_compliance_findings
        schema: clause_compliance_findings_schema.json
        primary_fields: [clause_id]
        description: >-
          Clause-level compliance findings. Each record captures whether a contract
          clause complies with company policy, with severity, summary, and redline.
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
    """Build an in-memory ZIP bundle: config.yaml + schemas + prompts."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("config.yaml", _CONFIG_YAML)
        zf.writestr("contract_metadata_schema.json", json.dumps(ContractMetadata.model_json_schema(), indent=2))
        zf.writestr("contract_metadata_prompt.txt", _CONTRACT_METADATA_SYSTEM_PROMPT)
        zf.writestr("contract_clauses_schema.json", json.dumps(ContractClause.model_json_schema(), indent=2))
        zf.writestr("contract_clauses_prompt.txt", _CONTRACT_CLAUSES_SYSTEM_PROMPT)
        zf.writestr("clause_compliance_findings_schema.json", json.dumps(ClauseComplianceFinding.model_json_schema(), indent=2))
        zf.writestr("compliance_judge_prompt.txt", _JUDGE_SYSTEM_PROMPT)
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
        timeout=180,
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


async def _list_collections(client: httpx.AsyncClient) -> dict:
    resp = await client.get(
        f"{_API_BASE}/applications/{_APP_NAME}/collections", timeout=10
    )
    resp.raise_for_status()
    return resp.json()


async def _query_structured_rest(
    client: httpx.AsyncClient,
    collection: str,
    filters: list[dict] | None = None,
) -> list[dict]:
    resp = await client.post(
        f"{_API_BASE}/applications/{_APP_NAME}/collections/{collection}/query",
        json={"filters": filters or [], "fields": None},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["records"]



# ---------------------------------------------------------------------------
# Main async loop
# ---------------------------------------------------------------------------


async def main() -> None:
    print()
    print("Contract Compliance Demo (REST API)")
    print("=" * 42)
    print(f"  model:   {_CHAT_MODEL}")
    print(f"  embed:   {_EMBED_MODEL}")
    print(f"  api:     {_API_BASE}")

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

            # ---- list -------------------------------------------------------
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
                        print(f"  {app['name']:<28}  status: {app['status']}")
                continue

            # ---- create -----------------------------------------------------
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

            # ---- delete <name> ----------------------------------------------
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

            # ---- reset ------------------------------------------------------
            if lower == "reset":
                confirm = input("  Delete application and all data? [y/N] ").strip().lower()
                if confirm == "y":
                    await _delete_app(client)
                    print("  Application deleted. Restart the demo to start fresh.")
                    break
                continue

            # ---- list collections -------------------------------------------
            if lower == "list collections":
                try:
                    cols = await _list_collections(client)
                except httpx.HTTPStatusError as exc:
                    print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
                    continue
                print(f"  structured: {cols.get('structured', [])}")
                print(f"  vector:     {cols.get('vector', [])}")
                continue

            # ---- query structured [<collection>] ----------------------------
            if lower == "query structured" or lower.startswith("query structured "):
                collection = (
                    raw[len("query structured "):].strip()
                    if lower.startswith("query structured ")
                    else _DEFAULT_STRUCTURED_COLLECTION
                )
                print(f"Querying structured collection '{collection}'...")

                try:
                    records = await _query_structured_rest(client, collection)
                except httpx.HTTPStatusError as exc:
                    print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
                    continue

                if not records:
                    print("  No records found.")
                else:
                    print(json.dumps(records, indent=2))
                continue

            # ---- ingest rules [<path>] --------------------------------------
            if lower == "ingest rules" or lower.startswith("ingest rules "):
                rest = raw[len("ingest rules"):].strip()
                if rest:
                    # ingest from file
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
                    # ingest built-in rules
                    documents = [
                        {"doc_id": doc.doc_id, "text": doc.text, "metadata": dict(doc.metadata)}
                        for doc in RULES_DOCUMENTS
                    ]
                    print(f"Ingesting {len(documents)} built-in rule documents...")

                try:
                    results = await _ingest_documents(client, documents)
                except httpx.HTTPStatusError as exc:
                    print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
                    continue
                for r in results:
                    if r["success"]:
                        print(f"  {r['doc_id']:<14}  OK  (rule chunks indexed)")
                    else:
                        print(f"  {r['doc_id']:<14}  FAILED: {r['error']}")
                continue

            # ---- ingest contracts -------------------------------------------
            if lower == "ingest contracts":
                print(f"Ingesting {len(CONTRACTS_DOCUMENTS)} built-in contracts...")
                documents = [
                    {"doc_id": doc.doc_id, "text": doc.text, "metadata": dict(doc.metadata)}
                    for doc in CONTRACTS_DOCUMENTS
                ]
                try:
                    results = await _ingest_documents(client, documents)
                except httpx.HTTPStatusError as exc:
                    print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
                    continue

                for r in results:
                    if r["success"]:
                        print(f"  {r['doc_id']:<14}  OK  ({r['records_extracted']} records extracted)")
                    else:
                        print(f"  {r['doc_id']:<14}  FAILED: {r['error']}")

                continue

            # ---- ingest contract <path> -------------------------------------
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
                    results = await _ingest_documents(client, documents)
                except httpx.HTTPStatusError as exc:
                    print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
                    continue
                r = results[0]
                if r["success"]:
                    print(f"  {doc_id}  OK  ({r['records_extracted']} records extracted)")
                else:
                    print(f"  {doc_id}  FAILED: {r['error']}")
                continue

            # ---- check <doc_id> ---------------------------------------------
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
                    async with client.stream(
                        "POST",
                        f"{_API_BASE}/applications/{_APP_NAME}/workflows/check-contract-compliance/stream",
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
                                finding = json.loads(payload)
                            except json.JSONDecodeError:
                                continue
                            if "error" in finding:
                                print(f"\n  ERROR: {finding['error']}")
                                continue
                            count += 1
                            status_val = finding.get("status", "")
                            status_marker = {
                                "compliant": "COMPLIANT    ",
                                "non_compliant": "NON-COMPLIANT",
                                "needs_review": "NEEDS REVIEW ",
                                "not_applicable": "N/A          ",
                            }.get(status_val, status_val.upper())
                            sev = (finding.get("severity") or "").upper()
                            print(f"  {finding.get('clause_id', ''):<28}  {status_marker}  {sev:<8}  {finding.get('summary', '')}")
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

            # ---- report <doc_id> --------------------------------------------
            if lower.startswith("report "):
                doc_id = raw[len("report "):].strip()
                if not doc_id:
                    print("  Usage: report <doc_id>")
                    continue

                try:
                    findings = await _query_structured_rest(
                        client,
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

                for status in ("non_compliant", "needs_review", "compliant", "not_applicable"):
                    group = by_status.get(status, [])
                    if not group:
                        continue
                    label = status.replace("_", " ").upper()
                    print(f"\n  [{label}] — {len(group)} clause(s)")
                    for f in group:
                        sev = (f.get("severity") or "").upper()
                        cid = f.get("clause_id", "")
                        summary = f.get("summary", "")
                        print(f"    {cid:<30}  {sev:<8}  {summary}")
                        redline = f.get("recommended_redline")
                        if redline:
                            print(f"      Suggested: {redline[:120]}")
                print()
                continue

            # ---- alerts -----------------------------------------------------
            if lower == "alerts":
                try:
                    findings = await _query_structured_rest(
                        client,
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

            # ---- natural-language question / anything else ------------------
            print("Thinking...")
            try:
                await _query_stream(client, raw)
            except httpx.HTTPStatusError as exc:
                print(f"  ERROR: {exc.response.status_code} {exc.response.text}")


if __name__ == "__main__":
    asyncio.run(main())
