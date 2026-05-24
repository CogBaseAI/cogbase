"""Legal Case Preparation Demo — drive CogBase via the REST API.

Usage
-----
    # Start the API server first:
    uvicorn api.main:app --reload --log-level info

    # Then run the demo (from the repo root):
    python examples/legal_case_prep_demo/demo.py

Requires OPENAI_API_KEY in a .env file at the repo root (or in the environment).
Set COGBASE_API_URL to override the default http://localhost:8000.

NOTE: The contradiction and gap workflows require persistent store backends
(SQLite + FAISS).  Configure cogbase_system.yaml with structured_store.type=sqlite
and vector_store.type=faiss, or set COGBASE_CONFIG to point to your system config.

Interactive commands
--------------------
    /ingest_demo_case               Ingest the built-in nine-document case bundle
    /inventory                      Show the document inventory
    /timeline [<issue>]             Show the chronological timeline of events
    /cast                           Show the cast of characters
    /facts [<issue>]                Show extracted facts (optionally filtered by issue)
    /reference_table [<kind>]       Show the structured data reference table
    /detect_contradictions <issue>  Run the contradiction-detection workflow
    /find_gaps <issue>              Run the evidence-gap workflow
    /contradictions [<issue>]       List saved contradictions
    /gaps [<issue>]                 List saved evidence gaps

Any other input is sent as a natural-language query to the legal-case-prep app.
"""

from __future__ import annotations

import asyncio
import io
import json
import pathlib
import sys
import zipfile
from collections import defaultdict

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
from examples.legal_case_prep_demo.case_data import CASE_DOCUMENTS  # noqa: E402
from examples.legal_case_prep_demo.schema import (  # noqa: E402
    CaseDocument,
    CaseDocumentRecord,
    Contradiction,
    ContradictionList,
    Entity,
    EntityRecord,
    EvidenceGap,
    EvidenceGapList,
    Fact,
    FactRecord,
    StructuredDataItem,
    StructuredDataItemRecord,
    TimelineEvent,
    TimelineEventRecord,
)

configure_logging()

_APP_NAME = "legal-case-prep"
_DEFAULT_STRUCTURED_COLLECTION = "case_documents"

_PROMPT_FILES = [
    "case_document_prompt.txt",
    "timeline_event_prompt.txt",
    "entity_prompt.txt",
    "fact_prompt.txt",
    "structured_data_prompt.txt",
    "contradiction_judge_prompt.txt",
    "evidence_gap_judge_prompt.txt",
]


def _build_bundle() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(_DEMO_DIR / "config.yaml", "config.yaml")

        # Pipeline extraction & record schemas
        zf.writestr("case_document_record_schema.json",
                    json.dumps(CaseDocumentRecord.model_json_schema(), indent=2))
        zf.writestr("case_document_extraction_schema.json",
                    json.dumps(CaseDocument.model_json_schema(), indent=2))

        zf.writestr("timeline_event_record_schema.json",
                    json.dumps(TimelineEventRecord.model_json_schema(), indent=2))
        zf.writestr("timeline_event_extraction_schema.json",
                    json.dumps(TimelineEvent.model_json_schema(), indent=2))

        zf.writestr("entity_record_schema.json",
                    json.dumps(EntityRecord.model_json_schema(), indent=2))
        zf.writestr("entity_extraction_schema.json",
                    json.dumps(Entity.model_json_schema(), indent=2))

        zf.writestr("fact_record_schema.json",
                    json.dumps(FactRecord.model_json_schema(), indent=2))
        zf.writestr("fact_extraction_schema.json",
                    json.dumps(Fact.model_json_schema(), indent=2))

        zf.writestr("structured_data_record_schema.json",
                    json.dumps(StructuredDataItemRecord.model_json_schema(), indent=2))
        zf.writestr("structured_data_extraction_schema.json",
                    json.dumps(StructuredDataItem.model_json_schema(), indent=2))

        # Workflow output & list-wrapper schemas
        zf.writestr("contradiction_record_schema.json",
                    json.dumps(Contradiction.model_json_schema(), indent=2))
        zf.writestr("contradiction_list_schema.json",
                    json.dumps(ContradictionList.model_json_schema(), indent=2))
        zf.writestr("evidence_gap_record_schema.json",
                    json.dumps(EvidenceGap.model_json_schema(), indent=2))
        zf.writestr("evidence_gap_list_schema.json",
                    json.dumps(EvidenceGapList.model_json_schema(), indent=2))

        # Prompts
        for name in _PROMPT_FILES:
            zf.write(_DEMO_DIR / name, name)
    return buf.getvalue()


async def _ingest_built_in(client: CogBaseClient) -> None:
    print(f"Ingesting {len(CASE_DOCUMENTS)} built-in case documents...")
    documents = [
        {"doc_id": doc.doc_id, "text": doc.text, "metadata": dict(doc.metadata)}
        for doc in CASE_DOCUMENTS
    ]
    try:
        results = await client.upload_text_documents(documents, timeout=600)
    except httpx.HTTPStatusError as exc:
        print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
        return
    for r in results:
        if r["success"]:
            print(f"  {r['doc_id']:<10}  OK")
        else:
            print(f"  {r['doc_id']:<10}  FAILED: {r['error']}")


async def _show_inventory(client: CogBaseClient) -> None:
    records = await client.query_structured_collection("case_documents")
    if not records:
        print("  Inventory is empty. Run '/ingest_demo_case' first.")
        return
    records.sort(key=lambda r: r.get("document_date") or "")
    print(f"\nDocument inventory — {len(records)} document(s)")
    print("-" * 90)
    for r in records:
        date = r.get("document_date") or "(undated)"
        print(f"  {r['doc_id']:<10}  {date:<12}  {r.get('doc_type', ''):<20}  {r.get('title', '')}")
        if r.get("relevance_tag"):
            print(f"             relevance:   {r['relevance_tag']}")
        if r.get("summary"):
            print(f"             summary:     {r['summary']}")
    print()


async def _show_timeline(client: CogBaseClient, issue: str = "") -> None:
    filters = [{"field": "issue", "op": "=", "value": issue}] if issue else None
    records = await client.query_structured_collection("timeline_events", filters)
    if not records:
        msg = f" for issue {issue!r}" if issue else ""
        print(f"  No timeline events{msg}. Run '/ingest_demo_case' first.")
        return
    records.sort(key=lambda r: (r.get("date_start") or "", r.get("doc_id") or ""))
    label = f" for {issue}" if issue else ""
    print(f"\nChronological timeline{label} — {len(records)} event(s)")
    print("-" * 90)
    for r in records:
        date = r.get("date_start") or "????"
        end = f"–{r['date_end']}" if r.get("date_end") else ""
        kind = r.get("event_type") or "?"
        actors = ", ".join(r.get("actors") or [])
        desc = r.get("description") or ""
        src = r.get("doc_id", "")
        print(f"  {date}{end:<12} [{kind:<13}] {desc}")
        print(f"             actors:  {actors}")
        print(f"             source:  {src} — \"{(r.get('source_quote') or '').strip()[:120]}\"")
    print()


async def _show_cast(client: CogBaseClient) -> None:
    records = await client.query_structured_collection("entities")
    if not records:
        print("  No entities found. Run '/ingest_demo_case' first.")
        return

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        key = (r.get("name", ""), r.get("entity_type", ""))
        grouped[key].append(r)

    print(f"\nCast of characters — {len(grouped)} distinct name(s) across {len(records)} mentions")
    print("-" * 90)
    for (name, entity_type), mentions in sorted(grouped.items()):
        roles = sorted({m.get("role") or "" for m in mentions if m.get("role")})
        titles = sorted({m.get("title_at_time") or "" for m in mentions if m.get("title_at_time")})
        related = sorted({rel for m in mentions for rel in (m.get("related_to") or [])})
        docs = sorted({m.get("doc_id", "") for m in mentions})
        print(f"  {name}  ({entity_type})")
        print(f"     roles:       {', '.join(roles) or '—'}")
        if titles:
            print(f"     titles:      {', '.join(titles)}")
        if related:
            print(f"     related to:  {', '.join(related)}")
        print(f"     appears in:  {', '.join(docs)}")
    print()


async def _show_facts(client: CogBaseClient, issue: str = "") -> None:
    filters = [{"field": "issue", "op": "=", "value": issue}] if issue else None
    records = await client.query_structured_collection("facts", filters)
    if not records:
        msg = f" for issue {issue!r}" if issue else ""
        print(f"  No facts{msg}. Run '/ingest_demo_case' first.")
        return

    by_issue: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_issue[r.get("issue") or "(untagged)"].append(r)

    label = f" for {issue}" if issue else ""
    print(f"\nFact matrix{label} — {len(records)} fact(s) across {len(by_issue)} issue(s)")
    print("-" * 90)
    for issue_key, group in sorted(by_issue.items()):
        print(f"\n[issue: {issue_key}]")
        for f in group:
            print(f"  - {f.get('asserting_party', '?')} ({f.get('doc_id', '')}): "
                  f"{f.get('assertion', '')}")
            print(f"        \"{(f.get('source_quote') or '').strip()[:140]}\"")
    print()


async def _show_reference_table(client: CogBaseClient, kind: str = "") -> None:
    filters = [{"field": "kind", "op": "=", "value": kind}] if kind else None
    records = await client.query_structured_collection("structured_data", filters)
    if not records:
        msg = f" for kind {kind!r}" if kind else ""
        print(f"  No structured items{msg}. Run '/ingest_demo_case' first.")
        return

    by_kind: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_kind[r.get("kind") or "other"].append(r)

    print(f"\nStructured reference table — {len(records)} item(s) across {len(by_kind)} kind(s)")
    print("-" * 90)
    for k, group in sorted(by_kind.items()):
        print(f"\n[{k}]")
        for it in group:
            amount = f" {it.get('amount')} {it.get('currency') or ''}".rstrip() if it.get("amount") else ""
            date = f"  date={it.get('date')}" if it.get("date") else ""
            party = f"  party={it.get('party_responsible')}" if it.get("party_responsible") else ""
            ref = f"  ref={it.get('clause_reference')}" if it.get("clause_reference") else ""
            print(f"  {it.get('doc_id', ''):<10}{amount}{date}{party}{ref}")
            print(f"      {it.get('description', '')}")
    print()


async def _run_workflow_stream(
    client: CogBaseClient, workflow_name: str, params: dict
) -> list[dict]:
    saved: list[dict] = []
    try:
        async with client._http.stream(
            "POST",
            f"{client.api_base}/applications/{client.app_name}/workflows/{workflow_name}/stream",
            json={"params": params},
            timeout=600,
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
                record = data.get("record", data)
                saved.append(record)
                print(json.dumps(record, indent=2))
    except httpx.HTTPStatusError as exc:
        print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
    return saved


async def _detect_contradictions(client: CogBaseClient, issue: str) -> None:
    print(f"Detecting contradictions for issue {issue!r}...")
    saved = await _run_workflow_stream(
        client, "detect-contradictions", {"issue": issue}
    )
    if not saved:
        print("  No contradictions returned.")
        return
    print(f"\n  Saved {len(saved)} contradiction(s) to the contradictions collection.")


async def _find_gaps(client: CogBaseClient, issue: str) -> None:
    print(f"Identifying evidence gaps for issue {issue!r}...")
    saved = await _run_workflow_stream(
        client, "identify-evidence-gaps", {"issue": issue}
    )
    if not saved:
        print("  No evidence gaps returned.")
        return
    print(f"\n  Saved {len(saved)} gap(s) to the evidence_gaps collection.")


async def _show_contradictions(client: CogBaseClient, issue: str = "") -> None:
    filters = [{"field": "issue", "op": "=", "value": issue}] if issue else None
    records = await client.query_structured_collection("contradictions", filters)
    if not records:
        msg = f" for issue {issue!r}" if issue else ""
        print(f"  No saved contradictions{msg}. Run '/detect_contradictions <issue>' first.")
        return
    rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    records.sort(key=lambda r: rank.get(r.get("significance", ""), 9))
    print(f"\nContradictions — {len(records)} finding(s)")
    print("-" * 90)
    for r in records:
        sev = (r.get("significance") or "").upper()
        print(f"  [{sev:<8}] issue={r.get('issue', '')}  ({r.get('doc_a_id', '')} vs {r.get('doc_b_id', '')})")
        print(f"       A ({r.get('asserting_party_a', '')}): \"{(r.get('quote_a') or '').strip()[:140]}\"")
        print(f"       B ({r.get('asserting_party_b', '')}): \"{(r.get('quote_b') or '').strip()[:140]}\"")
        print(f"       → {r.get('explanation', '')}")
    print()


async def _show_gaps(client: CogBaseClient, issue: str = "") -> None:
    filters = [{"field": "issue", "op": "=", "value": issue}] if issue else None
    records = await client.query_structured_collection("evidence_gaps", filters)
    if not records:
        msg = f" for issue {issue!r}" if issue else ""
        print(f"  No saved evidence gaps{msg}. Run '/find_gaps <issue>' first.")
        return
    rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    records.sort(key=lambda r: rank.get(r.get("potential_impact", ""), 9))
    print(f"\nEvidence gaps — {len(records)} finding(s)")
    print("-" * 90)
    for r in records:
        impact = (r.get("potential_impact") or "").upper()
        print(f"  [{impact:<8}] issue={r.get('issue', '')}  asserted by {r.get('asserting_party', '')} in {r.get('doc_id', '')}")
        print(f"       missing:    {r.get('gap_description', '')}")
        print(f"       suggested:  {r.get('suggested_action', '')}")
    print()


async def main() -> None:
    print()
    print("Legal Case Preparation Demo (REST API)")
    print("=" * 42)

    async with CogBaseClient() as client:
        client.use_app(_APP_NAME)
        print(f"  api:    {client.api_base}")
        print()

        app_info = await cmd_startup(client, _build_bundle())
        if app_info is None:
            return
        print()

        async def handler(raw: str, lower: str) -> bool:
            if lower == "/ingest_demo_case":
                await _ingest_built_in(client)
                return True

            if lower == "/inventory":
                try:
                    await _show_inventory(client)
                except httpx.HTTPStatusError as exc:
                    print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
                return True

            if lower == "/timeline" or lower.startswith("/timeline "):
                arg = raw[len("/timeline"):].strip()
                try:
                    await _show_timeline(client, arg)
                except httpx.HTTPStatusError as exc:
                    print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
                return True

            if lower == "/cast":
                try:
                    await _show_cast(client)
                except httpx.HTTPStatusError as exc:
                    print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
                return True

            if lower == "/facts" or lower.startswith("/facts "):
                arg = raw[len("/facts"):].strip()
                try:
                    await _show_facts(client, arg)
                except httpx.HTTPStatusError as exc:
                    print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
                return True

            if lower == "/reference_table" or lower.startswith("/reference_table "):
                arg = raw[len("/reference_table"):].strip()
                try:
                    await _show_reference_table(client, arg)
                except httpx.HTTPStatusError as exc:
                    print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
                return True

            if lower.startswith("/detect_contradictions"):
                issue = raw[len("/detect_contradictions"):].strip()
                if not issue:
                    print("  Usage: /detect_contradictions <issue>")
                    return True
                await _detect_contradictions(client, issue)
                return True

            if lower.startswith("/find_gaps"):
                issue = raw[len("/find_gaps"):].strip()
                if not issue:
                    print("  Usage: /find_gaps <issue>")
                    return True
                await _find_gaps(client, issue)
                return True

            if lower == "/contradictions" or lower.startswith("/contradictions "):
                arg = raw[len("/contradictions"):].strip()
                try:
                    await _show_contradictions(client, arg)
                except httpx.HTTPStatusError as exc:
                    print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
                return True

            if lower == "/gaps" or lower.startswith("/gaps "):
                arg = raw[len("/gaps"):].strip()
                try:
                    await _show_gaps(client, arg)
                except httpx.HTTPStatusError as exc:
                    print(f"  ERROR: {exc.response.status_code} {exc.response.text}")
                return True

            return False

        await run_interactive_loop(
            client, _build_bundle,
            default_collection=_DEFAULT_STRUCTURED_COLLECTION,
            handler=handler,
            extra_commands=[
                "/ingest_demo_case",
                "/inventory", "/timeline", "/cast", "/facts", "/reference_table",
                "/detect_contradictions", "/find_gaps",
                "/contradictions", "/gaps",
            ],
        )


if __name__ == "__main__":
    asyncio.run(main())
