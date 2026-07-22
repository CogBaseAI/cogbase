"""Integration tests against the live contract-compliance REST API.

Assumes:
  - The API server is running (default: http://localhost:8000).
    Override with the COGBASE_API_URL environment variable.
  - The 'contract-compliance' application was created and all documents were
    ingested via demo.py ('create', 'ingest rules', 'ingest contracts').
  - The compliance workflow was run for all three contracts via demo.py
    ('check contract-001', 'check contract-002', 'check contract-003').
    The findings tests and some query tests require stored findings.

Run with::

    pytest examples/contract_compliance_demo/test_queries.py -v -s

What is verified
----------------
Setup
  - The application exists and is active

Ingestion (contract_metadata collection)
  - All 3 contracts produced exactly one metadata record each
  - Core fields (governing_law, termination_notice_days, parties, dates) are
    present and correct per contract

Clause extraction (contract_clauses collection)
  - Each contract produced a non-trivial number of clause records

Compliance findings (clause_compliance_findings collection)
  - Known non-compliant clauses are flagged with the right severity
  - contract-003 is the only contract with a critical-severity finding
  - Known compliant clauses in contract-003 are not flagged as non-compliant

Query: natural-language questions over all collections
"""

from __future__ import annotations

import asyncio
import logging
import os

import httpx
import pytest

_APP_NAME = "contract-compliance"
_API_BASE = os.environ.get("COGBASE_API_URL", "http://localhost:8000").rstrip("/")
_NAMESPACE = os.environ.get("COGBASE_NAMESPACE", "default")
_APP_BASE = f"{_API_BASE}/namespaces/{_NAMESPACE}/applications/{_APP_NAME}"

_CONTRACT_IDS = {"contract-001", "contract-002", "contract-003"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _query(text: str) -> dict:
    """POST to /query and return the parsed JSON response dict."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{_APP_BASE}/query",
            json={"text": text},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()


async def _fetch_collection(collection: str, filters: list[dict] | None = None) -> list[dict]:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{_APP_BASE}/collections/{collection}/query",
            json={"filters": filters or [], "fields": None},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["records"]


# ---------------------------------------------------------------------------
# Module-scoped fixtures: fetch records once per test session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def metadata_records() -> list[dict]:
    return asyncio.run(_fetch_collection("contract_metadata"))


@pytest.fixture(scope="module")
def clause_records() -> list[dict]:
    return asyncio.run(_fetch_collection("contract_clauses"))


@pytest.fixture(scope="module")
def findings_records() -> list[dict]:
    return asyncio.run(_fetch_collection("clause_compliance_findings"))


# ---------------------------------------------------------------------------
# 0. Sanity: application exists
# ---------------------------------------------------------------------------

def test_application_exists():
    """The contract-compliance application must exist and be active."""
    resp = httpx.get(f"{_APP_BASE}", timeout=10)
    assert resp.status_code == 200, (
        f"Expected application '{_APP_NAME}' to exist. "
        f"Got {resp.status_code}: {resp.text[:200]}"
    )
    assert resp.json()["status"] == "active"


# ---------------------------------------------------------------------------
# 1. Ingestion — contract_metadata
# ---------------------------------------------------------------------------

class TestIngestion:
    def test_all_three_contracts_ingested(self, metadata_records):
        assert len(metadata_records) == 3, (
            f"Expected 3 metadata records, got {len(metadata_records)}"
        )

    def test_each_doc_id_present(self, metadata_records):
        found_ids = {r["doc_id"] for r in metadata_records}
        assert found_ids == _CONTRACT_IDS

    def test_contract_types_extracted(self, metadata_records):
        for row in metadata_records:
            assert row.get("contract_type") is not None, (
                f"contract_type missing for {row['doc_id']}"
            )

    def test_parties_extracted_for_all_contracts(self, metadata_records):
        for row in metadata_records:
            parties = row.get("parties") or []
            assert len(parties) >= 2, (
                f"Expected at least 2 parties for {row['doc_id']}, got {parties}"
            )

    def test_contract_001_key_fields(self, metadata_records):
        row = next(r for r in metadata_records if r["doc_id"] == "contract-001")
        assert row["governing_law"] is not None
        assert "new york" in (row["governing_law"] or "").lower()
        assert row["termination_notice_days"] == 90
        assert row["effective_date"] == "2025-03-01"
        assert row["expiry_date"] == "2027-02-28"
        party_names = {p["name"] for p in row["parties"]}
        assert any("apex" in n.lower() for n in party_names)
        assert any("acme" in n.lower() for n in party_names)

    def test_contract_002_key_fields(self, metadata_records):
        row = next(r for r in metadata_records if r["doc_id"] == "contract-002")
        assert "delaware" in (row.get("governing_law") or "").lower()
        assert row["termination_notice_days"] == 14
        assert row["effective_date"] == "2025-06-01"
        assert row["expiry_date"] == "2026-05-31"
        party_names = {p["name"] for p in row["parties"]}
        assert any("meridian" in n.lower() for n in party_names)

    def test_contract_003_key_fields(self, metadata_records):
        row = next(r for r in metadata_records if r["doc_id"] == "contract-003")
        assert "new york" in (row.get("governing_law") or "").lower()
        assert row["termination_notice_days"] == 60
        assert row["effective_date"] == "2025-09-01"
        assert row["expiry_date"] == "2028-08-31"
        assert row.get("contract_value") == pytest.approx(840_000, rel=0.01)
        party_names = {p["name"] for p in row["parties"]}
        assert any("datasphere" in n.lower() for n in party_names)


# ---------------------------------------------------------------------------
# 2. Clause extraction — contract_clauses
# ---------------------------------------------------------------------------

class TestClauseExtraction:
    def test_clauses_extracted_for_all_contracts(self, clause_records):
        for doc_id in _CONTRACT_IDS:
            clauses = [r for r in clause_records if r["doc_id"] == doc_id]
            assert len(clauses) >= 3, (
                f"Expected at least 3 clauses for {doc_id}, got {len(clauses)}"
            )

    def test_clause_types_labeled(self, clause_records):
        labeled = [r for r in clause_records if r.get("clause_type") is not None]
        assert len(labeled) / len(clause_records) >= 0.7, (
            "Expected at least 70% of clauses to have a clause_type label"
        )

    def test_clause_text_is_non_empty(self, clause_records):
        for r in clause_records:
            assert r.get("text"), f"Empty clause text for clause_id={r.get('clause_id')}"


# ---------------------------------------------------------------------------
# 3. Compliance findings — clause_compliance_findings
# ---------------------------------------------------------------------------

class TestComplianceFindings:
    """Requires 'check contract-001/002/003' to have been run via demo.py."""

    def test_findings_present_for_all_contracts(self, findings_records):
        for doc_id in _CONTRACT_IDS:
            findings = [r for r in findings_records if r["doc_id"] == doc_id]
            assert len(findings) >= 3, (
                f"Expected findings for {doc_id}. "
                "Run 'check <doc_id>' in demo.py first."
            )

    def test_contract_001_liability_cap_non_compliant(self, findings_records):
        """contract-001 caps liability at 3 months; rule requires 12 months."""
        liability = [
            r for r in findings_records
            if r["doc_id"] == "contract-001"
            and (r.get("clause_type") or "").lower() in ("liability", "limitation_of_liability")
            and r["status"] == "non_compliant"
        ]
        assert liability, (
            "Expected a non_compliant liability finding for contract-001"
        )
        assert any(r["severity"] in ("high", "critical") for r in liability)

    def test_contract_001_breach_notification_non_compliant(self, findings_records):
        """contract-001 notifies within 48 hours; rule requires 24 hours."""
        breach = [
            r for r in findings_records
            if r["doc_id"] == "contract-001"
            and r["status"] == "non_compliant"
            and any(
                kw in (r.get("summary") or r.get("clause_type") or "").lower()
                for kw in ("breach", "notification", "privacy", "data")
            )
        ]
        assert breach, (
            "Expected a non_compliant data breach notification finding for contract-001"
        )

    def test_contract_002_has_multiple_non_compliant_findings(self, findings_records):
        """contract-002 has at least 4 known non-compliant clauses."""
        non_compliant = [
            r for r in findings_records
            if r["doc_id"] == "contract-002" and r["status"] == "non_compliant"
        ]
        assert len(non_compliant) >= 4, (
            f"Expected ≥4 non-compliant findings for contract-002, got {len(non_compliant)}"
        )

    def test_contract_002_payment_terms_non_compliant(self, findings_records):
        """contract-002 uses net-15; rule requires minimum net-30."""
        payment = [
            r for r in findings_records
            if r["doc_id"] == "contract-002"
            and (r.get("clause_type") or "").lower() == "payment"
            and r["status"] == "non_compliant"
        ]
        assert payment, (
            "Expected a non_compliant payment finding for contract-002 (net-15 < net-30)"
        )

    def test_contract_002_termination_non_compliant(self, findings_records):
        """contract-002 uses 14-day convenience notice; rule requires minimum 30 days."""
        termination = [
            r for r in findings_records
            if r["doc_id"] == "contract-002"
            and (r.get("clause_type") or "").lower() == "termination"
            and r["status"] == "non_compliant"
        ]
        assert termination, (
            "Expected a non_compliant termination finding for contract-002 (14 days < 30 days)"
        )

    def test_contract_003_has_critical_or_high_finding(self, findings_records):
        """contract-003 breach notification (72 hours) should be rated critical or high."""
        critical_or_high = [
            r for r in findings_records
            if r["doc_id"] == "contract-003" and r["severity"] in ("critical", "high")
        ]
        assert critical_or_high, (
            "Expected at least one critical or high finding for contract-003 "
            "(72-hour breach notification)"
        )

    def test_recommended_redline_present_for_non_compliant(self, findings_records):
        """Non-compliant findings should include suggested redline language."""
        non_compliant = [r for r in findings_records if r["status"] == "non_compliant"]
        with_redline = [r for r in non_compliant if r.get("recommended_redline")]
        assert len(with_redline) / len(non_compliant) >= 0.7, (
            "Expected ≥70% of non-compliant findings to have a recommended_redline"
        )


# ---------------------------------------------------------------------------
# 4. Query: natural-language questions
# ---------------------------------------------------------------------------

class TestQuery:
    async def test_non_compliant_clauses_for_contract_001(self):
        """Answer must name at least one non-compliant clause from contract-001."""
        result = await _query("show all non-compliant clauses for contract-001")
        answer = result["answer"].lower()
        assert "contract-001" in answer or "non-compliant" in answer or "non_compliant" in answer, (
            f"Expected non-compliant clause information. Got:\n{result['answer']}"
        )
        assert any(
            kw in answer
            for kw in ["liability", "breach", "consequential", "non-compliant", "violation"]
        ), f"Expected clause type or violation language. Got:\n{result['answer']}"

    async def test_high_severity_findings(self):
        """Answer must reference high or critical severity findings."""
        result = await _query("which clauses have high-severity findings?")
        answer = result["answer"].lower()
        assert "high" in answer, (
            f"Expected 'high' severity mention. Got:\n{result['answer']}"
        )
        assert any(
            kw in answer
            for kw in ["liability", "payment", "termination", "indemnification"]
        ), f"Expected a known high-severity clause type. Got:\n{result['answer']}"

    async def test_liability_clause_rule_violation(self):
        """Answer must cite the 12-month rule when explaining the liability finding."""
        result = await _query(
            "what company rule does the liability clause in contract-001 violate?"
        )
        answer = result["answer"].lower()
        assert "liability" in answer, (
            f"Expected 'liability' in answer. Got:\n{result['answer']}"
        )
        assert "12" in answer or "twelve" in answer or "months" in answer, (
            f"Expected 12-month rule reference. Got:\n{result['answer']}"
        )

    async def test_compliance_report_summary_contract_001(self):
        """Summary must cover multiple finding statuses for contract-001."""
        result = await _query("summarize the compliance report for contract-001")
        answer = result["answer"].lower()
        assert len(answer) > 100, "Expected substantive compliance summary"
        assert any(
            kw in answer
            for kw in ["non-compliant", "non_compliant", "compliant", "finding"]
        ), f"Expected compliance status language. Got:\n{result['answer']}"

    async def test_contracts_with_no_findings(self):
        """Answer should note that all three contracts have findings (none are clean)."""
        result = await _query("are there any contracts with no compliance findings?")
        answer = result["answer"].lower()
        assert len(answer) > 20, "Expected a substantive answer"
        assert any(
            kw in answer
            for kw in ["no", "all", "each", "every", "findings", "compliant"]
        ), f"Expected a meaningful response. Got:\n{result['answer']}"

    async def test_data_breach_notification_violations(self):
        """Both contract-001 (48h) and contract-003 (72h) violate the 24-hour rule."""
        result = await _query(
            "which contracts have non-compliant data breach notification clauses?"
        )
        answer = result["answer"].lower()
        assert "breach" in answer
        assert (
            "contract-001" in answer or "apex" in answer
            or "contract-003" in answer or "datasphere" in answer
        ), f"Expected contract-001 or contract-003 in answer. Got:\n{result['answer']}"
        assert any(
            kw in answer for kw in ["24", "48", "72", "hours", "hour"]
        ), f"Expected notification time reference. Got:\n{result['answer']}"

    async def test_governing_law_non_compliant(self):
        """contract-002 uses Delaware law; policy requires New York."""
        result = await _query(
            "which contracts use a governing law other than New York?"
        )
        answer = result["answer"].lower()
        assert (
            "contract-002" in answer or "meridian" in answer or "delaware" in answer
        ), f"Expected contract-002 / Delaware in answer. Got:\n{result['answer']}"

    async def test_ip_ownership_findings(self):
        """contract-002 and contract-003 retain deliverable IP with the Vendor."""
        result = await _query(
            "which contracts have issues with intellectual property ownership of deliverables?"
        )
        answer = result["answer"].lower()
        assert any(
            kw in answer
            for kw in ["ip", "intellectual property", "deliverable", "ownership", "vendor"]
        ), f"Expected IP ownership language. Got:\n{result['answer']}"
        assert (
            "contract-002" in answer or "meridian" in answer
            or "contract-003" in answer or "datasphere" in answer
        ), f"Expected contract-002 or contract-003 in answer. Got:\n{result['answer']}"

    async def test_critical_and_high_findings_across_all_contracts(self):
        """The alerts query should surface findings from all three contracts."""
        result = await _query("list all high and critical compliance findings across all contracts")
        answer = result["answer"].lower()
        assert len(answer) > 100, "Expected substantive findings list"
        doc_ids_mentioned = sum(
            1 for doc_id in ("contract-001", "contract-002", "contract-003")
            if doc_id in answer
        )
        assert doc_ids_mentioned >= 2, (
            f"Expected findings from ≥2 contracts. Got ({doc_ids_mentioned}):\n{result['answer']}"
        )

    async def test_termination_notice_comparison(self):
        """Answer must surface the 14-day outlier (contract-002) vs 60-90-day others."""
        result = await _query(
            "compare the termination notice periods across all three contracts"
        )
        answer = result["answer"].lower()
        assert any(
            period in answer for period in ["14", "60", "90"]
        ), f"Expected at least one notice period in answer. Got:\n{result['answer']}"
        assert (
            "contract-002" in answer or "meridian" in answer or "14" in answer
        ), f"Expected the non-compliant 14-day notice to be surfaced. Got:\n{result['answer']}"

    async def test_compliant_contracts_for_governing_law(self):
        """contract-001 and contract-003 both use New York law (compliant)."""
        result = await _query("which contracts have compliant governing law?")
        answer = result["answer"].lower()
        assert "new york" in answer, (
            f"Expected 'New York' in answer. Got:\n{result['answer']}"
        )
        assert (
            "contract-001" in answer or "apex" in answer
            or "contract-003" in answer or "datasphere" in answer
        ), f"Expected at least one New York contract named. Got:\n{result['answer']}"

    async def test_subprocessor_policy_violation(self):
        """contract-003 allows subprocessors without prior written consent."""
        result = await _query(
            "which contracts allow the vendor to use subprocessors without prior written consent?"
        )
        answer = result["answer"].lower()
        assert (
            "contract-003" in answer or "datasphere" in answer or "subprocessor" in answer
        ), f"Expected contract-003 / subprocessor reference. Got:\n{result['answer']}"

    async def test_payment_terms_summary(self):
        """Answer should cover net-30, net-15, and net-45 across the three contracts."""
        result = await _query("what are the payment terms for each contract?")
        answer = result["answer"].lower()
        payment_terms_found = sum(
            1 for term in ("net-30", "net-15", "net-45", "30 days", "15 days", "45 days")
            if term in answer
        )
        assert payment_terms_found >= 2, (
            f"Expected ≥2 payment terms in answer. Got:\n{result['answer']}"
        )

    async def test_redline_for_non_compliant_clause(self):
        """Answer should include suggested replacement language for a known violation."""
        result = await _query(
            "what redline language is recommended for the liability clause in contract-001?"
        )
        answer = result["answer"].lower()
        assert "liability" in answer, (
            f"Expected 'liability' in answer. Got:\n{result['answer']}"
        )
        assert any(
            kw in answer
            for kw in ["12 months", "twelve months", "fees paid", "preceding", "redline", "replace", "suggest"]
        ), f"Expected redline or replacement language. Got:\n{result['answer']}"
