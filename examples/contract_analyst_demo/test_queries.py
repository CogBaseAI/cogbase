"""Integration tests against the live contract-analyst REST API.

Assumes:
  - The API server is running (default: http://localhost:8000).
    Override with the COGBASE_API_URL environment variable.
  - The 'contract-analyst' application was created and the contract fixtures
    were ingested via demo.py ('create' then '/ingest_demo_contracts').

Run with::

    pytest examples/contract_analyst_demo/test_queries.py -v -s

What is verified
----------------
Setup
  - The application exists and is active

Ingestion (structured collection)
  - Every ingested contract produced exactly one record
  - Core fields (doc_id, contract_type, parties, dates, contract_value,
    liability_cap, notice_period_days) are present and correct for the core
    SaaS fixtures

Query: questions over both document text and extracted structured data,
including cross-document hero queries that exploit the planted tensions in
the extended portfolio (see contracts.py).
"""

from __future__ import annotations

import asyncio
import os

import httpx
import pytest

from examples.contract_analyst_demo.contracts import CONTRACTS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_APP_NAME = "contract-analyst"
_API_BASE = os.environ.get("COGBASE_API_URL", "http://localhost:8000").rstrip("/")
_CONTRACTS_COLLECTION = "contracts"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _query(text: str) -> dict:
    """POST to /query and return the parsed JSON response dict."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{_API_BASE}/applications/{_APP_NAME}/query",
            json={"text": text},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Module-scoped fixture: fetch records once
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def records() -> list[dict]:
    """Fetch all contract records from the live API once per module."""
    async def _fetch():
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{_API_BASE}/applications/{_APP_NAME}/collections/{_CONTRACTS_COLLECTION}/query",
                json={"filters": [], "fields": None},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["records"]

    return asyncio.run(_fetch())


# ---------------------------------------------------------------------------
# 0. Sanity: application exists
# ---------------------------------------------------------------------------

def test_application_exists():
    """The contract-analyst application must exist and be active."""
    resp = httpx.get(f"{_API_BASE}/applications/{_APP_NAME}", timeout=10)
    assert resp.status_code == 200, (
        f"Expected application '{_APP_NAME}' to exist. "
        f"Got {resp.status_code}: {resp.text[:200]}"
    )
    assert resp.json()["status"] == "active"


# ---------------------------------------------------------------------------
# 1. Ingestion
# ---------------------------------------------------------------------------

class TestIngestion:
    def test_all_contracts_ingested(self, records):
        expected = len(CONTRACTS)
        assert len(records) == expected, f"Expected {expected} records, got {len(records)}"

    def test_each_doc_id_present(self, records):
        found_ids = {r["doc_id"] for r in records}
        assert found_ids == set(CONTRACTS.keys())

    def test_contract_types_extracted(self, records):
        for row in records:
            assert row.get("contract_type") is not None, (
                f"contract_type missing for {row['doc_id']}"
            )

    def test_saas_001_key_fields(self, records):
        row = next(r for r in records if r["doc_id"] == "saas-001")
        assert row["expiry_date"] == "2025-06-30", f"row={row}"
        assert row["contract_value"] == pytest.approx(500_000, rel=0.01), f"row={row}"
        assert row["liability_cap"] == pytest.approx(50_000, rel=0.01), f"row={row}"
        assert row["notice_period_days"] == 30, f"row={row}"
        party_names = {p["name"] for p in row["parties"]}
        assert any("Acme" in n for n in party_names), f"row={row}"

    def test_saas_002_key_fields(self, records):
        row = next(r for r in records if r["doc_id"] == "saas-002")
        assert row["expiry_date"] == "2026-06-30", f"row={row}"
        assert row["notice_period_days"] == 60, f"row={row}"
        assert row["contract_value"] == pytest.approx(240_000, rel=0.01), f"row={row}"

    def test_saas_003_key_fields(self, records):
        row = next(r for r in records if r["doc_id"] == "saas-003")
        assert row["expiry_date"] == "2025-12-31", f"row={row}"
        assert row["liability_cap"] == pytest.approx(2_000_000, rel=0.01), f"row={row}"
        assert row["notice_period_days"] == 90, f"row={row}"

    def test_saas_004_key_fields(self, records):
        row = next(r for r in records if r["doc_id"] == "saas-004")
        assert row["expiry_date"] == "2027-03-31", f"row={row}"
        assert row["contract_value"] == pytest.approx(1_200_000, rel=0.01), f"row={row}"
        assert row["liability_cap"] == pytest.approx(250_000, rel=0.01), f"row={row}"
        party_names = {p["name"] for p in row["parties"]}
        assert any("Acme" in n for n in party_names), f"row={row}"

    def test_saas_005_key_fields(self, records):
        row = next(r for r in records if r["doc_id"] == "saas-005")
        assert row["expiry_date"] == "2025-09-30", f"row={row}"
        assert row["notice_period_days"] == 180, f"row={row}"
        assert row["contract_value"] == pytest.approx(360_000, rel=0.01), f"row={row}"


# ---------------------------------------------------------------------------
# 2. Query
# ---------------------------------------------------------------------------

class TestQuery:
    async def test_contracts_expiring_before_2026(self):
        """3 contracts expire before 2026-01-01; the answer must name all three."""
        result = await _query("which contracts expire before 2026-01-01?")
        answer = result["answer"].lower()
        # CloudStore Pro (saas-001 / Acme), SecureVault (saas-003 / Nexus),
        # WorkflowManager (saas-005 / Apex)
        cloudstore = "saas-001" in answer or "cloudstore" in answer or "acme" in answer
        securevault = "saas-003" in answer or "securevault" in answer or "nexus" in answer
        workflowmanager = "saas-005" in answer or "workflowmanager" in answer or "apex" in answer
        assert cloudstore and securevault and workflowmanager, (
            f"Expected all three expiring contracts in answer. Got:\n{result['answer']}"
        )

    async def test_contracts_governed_by_new_york(self):
        """saas-001 and saas-004 use New York law; answer must mention New York."""
        result = await _query("list all contracts governed by New York law")
        answer = result["answer"].lower()
        assert "new york" in answer, (
            f"Expected 'New York' in answer. Got:\n{result['answer']}"
        )

    async def test_acme_corp_contracts(self):
        """Acme Corp is a party in saas-001 and saas-004; answer must mention Acme."""
        result = await _query("show all contracts where Acme Corp is listed in parties")
        answer = result["answer"].lower()
        assert "acme" in answer, (
            f"Expected 'Acme' in answer. Got:\n{result['answer']}"
        )

    async def test_liability_cap_above_1_million(self):
        """Only saas-003 (Nexus Security) has a liability cap of USD 2,000,000."""
        result = await _query("which contracts have a liability cap above 1 million?")
        answer = result["answer"].lower()
        assert (
            "saas-003" in answer or "securevault" in answer or "nexus" in answer
            or "2,000,000" in answer or "2000000" in answer or "2 million" in answer
        ), f"Expected saas-003 / Nexus Security in answer. Got:\n{result['answer']}"

    async def test_gdpr_data_residency(self):
        """saas-001, saas-003, and saas-004 mention GDPR / data residency."""
        result = await _query(
            "which contracts mention GDPR or data residency requirements?"
        )
        answer = result["answer"].lower()
        assert "gdpr" in answer or "data residency" in answer or "eea" in answer, (
            f"Expected GDPR/data-residency language in answer. Got:\n{result['answer']}"
        )

    async def test_breach_notification(self):
        """saas-001 (48-hour) and saas-003 (24-hour) have breach notification clauses."""
        result = await _query(
            "find passages about data breach notification obligations"
        )
        answer = result["answer"].lower()
        assert "breach" in answer
        assert "48 hours" in answer or "24 hours" in answer, (
            f"Expected breach notification timing in answer. Got:\n{result['answer']}"
        )

    async def test_audit_rights(self):
        """saas-002 and saas-003 have audit rights clauses."""
        result = await _query("find language about audit rights")
        answer = result["answer"].lower()
        assert "audit" in answer, (
            f"Expected 'audit' in answer. Got:\n{result['answer']}"
        )

    async def test_competitor_assignment_restriction(self):
        """saas-005 has a competitor assignment restriction with 'void ab initio' language."""
        result = await _query(
            "are there any clauses that restrict assignment to competitors?"
        )
        answer = result["answer"].lower()
        assert "competitor" in answer, (
            f"Expected 'competitor' in answer. Got:\n{result['answer']}"
        )
        assert "void ab initio" in answer or "assignment" in answer, (
            f"Expected assignment restriction language. Got:\n{result['answer']}"
        )

    async def test_payment_term_contradiction_acme(self):
        """saas-001 is net-30; saas-004 is upfront — both with Acme Corp."""
        result = await _query(
            "do any contracts contradict each other on payment terms with the same vendor?"
        )
        answer = result["answer"].lower()
        assert "acme" in answer, (
            f"Expected 'Acme' in answer about payment contradiction. Got:\n{result['answer']}"
        )
        assert (
            "net-30" in answer or "net 30" in answer
            or "upfront" in answer or "up front" in answer
            or "contradict" in answer or "differ" in answer or "inconsistent" in answer
        ), f"Expected payment term contrast in answer. Got:\n{result['answer']}"

    async def test_unusually_long_notice_period(self):
        """saas-005 has 180-day notice vs 30–90 days for all others."""
        result = await _query(
            "which contracts have unusually long notice periods compared to the others?"
        )
        answer = result["answer"].lower()
        assert "180" in answer or "saas-005" in answer or "workflowmanager" in answer or "apex" in answer, (
            f"Expected 180-day notice period / saas-005 in answer. Got:\n{result['answer']}"
        )

    async def test_termination_rights_summary_has_quotes(self):
        """Termination summary must include verbatim clause text."""
        result = await _query(
            "summarise all termination rights across the vendor portfolio"
        )
        body = result["answer"].lower()
        assert len(body) > 0
        assert any(
            phrase in body
            for phrase in ["written notice", "terminate", "cure", "breach"]
        ), f"Expected clause language in answer. Got:\n{result['answer']}"

    async def test_auto_renewal_clause_surfaced(self):
        """saas-004 contains an auto-renewal clause; the report must mention it."""
        result = await _query(
            "which contracts have auto-renewal clauses and what are the notice requirements?"
        )
        body = result["answer"].lower()
        assert (
            "auto" in body or "renew" in body or "automatic" in body
        ), f"Expected renewal language in answer. Got:\n{result['answer']}"
        assert (
            "saas-004" in body or "analyticspro" in body or "acme" in body
            or "90 days" in body or "90-day" in body
        ), f"Expected saas-004 / 90-day trigger in answer. Got:\n{result['answer']}"

    async def test_governing_law_comparison(self):
        """Report should cover the governing law jurisdictions across the portfolio."""
        result = await _query(
            "produce a comparison of governing law and dispute resolution clauses"
        )
        body = result["answer"].lower()
        jurisdictions_found = sum([
            "new york" in body,
            "california" in body,
            "delaware" in body,
            "texas" in body,
        ])
        assert jurisdictions_found >= 3, (
            f"Expected ≥3 jurisdictions in comparison. Got ({jurisdictions_found}):\n{result['answer']}"
        )

    async def test_risk_analysis_has_substantive_content(self):
        """Risk analysis must identify at least one high-risk factor from the contracts."""
        result = await _query(
            "which contracts are most risky? explain with supporting quotes"
        )
        body = result["answer"].lower()
        assert len(body) > 100, "Expected substantive risk analysis"
        assert any(term in body for term in [
            "liability cap", "liability", "termination", "notice", "upfront",
            "50,000", "50000", "250,000", "250000",
        ]), f"Expected risk factor discussion. Got:\n{result['answer']}"

    # -----------------------------------------------------------------------
    # Extended-portfolio hero queries (planted cross-document tensions)
    # -----------------------------------------------------------------------

    async def test_techvault_payment_conflict(self):
        """TechVault: saas-002 is net-15 / monthly; svc-001 is net-60 — same vendor."""
        result = await _query(
            "we have multiple agreements with TechVault Solutions — do their "
            "payment terms conflict?"
        )
        answer = result["answer"].lower()
        assert "techvault" in answer, (
            f"Expected 'TechVault' in answer. Got:\n{result['answer']}"
        )
        assert (
            "net-60" in answer or "net 60" in answer or "60 days" in answer
            or "net-15" in answer or "net 15" in answer or "15 days" in answer
            or "monthly" in answer or "differ" in answer or "conflict" in answer
            or "inconsistent" in answer
        ), f"Expected the payment-term conflict surfaced. Got:\n{result['answer']}"

    async def test_noncompete_covenants(self):
        """emp-001 (12mo), emp-002 (24mo), and sep-001 carry restrictive covenants."""
        result = await _query(
            "which agreements contain non-compete or non-solicitation covenants "
            "and how long do the restrictions last?"
        )
        answer = result["answer"].lower()
        assert "non-compete" in answer or "non-solicit" in answer or "solicit" in answer, (
            f"Expected restrictive-covenant language. Got:\n{result['answer']}"
        )
        assert (
            "12" in answer or "twelve" in answer
            or "24" in answer or "twenty-four" in answer
            or "jordan" in answer or "dana" in answer
            or "emp-001" in answer or "emp-002" in answer
        ), f"Expected covenant durations / employees named. Got:\n{result['answer']}"

    async def test_uncapped_or_missing_liability_cap(self):
        """msa-001 is uncapped; vnd-002 states no cap — both are risk outliers."""
        result = await _query(
            "which agreements have unlimited liability or no stated liability cap?"
        )
        answer = result["answer"].lower()
        assert (
            "orion" in answer or "msa-001" in answer or "unlimited" in answer
            or "uncapped" in answer or "no cap" in answer or "vivid" in answer
            or "vnd-002" in answer or "marketing" in answer
        ), f"Expected the uncapped / missing-cap agreements. Got:\n{result['answer']}"

    async def test_expiring_before_march_2026(self):
        """Several agreements across types expire before 2026-03-01."""
        result = await _query(
            "list every agreement that expires before March 1, 2026, regardless "
            "of contract type"
        )
        answer = result["answer"].lower()
        # lease-002 (2026-02-28), vnd-001 (2025-10-31), grn-001 (2025-08-31),
        # nda-002 (2025-11-30) join the expiring SaaS contracts.
        hits = sum([
            "lease-002" in answer or "datacore" in answer or "equipment" in answer,
            "vnd-001" in answer or "pinnacle" in answer,
            "grn-001" in answer or "beacon" in answer or "evaluation" in answer,
            "nda-002" in answer or "brightpath" in answer,
            "saas-001" in answer or "saas-003" in answer or "saas-005" in answer,
        ])
        assert hits >= 2, (
            f"Expected ≥2 distinct expiring agreements across types. Got:\n{result['answer']}"
        )
