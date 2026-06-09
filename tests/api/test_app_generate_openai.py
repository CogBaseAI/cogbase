"""Live integration tests for api/routers/app_generate.py.

LLM and embedding backends are loaded from .env.yaml (same config that
``api/main.py`` uses).  Falls back to OpenAI via ``OPENAI_API_KEY`` when
.env.yaml is absent or contains no ``llm`` / ``embedding`` section.
"""

from __future__ import annotations

import logging
import json
from unittest.mock import MagicMock

import pytest
import yaml

from api.models import ChatMessage, GenerateChatRequest
from api.routers.app_generate import chat
from cogbase.core.app_generator import _collect_save_targets
from cogbase.config.config import AppConfig
from tests.live_setup import make_llm, make_embedding

logger = logging.getLogger(__name__)

_llm = make_llm()
_embedder = make_embedding()

openai = pytest.importorskip("openai", reason="openai package not installed")

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(_llm is None, reason="No LLM configured: set llm in .env.yaml or OPENAI_API_KEY"),
]


@pytest.fixture(scope="module")
def llm():
    return _llm


@pytest.fixture(scope="module")
def embedder():
    return _embedder


_MAX_ROUNDS = 5  # initial turn + up to 4 confirmations

_CONTRACT_CONVERSATION: list[dict] = [
    {
        "role": "user",
        "content": (
            "I want to build a contract analysis app. Users upload commercial "
            "contracts (PDFs) and ask questions about vendors, dates, and "
            "payment terms."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Got it. I'll design a single pipeline with chunk-embed-upsert, "
            "extract-structured, and document-embed-upsert. For the structured "
            "collection 'contracts' I propose these fields:\n"
            "- vendor_name — name of the vendor\n"
            "- effective_date — contract start date (ISO 8601)\n"
            "- expiry_date — contract end date (ISO 8601)\n"
            "- total_value — total contract value in USD\n"
            "- governing_law — jurisdiction governing the contract\n"
            "Confirm or edit?"
        ),
    },
    {
        "role": "user",
        "content": "Looks good, those five fields are exactly right.",
    },
]


# ---------------------------------------------------------------------------
# End-to-end: drive the chat endpoint until it returns a validated config.
# ---------------------------------------------------------------------------


class TestChatEndpointLive:
    async def test_chat_returns_text_for_open_question(self, llm):
        body = GenerateChatRequest(
            text="What CogBase pipeline step type produces structured records?",
            history=[],
        )
        response = await chat(body, MagicMock(llm=llm))
        assert response.content
        assert response.config_yaml is None

    async def test_chat_generates_validated_config_yaml(self, llm):
        history = [ChatMessage(**m) for m in _CONTRACT_CONVERSATION[:-1]]
        body = GenerateChatRequest(
            text=(
                "Yes, those five fields are right. Please generate the schema "
                "and the full app config now."
            ),
            history=history,
        )
        response = await chat(body, MagicMock(llm=llm))

        assert response.config_yaml, (
            "expected the agent loop to call propose_app_config and return a "
            f"validated config_yaml. Final assistant text: {response.content!r}"
        )
        config = AppConfig.from_yaml(response.config_yaml)
        assert config.name
        assert config.pipelines

    async def test_full_conversation_contract_app_from_scratch(self, llm):
        """Multi-turn conversation for a contract analysis app.

        Turn 1 asks the model to propose fields; turn 2 confirms and requests
        generation; subsequent turns say "Yes" until a config_yaml is returned
        (up to _MAX_ROUNDS total). Verifies the final config is structurally
        valid with doc_id injected.
        """
        history: list[ChatMessage] = []
        final_response = None

        for round_num in range(_MAX_ROUNDS):
            if round_num == 0:
                text = (
                    "I want to build a contract analysis app. "
                    "Users upload PDF contracts and ask about vendor names, payment terms, "
                    "and expiry dates. What structured fields should I extract?"
                )
            elif round_num == 1:
                text = (
                    "Those fields look exactly right. "
                    "Please generate the extraction schema and the full app config now."
                )
            else:
                text = "Yes"

            response = await chat(
                GenerateChatRequest(text=text, history=history),
                MagicMock(llm=llm),
            )
            logger.info("round %d content=%s", round_num, response.content)
            logger.info("round %d config_yaml=%s", round_num, response.config_yaml)
            history = history + [
                ChatMessage(role="user", content=text),
                ChatMessage(role="assistant", content=response.content),
            ]
            final_response = response
            if final_response.config_yaml:
                break

        assert final_response is not None
        assert final_response.config_yaml, (
            f"expected config_yaml for the contract app within {_MAX_ROUNDS} round(s).\n"
            f"last response: {final_response.content!r}"
        )
        config = AppConfig.from_yaml(final_response.config_yaml)
        assert config.name
        assert config.pipelines

        first_step = config.pipelines[0].steps[0]
        assert getattr(first_step, "tool", None) == "chunk-embed-upsert", (
            f"first pipeline step must be chunk-embed-upsert, "
            f"got {getattr(first_step, 'tool', None)!r}"
        )

        data = yaml.safe_load(final_response.config_yaml)
        extract_targets = {
            step.get("collection")
            for p in data.get("pipelines", [])
            for step in p.get("steps", [])
            if step.get("tool") == "extract-structured"
        }
        assert extract_targets, "expected at least one extract-structured step"

        for sc in data.get("structured_collections", []):
            if sc["name"] not in extract_targets:
                continue
            schema_str = sc.get("schema")
            assert schema_str, f"collection {sc['name']!r} is missing its schema"
            record_schema = json.loads(schema_str)
            props = record_schema.get("properties", {})
            assert "doc_id" in props, f"doc_id not injected in {sc['name']!r}"
            user_fields = [k for k in props if k != "doc_id"]
            assert user_fields, f"no user-defined fields found in {sc['name']!r}"

    async def test_full_conversation_workflow_app_from_scratch(self, llm):
        """Multi-turn conversation for an app that requires a workflow.

        Turn 1 describes a clause-level compliance app with an explicit workflow
        requirement. Subsequent turns confirm the design with "yes" until the LLM
        returns a config_yaml (up to 4 confirmation rounds). Verifies the final
        config contains at least one workflow and that all structured-save target
        collections have schemas set.
        """
        text = (
            "Build a contract compliance app. "
            "The app needs to extract each clause from uploaded contracts. "
            "Also iterates over contract_clauses, runs LLM judgment on each clause, "
            "and saves a compliance finding per clause."
        )

        history: list[ChatMessage] = []
        final_response = None

        for round_num in range(_MAX_ROUNDS):
            response = await chat(
                GenerateChatRequest(text=text, history=history),
                MagicMock(llm=llm),
            )
            logger.info("round %d content=%s", round_num, response.content)
            logger.info("round %d config_yaml=%s", round_num, response.config_yaml)
            history = history + [
                ChatMessage(role="user", content=text),
                ChatMessage(role="assistant", content=response.content),
            ]
            final_response = response
            if final_response.config_yaml:
                break
            text = 'Yes'

        assert final_response is not None
        assert final_response.config_yaml, (
            f"expected config_yaml for the compliance workflow app within "
            f"{_MAX_ROUNDS} round(s).\n"
            f"last response: {final_response.content!r}"
        )
        config = AppConfig.from_yaml(final_response.config_yaml)
        assert config.name
        assert config.pipelines
        assert config.workflows, "expected at least one workflow in the config"

        data = yaml.safe_load(final_response.config_yaml)
        sc_by_name = {sc["name"]: sc for sc in data.get("structured_collections", [])}

        save_targets: set[str] = set()
        for wf in data.get("workflows", []):
            _collect_save_targets(wf.get("steps", []), save_targets)

        assert save_targets, "expected at least one structured-save step in the workflow"
        for target in save_targets:
            sc = sc_by_name.get(target)
            assert sc, (
                f"workflow save target '{target}' not declared in structured_collections"
            )
            schema_str = sc.get("schema")
            assert schema_str, f"save target '{target}' is missing its schema"
            record_schema = json.loads(schema_str)
            assert record_schema.get("properties"), (
                f"save target '{target}' schema has no properties"
            )


# ---------------------------------------------------------------------------
# End-to-end: chat → config → ingest → workflow/query.
# Each class covers one demo scenario; assertions are meaning-based so they
# hold regardless of what collection/field names the LLM chose.
# ---------------------------------------------------------------------------


class TestContractComplianceEndToEndLive:
    """Contract compliance demo: two-pipeline app + compliance workflow.

    Chat generates the config (rules pipeline for policy docs, contracts
    pipeline for clause extraction, compliance workflow that fans out over
    clauses and saves findings). The test ingests the demo data, runs the
    workflow for contract-001, and verifies queries against live findings.
    """

    async def test_ingest_workflow_and_query(self, llm, embedder):
        from api.factory import build_app
        from api.system_resources import SystemResources
        from cogbase.core.models import Document
        from cogbase.core.query_runner import QueryResult
        from cogbase.stores.structured.memory import InMemoryStructuredStore
        from cogbase.stores.vector.faiss_store import FAISSVectorStore

        # Short documents — enough coverage for the assertions, fast to ingest.
        # Rules: 12-month liability cap (mutual), mutual consequential exclusion,
        #        net-30 minimum payment, ≤1.5%/month late interest.
        # Contract-001: 3-month liability cap (NON-COMPLIANT), vendor-only
        #   consequential exclusion (NON-COMPLIANT), net-30 payment (COMPLIANT),
        #   New York governing law (needed for the governing-law query).
        RULES_DOCUMENTS = [
            Document(
                doc_id="rules-001",
                metadata={"doc_type": "rules", "topic": "liability"},
                text="""\
COMPANY VENDOR CONTRACT STANDARDS — LIABILITY AND INDEMNIFICATION

1. LIABILITY CAP
1.1  Each party's total aggregate liability shall not exceed the total fees paid
in the twelve (12) months immediately preceding the event giving rise to the claim.

2. EXCLUSION OF CONSEQUENTIAL DAMAGES
2.1  Neither party shall be liable for indirect, incidental, special, or consequential
damages, including loss of profits or data.
2.2  The consequential damages exclusion must be mutual. Excluding them for the Vendor
only while preserving them for the Company is not acceptable.

3. INDEMNIFICATION
3.1  Indemnification obligations must be mutual. Requiring the Company to indemnify
the Vendor for the Vendor's own IP infringement is not acceptable.
""",
            ),
            Document(
                doc_id="rules-002",
                metadata={"doc_type": "rules", "topic": "payment_terms"},
                text="""\
COMPANY VENDOR CONTRACT STANDARDS — PAYMENT TERMS

1. STANDARD PAYMENT TERMS
1.1  Payment terms shorter than net-30 are not acceptable without CFO approval.

2. LATE PAYMENT INTEREST
2.1  Late-payment interest shall not exceed 1.5% per month on overdue amounts.
""",
            ),
        ]
        CONTRACTS_DOCUMENTS = [
            Document(
                doc_id="contract-001",
                metadata={"doc_type": "contract", "source": "apex_cloud_saas_agreement.txt"},
                text="""\
CLOUD SOFTWARE SERVICES AGREEMENT
Effective Date: March 1, 2025
Vendor: Apex Cloud Solutions Inc. ("Vendor")
Company: Acme Corporation ("Company")

ARTICLE 2 — PAYMENT TERMS
2.1  Company shall pay each undisputed invoice within thirty (30) days of receipt.
2.2  Overdue amounts accrue interest at 1.5% per month until paid in full.

ARTICLE 5 — LIMITATION OF LIABILITY
5.1  VENDOR'S TOTAL AGGREGATE LIABILITY SHALL NOT EXCEED THE TOTAL FEES PAID BY
COMPANY TO VENDOR IN THE THREE (3) MONTHS IMMEDIATELY PRECEDING THE EVENT GIVING
RISE TO THE CLAIM.
5.2  IN NO EVENT SHALL VENDOR BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL,
PUNITIVE, OR CONSEQUENTIAL DAMAGES, INCLUDING LOSS OF PROFITS OR DATA.

ARTICLE 9 — GOVERNING LAW
9.1  This Agreement shall be governed by the laws of the State of New York,
without regard to conflict-of-law principles.

IN WITNESS WHEREOF, the parties have executed this Agreement as of the Effective Date.
""",
            ),
        ]

        text = (
            "Build a contract compliance app with two document types: "
            "Policy rule documents and Vendor contracts. Check whether "
            "the clauses in a contract is compliant with policy."
        )
        history: list[ChatMessage] = []
        final_response = None

        for round_num in range(_MAX_ROUNDS):
            response = await chat(
                GenerateChatRequest(text=text, history=history),
                MagicMock(llm=llm),
            )
            logger.info("round %d content=%s", round_num, response.content)
            logger.info("round %d config_yaml=%s", round_num, response.config_yaml)
            history = history + [
                ChatMessage(role="user", content=text),
                ChatMessage(role="assistant", content=response.content),
            ]
            final_response = response
            if final_response.config_yaml:
                break
            text = "Yes"

        assert final_response is not None
        assert final_response.config_yaml, (
            f"expected config_yaml for the compliance workflow app within "
            f"{_MAX_ROUNDS} round(s).\n"
            f"last response: {final_response.content!r}"
        )

        config = AppConfig.from_yaml(final_response.config_yaml)
        assert config.workflows, "generated config must have at least one workflow"

        data = yaml.safe_load(final_response.config_yaml)
        save_targets: set[str] = set()
        for wf in data.get("workflows", []):
            _collect_save_targets(wf.get("steps", []), save_targets)
        assert save_targets, "workflow must contain at least one structured-save step"

        workflow_cfg = config.workflows[0]
        workflow_input_key = "doc_id"

        system = SystemResources(
            structured_store=InMemoryStructuredStore(),
            vector_store=FAISSVectorStore(),
            llm=llm,
            embedder=embedder,
        )
        app = await build_app(config, system=system, app_id=config.name, app_status="new")

        results = await app.ingest_documents(
            RULES_DOCUMENTS + CONTRACTS_DOCUMENTS
        )
        failed = [r for r in results if not r.success]
        assert not failed, (
            f"{len(failed)} document(s) failed ingestion: "
            + ", ".join(f"{r.doc_id}: {r.error}" for r in failed)
        )
        contract_results = [r for r in results if r.doc_id.startswith("contract-")]
        assert all(r.records_extracted > 0 for r in contract_results), (
            "expected each contract to produce at least one extracted record — "
            "check that the contracts pipeline includes an extract-structured step"
        )

        findings: list[dict] = []
        workflow = app.get_workflow(workflow_cfg.name)
        async for record in workflow.run({workflow_input_key: "contract-001"}):
            findings.append(record)

        if not findings:
            diag_parts = [
                f"workflow '{workflow_cfg.name}' produced no findings for contract-001.\n",
                "--- generated config YAML ---\n",
                final_response.config_yaml or "(none)",
                "\n--- ingest results ---",
            ]
            for r in results:
                diag_parts.append(
                    f"  {r.doc_id}: success={r.success} records_extracted={r.records_extracted}"
                )
            diag_parts.append("\n--- structured store contents ---")
            for coll_name in system.structured_store._schemas:
                try:
                    all_records = await system.structured_store.query(coll_name)
                    contract_records = [
                        rec for rec in all_records if rec.get("doc_id") == "contract-001"
                    ]
                    diag_parts.append(
                        f"  {coll_name}: {len(all_records)} total record(s), "
                        f"{len(contract_records)} for contract-001"
                    )
                    if contract_records:
                        diag_parts.append(
                            f"    sample: {json.dumps(contract_records[:2], default=str)}"
                        )
                except Exception as exc:
                    diag_parts.append(f"  {coll_name}: query failed — {exc}")
            diag_parts.append(
                f"\n--- workflow steps ---\n{yaml.dump(workflow_cfg.model_dump(mode='json', exclude_none=True))}"
            )
            assert False, "\n".join(diag_parts)

        # Signals for non-compliance and compliance (LLM phrasing varies).
        _NON_SIGNALS = (
            "non-compliant", "non_compliant", "noncompliant",
            "not compliant", "does not comply", "violat", "breach",
            "not met", "fails to", "not satisfy",
        )
        _POS_SIGNALS = (
            "complies", "compliant", "satisf", "conform", "adhere",
            "passes", "meets", "met",
        )
        def _is_non_compliant(finding: dict) -> bool:
            if "compliant" in finding:
                return finding["compliant"] is False
            val = str(finding).lower()
            return any(sig in val for sig in _NON_SIGNALS)

        def _is_compliant(finding: dict) -> bool:
            if "compliant" in finding:
                return finding["compliant"] is True
            s = str(finding).lower()
            for sig in _NON_SIGNALS:
                s = s.replace(sig, "")
            return any(sig in s for sig in _POS_SIGNALS)

        async def _llm_judge(question: str) -> tuple[bool, str]:
            """Ask the LLM to judge findings when the deterministic check is inconclusive."""
            from cogbase.llms.base import ChatMessage
            result = await llm.complete([
                ChatMessage(role="system", content=(
                    "You are a strict compliance judge. Answer only with a JSON object: "
                    '{"answer": true/false, "reason": "<one sentence>"}'
                )),
                ChatMessage(role="user", content=(
                    f"{question}\n\nFindings:\n{json.dumps(findings, indent=2)}"
                )),
            ], max_tokens=200, temperature=0)
            try:
                parsed = json.loads(result.content)
                return bool(parsed["answer"]), str(parsed.get("reason", ""))
            except Exception:
                text = (result.content or "").lower()
                return "true" in text and "false" not in text[:20], result.content or ""

        non_compliant = [f for f in findings if _is_non_compliant(f)]
        if not non_compliant:
            verdict, reason = await _llm_judge(
                "Do any of these findings indicate a non-compliant clause? "
                "A 3-month liability cap that violates a 12-month policy rule should be non-compliant."
            )
            assert verdict, (
                "expected at least one non-compliant finding for contract-001 — "
                "the liability cap (3 months) should violate the 12-month policy rule. "
                f"LLM judge also said no non-compliant finding: {reason}. "
                f"findings: {findings}"
            )

        compliant = [f for f in findings if _is_compliant(f)]
        if not compliant:
            verdict, reason = await _llm_judge(
                "Do any of these findings indicate a compliant clause?"
            )
            assert verdict, (
                f"expected at least one compliant finding. "
                f"LLM judge also said no compliant finding: {reason}. "
                f"findings: {findings}"
            )

        async def _query(text: str) -> str:
            result = None
            async for chunk in app.query_stream(text):
                if isinstance(chunk, QueryResult):
                    result = chunk
            assert result is not None, f"query_stream produced no QueryResult for: {text!r}"
            return result.answer

        answer1 = (await _query(
            "what are the non-compliant clauses in contract-001?"
        )).lower()
        assert any(
            kw in answer1
            for kw in ("non-compliant", "non_compliant", "violat", "liability", "breach", "finding")
        ), f"expected a non-compliance reference in the answer:\n{answer1}"

        answer2 = (await _query(
            "what is the governing law for contract-001?"
        )).lower()
        assert "new york" in answer2, (
            f"expected 'New York' governing law for contract-001:\n{answer2}"
        )


class TestContractAnalystEndToEndLive:
    """Contract analyst demo: single-pipeline app with nested object extraction.

    Chat generates the config requesting Party (list of named parties with role
    and jurisdiction) and PaymentTerms (structured payment clause) as nested
    objects. Verifies that the generated JSON schema contains nested definitions
    and that queries over the ingested data return correct answers.
    """

    async def test_ingest_and_query(self, llm, embedder):
        from api.factory import build_app
        from api.system_resources import SystemResources
        from cogbase.core.models import Document
        from cogbase.core.query_runner import QueryResult
        from cogbase.stores.structured.memory import InMemoryStructuredStore
        from cogbase.stores.vector.faiss_store import FAISSVectorStore

        # Two short contracts designed to exercise nested Party / PaymentTerms extraction.
        #
        # saas-003: Nexus Security (Provider, Delaware) + Meridian Analytics (Customer, California)
        #   Payment: USD 180,000 upfront, due 2024-01-15, late penalty 1.5 %/month
        #   Liability cap: USD 2,000,000 / expires 2025-12-31
        #
        # saas-005: Apex Systems (Provider, Texas) + Meridian Analytics (Customer, California)
        #   Payment: USD 360,000 net-30, due 2023-10-31, late penalty 2 %/month
        #   Liability cap: USD 500,000 / expires 2025-09-30
        #
        # Both expire before 2026-01-01  →  answer1 must name both.
        # Only saas-003 has cap > $1M    →  answer2 must name saas-003 / Nexus.
        # saas-005 payment is net-30     →  answer3 must mention "30" or "net".
        # Nexus party jurisdiction is Delaware → answer4 must mention Delaware.
        CONTRACTS = {
            "saas-003": """\
CLOUD SECURITY PLATFORM SUBSCRIPTION AGREEMENT
Contract ID: CSPSA-2024-0512
Effective Date: January 1, 2024
Expiry Date: December 31, 2025
Customer: Meridian Analytics Inc., incorporated in the State of California ("Customer" / buyer)
Provider: Nexus Security Ltd., incorporated in the State of Delaware ("Provider" / seller)

1. FEES AND PAYMENT
Annual subscription fee: USD 180,000, payable upfront.
Payment due date: January 15, 2024 (within 15 days of the Effective Date).
Overdue amounts accrue interest at 1.5% per month until paid in full.

2. LIMITATION OF LIABILITY
2.1  Provider's total aggregate liability for all claims shall not exceed USD 2,000,000.
2.2  Neither party shall be liable for indirect, incidental, or consequential damages.

3. TERMINATION
Either party may terminate for convenience upon ninety (90) days' prior written notice.

4. GOVERNING LAW
This Agreement is governed by the laws of the State of Delaware.
""",
            "saas-005": """\
ENTERPRISE WORKFLOW MANAGEMENT SUBSCRIPTION AGREEMENT
Contract Reference: EWMSA-2023-0312
Effective Date: October 1, 2023
Expiry Date: September 30, 2025
Customer: Meridian Analytics Inc., incorporated in the State of California ("Customer" / buyer)
Provider: Apex Systems Inc., incorporated in the State of Texas ("Provider" / seller)

1. FEES AND PAYMENT
Annual subscription fee: USD 360,000, net-30 payment terms.
Payment due date: October 31, 2023 (within 30 days of invoice receipt).
Overdue amounts accrue a late fee of 2% per month on the outstanding balance.

2. LIMITATION OF LIABILITY
Provider's aggregate liability shall not exceed USD 500,000.
Neither party shall be liable for indirect, consequential, or punitive damages.

3. TERMINATION
3.1  Either party may terminate for convenience upon one hundred eighty (180) days' prior
written notice.
3.2  Either party may terminate for cause if the breaching party fails to cure a material
breach within 30 days of written notice.

4. GOVERNING LAW
This Agreement is governed by the laws of the State of Texas.
""",
        }

        text = (
            "I need a contract analysis app for SaaS vendor agreements. "
            "Extract the following from each contract:\n"
            "- vendor name, customer name, effective date (YYYY-MM-DD), expiry date (YYYY-MM-DD), "
            "governing law jurisdiction, liability cap amount\n"
            "- parties: a list of Party objects, each with: name (full legal name of the party), "
            "role (e.g. buyer, seller, licensor, licensee), "
            "jurisdiction (state or country of incorporation)\n"
            "- payment_terms: a PaymentTerms object with: "
            "schedule (e.g. net-30, upfront, milestone-based), "
            "due_date (YYYY-MM-DD if stated), "
            "late_penalty (verbatim penalty clause if present), "
            "verbatim (the verbatim payment clause from the contract)"
        )
        history: list[ChatMessage] = []
        final_response = None

        for round_num in range(_MAX_ROUNDS):
            response = await chat(
                GenerateChatRequest(text=text, history=history),
                MagicMock(llm=llm),
            )
            logger.info("round %d content=%s", round_num, response.content)
            logger.info("round %d config_yaml=%s", round_num, response.config_yaml)
            history = history + [
                ChatMessage(role="user", content=text),
                ChatMessage(role="assistant", content=response.content),
            ]
            final_response = response
            if final_response.config_yaml:
                break
            text = "Yes"

        assert final_response is not None
        assert final_response.config_yaml, (
            f"expected config_yaml for the contract analyst app within "
            f"{_MAX_ROUNDS} round(s).\n"
            f"last response: {final_response.content!r}"
        )

        config = AppConfig.from_yaml(final_response.config_yaml)
        assert config.pipelines, "generated config must have at least one pipeline"

        # Verify the generated JSON schema contains nested object definitions for
        # parties (array of objects) and payment_terms (object).
        raw = yaml.safe_load(final_response.config_yaml)
        extract_collections = {
            step.get("collection")
            for p in raw.get("pipelines", [])
            for step in p.get("steps", [])
            if step.get("tool") == "extract-structured"
        }
        assert extract_collections, "expected at least one extract-structured step"

        for sc in raw.get("structured_collections", []):
            if sc["name"] not in extract_collections:
                continue
            schema_str = sc.get("schema")
            assert schema_str, f"collection {sc['name']!r} is missing its schema"
            record_schema = json.loads(schema_str)
            props = record_schema.get("properties", {})
            defs = record_schema.get("$defs", record_schema.get("definitions", {}))

            def _is_nested(prop: dict) -> bool:
                if "$ref" in prop:
                    return True
                t = prop.get("type")
                if t == "object" and "properties" in prop:
                    return True
                if t == "array":
                    items = prop.get("items", {})
                    return items.get("type") == "object" or "$ref" in items
                return False

            nested_props = [k for k, v in props.items() if isinstance(v, dict) and _is_nested(v)]
            assert nested_props or defs, (
                f"expected nested object/array fields (parties, payment_terms) in "
                f"schema {sc['name']!r}; got properties: {list(props)}"
            )

        system = SystemResources(
            structured_store=InMemoryStructuredStore(),
            vector_store=FAISSVectorStore(),
            llm=llm,
            embedder=embedder,
        )
        app = await build_app(config, system=system, app_id=config.name, app_status="new")

        documents = [
            Document(doc_id=doc_id, text=contract_text)
            for doc_id, contract_text in CONTRACTS.items()
        ]
        results = await app.ingest_documents(documents)
        failed = [r for r in results if not r.success]
        assert not failed, (
            f"{len(failed)} document(s) failed ingestion: "
            + ", ".join(f"{r.doc_id}: {r.error}" for r in failed)
        )
        assert all(r.records_extracted > 0 for r in results), (
            "expected each contract to produce at least one extracted record"
        )

        # Verify that at least one extracted record contains a nested dict or list
        # value (populated Party or PaymentTerms object).
        nested_record_found = False
        for coll_name in extract_collections:
            try:
                records = await system.structured_store.query(coll_name)
                for rec in records:
                    if any(isinstance(v, (dict, list)) for v in rec.values()):
                        nested_record_found = True
                        break
            except Exception:
                pass
            if nested_record_found:
                break
        assert nested_record_found, (
            "expected at least one extracted record to contain a nested dict or list "
            "(Party or PaymentTerms object). Check that the LLM schema includes nested types."
        )

        async def _query(q: str) -> str:
            result = None
            async for chunk in app.query_stream(q):
                if isinstance(chunk, QueryResult):
                    result = chunk
            assert result is not None, f"query_stream produced no QueryResult for: {q!r}"
            return result.answer

        answer1 = (await _query("which contracts expire before 2026-01-01?")).lower()
        expiring_hits = sum(
            any(kw in answer1 for kw in ids)
            for ids in [
                ("saas-003", "nexus"),
                ("saas-005", "apex"),
            ]
        )
        assert expiring_hits >= 2, (
            f"expected both contracts (saas-003, saas-005) expiring before 2026 to be named:\n{answer1}"
        )

        answer2 = (await _query(
            "which contracts have a liability cap above 1 million dollars?"
        )).lower()
        assert any(
            kw in answer2
            for kw in ("saas-003", "nexus", "2,000,000", "2000000", "2 million")
        ), f"expected saas-003 / Nexus / $2M cap in answer:\n{answer2}"

        answer3 = (await _query(
            "what are the payment terms for saas-005?"
        )).lower()
        assert any(
            kw in answer3
            for kw in ("net-30", "net 30", "30 day", "30-day", "30 days", "within 30")
        ), f"expected net-30 payment terms for saas-005 in answer:\n{answer3}"

        answer4 = (await _query(
            "which party in saas-003 is incorporated in Delaware?"
        )).lower()
        assert any(
            kw in answer4
            for kw in ("nexus", "provider", "delaware")
        ), f"expected Nexus / Provider / Delaware for saas-003 party jurisdiction:\n{answer4}"
