"""Live integration tests for api/routers/app_generate.py against the real OpenAI API."""

from __future__ import annotations

import logging
import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from api.models import ChatMessage, GenerateChatRequest
from api.routers.app_generate import chat
from cogbase.core.app_generator import _collect_save_targets
from cogbase.config.config import AppConfig

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass


openai = pytest.importorskip("openai", reason="openai package not installed")

_openai_api_key = os.environ.get("OPENAI_API_KEY", "")
pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not _openai_api_key,
        reason="OPENAI_API_KEY not set in .env",
    ),
]

_MODEL = "gpt-5.4-mini"
_MINI_MODEL = "gpt-5.4-mini"


@pytest.fixture(scope="module")
def llm():
    from cogbase.llms.openai import OpenAILLM

    client = openai.AsyncOpenAI(api_key=_openai_api_key)
    return OpenAILLM(client, model=_MODEL, mini_model=_MINI_MODEL)


@pytest.fixture(scope="module")
def embedder():
    from cogbase.embeddings.openai import OpenAIEmbedding

    client = openai.AsyncOpenAI(api_key=_openai_api_key)
    return OpenAIEmbedding(client)


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
        """Two-turn conversation starting from no history.

        Turn 1 asks the model to propose fields for a contract app; turn 2
        confirms and triggers schema + config generation. Verifies the final
        config is structurally valid with doc_id injected.
        """
        turn1_text = (
            "I want to build a contract analysis app. "
            "Users upload PDF contracts and ask about vendor names, payment terms, "
            "and expiry dates. What structured fields should I extract?"
        )
        body1 = GenerateChatRequest(text=turn1_text, history=[])
        response1 = await chat(body1, MagicMock(llm=llm))
        assert response1.content, "expected a text proposal in turn 1"

        if response1.config_yaml:
            config = AppConfig.from_yaml(response1.config_yaml)
            assert config.name
            assert config.pipelines
            return

        history = [
            ChatMessage(role="user", content=turn1_text),
            ChatMessage(role="assistant", content=response1.content),
        ]
        body2 = GenerateChatRequest(
            text=(
                "Those fields look exactly right. "
                "Please generate the extraction schema and the full app config now."
            ),
            history=history,
        )
        response2 = await chat(body2, MagicMock(llm=llm))

        assert response2.config_yaml, (
            "expected config_yaml after confirming fields.\n"
            f"turn 1 response: {response1.content!r}\n"
            f"turn 2 response: {response2.content!r}"
        )
        config = AppConfig.from_yaml(response2.config_yaml)
        assert config.name
        assert config.pipelines

        first_step = config.pipelines[0].steps[0]
        assert getattr(first_step, "tool", None) == "chunk-embed-upsert", (
            f"first pipeline step must be chunk-embed-upsert, "
            f"got {getattr(first_step, 'tool', None)!r}"
        )

        data = yaml.safe_load(response2.config_yaml)
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
        _MAX_ROUNDS = 5  # initial turn + up to 4 confirmations

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
        from cogbase.core.query_runner import QueryResult
        from cogbase.stores.structured.memory import InMemoryStructuredStore
        from cogbase.stores.vector.faiss_store import FAISSVectorStore
        from examples.contract_compliance_demo.contracts_data import CONTRACTS_DOCUMENTS
        from examples.contract_compliance_demo.rules_data import RULES_DOCUMENTS

        _MAX_ROUNDS = 4

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
        app = await build_app(config, system=system, app_status="new")

        results = await app.ingest_documents(
            RULES_DOCUMENTS[:2] + CONTRACTS_DOCUMENTS[:1], concurrency=3
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

        def _status_value(finding: dict) -> str:
            for k, v in finding.items():
                if not isinstance(v, str):
                    continue
                kl = k.lower()
                if any(h in kl for h in (
                    "summary", "reason", "explanation",
                    "rationale", "description", "note",
                )):
                    continue
                if any(h in kl for h in ("status", "compliance", "compliant", "verdict")):
                    return v.lower()
            return ""

        non_compliant = [f for f in findings if "non" in _status_value(f)]
        assert non_compliant, (
            "expected at least one non-compliant finding for contract-001 — "
            "the liability cap (3 months) should violate the 12-month policy rule. "
            f"findings: {findings}"
        )

        compliant = [
            f for f in findings
            if (val := _status_value(f)) and "compliant" in val and "non" not in val
        ]
        assert compliant, f"expected at least one compliant finding. findings: {findings}"

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
    """Contract analyst demo: single-pipeline app, no workflow.

    Chat generates the config (one pipeline that chunks, embeds, and extracts
    key contract facts). The test ingests the 5 SaaS contracts and verifies
    queries that exercise cross-contract comparison and structured lookup.
    """

    async def test_ingest_and_query(self, llm, embedder):
        from api.factory import build_app
        from api.system_resources import SystemResources
        from cogbase.core.models import Document
        from cogbase.core.query_runner import QueryResult
        from cogbase.stores.structured.memory import InMemoryStructuredStore
        from cogbase.stores.vector.faiss_store import FAISSVectorStore
        from examples.contract_analyst_demo.saas_contracts import CONTRACTS

        _MAX_ROUNDS = 4

        text = (
            "I need a contract analysis app for SaaS vendor agreements. "
            "Users upload contracts and want to ask about vendors, expiry dates, "
            "liability caps, payment terms, and governing law. "
            "Extract the following fields from each contract: vendor name, "
            "customer name, governing law jurisdiction, effective date, expiry date, "
            "total contract value, liability cap amount, and termination notice "
            "period in days."
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

        system = SystemResources(
            structured_store=InMemoryStructuredStore(),
            vector_store=FAISSVectorStore(),
            llm=llm,
            embedder=embedder,
        )
        app = await build_app(config, system=system, app_status="new")

        documents = [
            Document(doc_id=doc_id, text=text) for doc_id, text in CONTRACTS.items()
        ]
        results = await app.ingest_documents(documents, concurrency=3)
        failed = [r for r in results if not r.success]
        assert not failed, (
            f"{len(failed)} document(s) failed ingestion: "
            + ", ".join(f"{r.doc_id}: {r.error}" for r in failed)
        )
        assert all(r.records_extracted > 0 for r in results), (
            "expected each contract to produce at least one extracted record"
        )

        async def _query(text: str) -> str:
            result = None
            async for chunk in app.query_stream(text):
                if isinstance(chunk, QueryResult):
                    result = chunk
            assert result is not None, f"query_stream produced no QueryResult for: {text!r}"
            return result.answer

        answer1 = (await _query("which contracts expire before 2026-01-01?")).lower()
        expiring_hits = sum(
            any(kw in answer1 for kw in ids)
            for ids in [
                ("saas-001", "cloudstore", "acme"),
                ("saas-003", "securevault", "nexus"),
                ("saas-005", "workflowmanager", "apex"),
            ]
        )
        assert expiring_hits >= 2, (
            f"expected ≥2 of the 3 contracts expiring before 2026 to be named:\n{answer1}"
        )

        answer2 = (await _query(
            "which contracts have a liability cap above 1 million dollars?"
        )).lower()
        assert any(
            kw in answer2
            for kw in ("saas-003", "securevault", "nexus", "2,000,000", "2000000", "2 million")
        ), f"expected saas-003 / Nexus / $2M cap in answer:\n{answer2}"

        answer3 = (await _query(
            "which contract has the longest termination notice period?"
        )).lower()
        assert any(
            kw in answer3
            for kw in ("180", "saas-005", "workflowmanager", "apex")
        ), f"expected 180-day notice / saas-005 / Apex in answer:\n{answer3}"
