"""Live integration tests for api/routers/generate.py against the real OpenAI API.

These tests build a real ``OpenAILLM`` backed by ``openai.AsyncOpenAI`` and the
``OPENAI_API_KEY`` from the repo-root ``.env``. The whole module is skipped when
the key is not set, mirroring ``tests/embeddings/test_openai_embeddings.py``.

Costs/latency: each test issues real OpenAI requests. The end-to-end chat test
may run the full agent loop with multiple tool calls.
"""

from __future__ import annotations

import logging
import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from api.models import ChatMessage, GenerateChatRequest
from api.routers.generate import (
    _collect_save_targets,
    _run_propose_config,
    _run_propose_extraction_schemas,
    _run_propose_workflow_schemas,
    chat,
)
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


@pytest.fixture(scope="module")
def llm():
    from cogbase.llms.openai import OpenAILLM

    client = openai.AsyncOpenAI(api_key=_openai_api_key)
    return OpenAILLM(client, model=_MODEL)


@pytest.fixture(scope="module")
def embedder():
    from cogbase.embeddings.openai import OpenAIEmbedding

    client = openai.AsyncOpenAI(api_key=_openai_api_key)
    return OpenAIEmbedding(client)


# Conversations are kept compact so the model has a clear, confirmed brief
# before any tool call — this keeps the agent loop short and deterministic.
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


_CONTRACT_COMPLIANCE_CONVERSATION: list[dict] = [
    {
        "role": "user",
        "content": (
            "Build a contract compliance app. Users upload contracts; the "
            "system extracts each clause, then a workflow checks every clause "
            "against company policy and saves a finding per clause."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Plan:\n"
            "Pipeline structured collection 'contract_clauses' with fields:\n"
            "- clause_id — stable id for the clause\n"
            "- clause_type — e.g. indemnity, termination, payment\n"
            "- text — verbatim clause text\n"
            "\n"
            "Workflow 'check-compliance' fans out over contract_clauses, "
            "applies LLM judgment, and saves into structured collection "
            "'clause_compliance_findings' with fields:\n"
            "- clause_id — id of the reviewed clause\n"
            "- doc_id — source contract id\n"
            "- status — compliant | non_compliant | unclear\n"
            "- severity — low | medium | high (null when compliant)\n"
            "- summary — plain-language finding\n"
            "Confirm?"
        ),
    },
    {
        "role": "user",
        "content": "Confirmed. Proceed.",
    },
]


# ---------------------------------------------------------------------------
# Direct sub-agent tests (no chat loop) — exercise each tool handler in
# isolation so failures point at the right component.
# ---------------------------------------------------------------------------


class TestProposeExtractionSchemasLive:
    async def test_returns_validated_schemas_for_contract_app(self, llm):
        message, schemas = await _run_propose_extraction_schemas(
            llm, _CONTRACT_CONVERSATION
        )
        assert schemas is not None, message
        assert message.startswith("Schemas validated.")
        assert "contracts" in schemas or any("contract" in k for k in schemas)

        # Each returned schema parses as a valid JSON Schema object with at
        # least one property and no injected doc_id.
        for collection_name, schema_json in schemas.items():
            schema = json.loads(schema_json)
            assert schema.get("type") == "object"
            props = schema.get("properties", {})
            assert props, f"{collection_name} must have properties"
            assert "doc_id" not in props, (
                f"{collection_name} must not include doc_id — it is injected"
            )


class TestProposeWorkflowSchemasLive:
    async def test_returns_validated_workflow_schemas(self, llm):
        extraction_schemas = {
            "contract_clauses": json.dumps(
                {
                    "type": "object",
                    "properties": {
                        "clause_id": {
                            "type": "string",
                            "description": "stable id for the clause",
                        },
                        "clause_type": {
                            "type": "string",
                            "description": "e.g. indemnity, termination, payment",
                        },
                        "text": {
                            "type": "string",
                            "description": "verbatim clause text",
                        },
                    },
                }
            )
        }
        message, schemas = await _run_propose_workflow_schemas(
            llm, _CONTRACT_COMPLIANCE_CONVERSATION, extraction_schemas
        )
        assert schemas is not None, message
        assert "Workflow schemas validated." in message
        # Expect at least one workflow output collection that traces back to
        # the source clause/document.
        assert schemas, "expected at least one workflow output collection"
        for name, schema_json in schemas.items():
            schema = json.loads(schema_json)
            props = schema.get("properties", {})
            assert props, f"{name} must have properties"


# ---------------------------------------------------------------------------
# Config generation — validates the YAML actually parses as AppConfig and
# satisfies the cross-field invariants.
# ---------------------------------------------------------------------------


class TestProposeAppConfigLive:
    async def test_generates_valid_app_config_for_contract_app(self, llm):
        extraction_schemas = {
            "contracts": json.dumps(
                {
                    "type": "object",
                    "properties": {
                        "vendor_name": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                            "description": "Name of the vendor",
                        },
                        "effective_date": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                            "description": "Contract start date (ISO 8601)",
                        },
                        "expiry_date": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                            "description": "Contract end date (ISO 8601)",
                        },
                        "total_value": {
                            "anyOf": [{"type": "number"}, {"type": "null"}],
                            "description": "Total contract value in USD",
                        },
                        "governing_law": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                            "description": "Jurisdiction governing the contract",
                        },
                    },
                }
            )
        }
        message, stored_yaml = await _run_propose_config(
            llm, _CONTRACT_CONVERSATION, extraction_schemas
        )
        assert stored_yaml is not None, message
        assert message == "Config validated."

        config = AppConfig.from_yaml(stored_yaml)
        assert config.name  # kebab-case name picked by the model
        assert config.pipelines, "expected at least one pipeline"

        # The pipeline must include a chunk-embed-upsert step first.
        first_pipeline = config.pipelines[0]
        first_step = first_pipeline.steps[0]
        assert getattr(first_step, "tool", None) == "chunk-embed-upsert"

        # The structured 'contracts' collection schema is injected with doc_id
        # (the model author's extraction_schema goes verbatim into the step).
        data = yaml.safe_load(stored_yaml)
        contracts_sc = next(
            sc
            for sc in data.get("structured_collections", [])
            if sc["name"] == "contracts"
        )
        record_schema = json.loads(contracts_sc["schema"])
        assert "doc_id" in record_schema["properties"]
        assert record_schema["required"][0] == "doc_id"
        assert "vendor_name" in record_schema["properties"]

    async def test_generates_config_with_workflow(self, llm):
        extraction_schemas = {
            "contract_clauses": json.dumps(
                {
                    "type": "object",
                    "properties": {
                        "clause_id": {
                            "type": "string",
                            "description": "stable id for the clause",
                        },
                        "clause_type": {
                            "type": "string",
                            "description": "e.g. indemnity, termination, payment",
                        },
                        "text": {
                            "type": "string",
                            "description": "verbatim clause text",
                        },
                    },
                }
            )
        }
        workflow_schemas = {
            "clause_compliance_findings": json.dumps(
                {
                    "type": "object",
                    "properties": {
                        "clause_id": {
                            "type": "string",
                            "description": "id of the reviewed clause",
                        },
                        "doc_id": {
                            "type": "string",
                            "description": "source contract id",
                        },
                        "status": {
                            "type": "string",
                            "description": "compliant | non_compliant | unclear",
                        },
                        "severity": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                            "description": "low | medium | high (null when compliant)",
                        },
                        "summary": {
                            "type": "string",
                            "description": "plain-language finding",
                        },
                    },
                }
            )
        }
        message, stored_yaml = await _run_propose_config(
            llm,
            _CONTRACT_COMPLIANCE_CONVERSATION,
            extraction_schemas,
            workflow_schemas,
        )
        assert stored_yaml is not None, message
        assert message == "Config validated."

        config = AppConfig.from_yaml(stored_yaml)
        assert config.workflows, "expected at least one workflow"

        data = yaml.safe_load(stored_yaml)
        findings_sc = next(
            sc
            for sc in data.get("structured_collections", [])
            if sc["name"] == "clause_compliance_findings"
        )
        # The workflow output collection schema is injected verbatim — doc_id
        # is part of the workflow schema (provenance), not auto-added.
        injected = json.loads(findings_sc["schema"])
        expected = json.loads(workflow_schemas["clause_compliance_findings"])
        assert injected == expected


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
        # No tool calls should fire on an informational question.
        assert response.config_yaml is None

    async def test_chat_generates_validated_config_yaml(self, llm):
        history = [ChatMessage(**m) for m in _CONTRACT_CONVERSATION]
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

        # Model may generate a config immediately if it's confident — accept it.
        if response1.config_yaml:
            config = AppConfig.from_yaml(response1.config_yaml)
            assert config.name
            assert config.pipelines
            return

        # Turn 2: confirm the proposal and explicitly request generation.
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

        # Verify doc_id was injected into every extract-structured target schema.
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
        """Two-turn conversation for an app that requires a workflow.

        Turn 1 describes a clause-level compliance app with an explicit workflow
        requirement. Turn 2 confirms the design and triggers full generation.
        Verifies the final config contains at least one workflow and that all
        structured-save target collections have schemas set.
        """
        turn1_text = (
            "Build a contract compliance app. "
            "The pipeline should extract each clause from uploaded contracts. "
            "Also iterates over contract_clauses, runs LLM judgment on each clause, "
            "and saves a compliance finding per clause."
        )
        body1 = GenerateChatRequest(text=turn1_text, history=[])
        response1 = await chat(body1, MagicMock(llm=llm))
        assert response1.content, "expected a text proposal in turn 1"

        logger.info("response1 content=%s", response1.content)
        logger.info("response1 config_yaml=%s", response1.config_yaml)

        final_response = response1
        history = [
            ChatMessage(role="user", content=turn1_text),
            ChatMessage(role="assistant", content=response1.content),
        ]

        if not response1.config_yaml:
            body2 = GenerateChatRequest(
                text=(
                    "Yes, that design is confirmed — all fields and the workflow look right. "
                    "Generate the extraction schemas, workflow schemas, and full config now."
                ),
                history=history,
            )
            final_response = await chat(body2, MagicMock(llm=llm))
            logger.info("final_response content=%s", final_response.content)
            logger.info("final_response config_yaml=%s", final_response.config_yaml)

        assert final_response.config_yaml, (
            "expected config_yaml for the compliance workflow app.\n"
            f"turn 1: {response1.content!r}\n"
            + (
                f"turn 2: {final_response.content!r}"
                if final_response is not response1
                else ""
            )
        )
        config = AppConfig.from_yaml(final_response.config_yaml)
        assert config.name
        assert config.pipelines
        assert config.workflows, "expected at least one workflow in the config"

        # Every structured-save target must exist in structured_collections with a schema.
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

        # ---- Step 1: chat to generate the app config ------------------------
        turn1_text = (
            "Build a contract compliance app with two document types: "
            "Policy rule documents and Vendor contracts. Check whether "
            "the clauses in a contract is compliant with policy."
        )
        body1 = GenerateChatRequest(text=turn1_text, history=[])
        response1 = await chat(body1, MagicMock(llm=llm))
        assert response1.content, "expected a proposal in turn 1"

        logger.info("response1 content=%s", response1.content)
        logger.info("response1 config_yaml=%s", response1.config_yaml)

        final_response = response1
        history = [
            ChatMessage(role="user", content=turn1_text),
            ChatMessage(role="assistant", content=response1.content),
        ]
        if not response1.config_yaml:
            body2 = GenerateChatRequest(
                text=(
                    "Yes, that design is confirmed. "
                    "Generate the extraction schemas, workflow schemas, and full config now."
                ),
                history=history,
            )
            final_response = await chat(body2, MagicMock(llm=llm))
            logger.info("final_response content=%s", final_response.content)
            logger.info("final_response config_yaml=%s", final_response.config_yaml)

        assert final_response.config_yaml, (
            "expected config_yaml from the chat agent.\n"
            f"turn 1: {response1.content!r}\n"
            + (f"turn 2: {final_response.content!r}" if final_response is not response1 else "")
        )

        # ---- Step 2: parse config, discover workflow details ----------------
        config = AppConfig.from_yaml(final_response.config_yaml)
        assert config.workflows, "generated config must have at least one workflow"

        # Discover the structured-save target collection(s) from the workflow steps.
        data = yaml.safe_load(final_response.config_yaml)
        save_targets: set[str] = set()
        for wf in data.get("workflows", []):
            _collect_save_targets(wf.get("steps", []), save_targets)
        assert save_targets, "workflow must contain at least one structured-save step"

        # The first workflow is the compliance workflow; discover its name and
        # the input key the LLM chose (should be doc_id, but inspect to be safe).
        workflow_cfg = config.workflows[0]
        workflow_input_key = "doc_id"

        # ---- Step 3: build the app in-process with in-memory stores ---------
        system = SystemResources(
            structured_store=InMemoryStructuredStore(),
            vector_store=FAISSVectorStore(),
            llm=llm,
            embedder=embedder,
        )
        app = await build_app(config, system=system, app_status="new")

        # ---- Step 4: ingest all demo documents ------------------------------
        results = await app.ingest_documents(
            RULES_DOCUMENTS + CONTRACTS_DOCUMENTS, concurrency=3
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

        # ---- Step 5: run the compliance workflow for contract-001 -----------
        findings: list[dict] = []
        workflow = app.get_workflow(workflow_cfg.name)
        async for record in workflow.run({workflow_input_key: "contract-001"}):
            findings.append(record)

        if not findings:
            # Dump structured store contents and config to aid debugging.
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

        # contract-001 has known non-compliant clauses (liability cap 3 months,
        # one-sided consequential exclusion, 48-hour breach notification).
        non_compliant = [
            f for f in findings
            if "non" in str(f.get("status", "")).lower()
            or "non_compliant" in str(f.get("status", "")).lower()
        ]
        assert non_compliant, (
            "expected at least one non-compliant finding for contract-001 — "
            "the liability cap (3 months) should violate the 12-month policy rule. "
            f"findings: {findings}"
        )

        # contract-001 also has compliant clauses (payment net-30, mutual indemnification).
        compliant = [
            f for f in findings
            if f.get("status") in ("compliant", "not_applicable")
            or str(f.get("status", "")).lower() == "compliant"
        ]
        assert compliant, "expected at least one compliant or not_applicable finding"

        # ---- Step 6: natural-language queries over the live data -------------
        async def _query(text: str) -> str:
            result = None
            async for chunk in app.query_stream(text):
                if isinstance(chunk, QueryResult):
                    result = chunk
            assert result is not None, f"query_stream produced no QueryResult for: {text!r}"
            return result.answer

        # Non-compliant findings query — must surface at least one violation.
        answer1 = (await _query(
            "what are the non-compliant clauses in contract-001?"
        )).lower()
        assert any(
            kw in answer1
            for kw in ("non-compliant", "non_compliant", "violat", "liability", "breach", "finding")
        ), f"expected a non-compliance reference in the answer:\n{answer1}"

        # Governing-law query — contract-001 is governed by New York law.
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

        # ---- Step 1: chat to generate the app config ------------------------
        turn1_text = (
            "I need a contract analysis app for SaaS vendor agreements. "
            "Users upload contracts and want to ask about vendors, expiry dates, "
            "liability caps, payment terms, and governing law. "
            "Extract the following fields from each contract: vendor name, "
            "customer name, governing law jurisdiction, effective date, expiry date, "
            "total contract value, liability cap amount, and termination notice "
            "period in days."
        )
        body1 = GenerateChatRequest(text=turn1_text, history=[])
        response1 = await chat(body1, MagicMock(llm=llm))
        assert response1.content, "expected a field proposal in turn 1"

        final_response = response1
        history = [
            ChatMessage(role="user", content=turn1_text),
            ChatMessage(role="assistant", content=response1.content),
        ]
        if not response1.config_yaml:
            body2 = GenerateChatRequest(
                text=(
                    "Those fields look right. "
                    "Generate the extraction schema and full config now."
                ),
                history=history,
            )
            final_response = await chat(body2, MagicMock(llm=llm))

        assert final_response.config_yaml, (
            "expected config_yaml from the chat agent.\n"
            f"turn 1: {response1.content!r}\n"
            + (f"turn 2: {final_response.content!r}" if final_response is not response1 else "")
        )

        # ---- Step 2: parse config -------------------------------------------
        config = AppConfig.from_yaml(final_response.config_yaml)
        assert config.pipelines, "generated config must have at least one pipeline"

        # ---- Step 3: build the app in-process with in-memory stores ---------
        system = SystemResources(
            structured_store=InMemoryStructuredStore(),
            vector_store=FAISSVectorStore(),
            llm=llm,
            embedder=embedder,
        )
        app = await build_app(config, system=system, app_status="new")

        # ---- Step 4: ingest all 5 SaaS contracts ----------------------------
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

        # ---- Step 5: queries over extracted records and full text -----------
        async def _query(text: str) -> str:
            result = None
            async for chunk in app.query_stream(text):
                if isinstance(chunk, QueryResult):
                    result = chunk
            assert result is not None, f"query_stream produced no QueryResult for: {text!r}"
            return result.answer

        # Three contracts expire before 2026-01-01:
        #   saas-001 (Acme / CloudStore Pro, Jun 2025)
        #   saas-003 (Nexus / SecureVault, Dec 2025)
        #   saas-005 (Apex / WorkflowManager, Sep 2025)
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

        # Only saas-003 has a liability cap above $1M (USD 2,000,000, Nexus Security).
        answer2 = (await _query(
            "which contracts have a liability cap above 1 million dollars?"
        )).lower()
        assert any(
            kw in answer2
            for kw in ("saas-003", "securevault", "nexus", "2,000,000", "2000000", "2 million")
        ), f"expected saas-003 / Nexus / $2M cap in answer:\n{answer2}"

        # saas-005 (Apex / WorkflowManager) has an unusually long 180-day notice period.
        answer3 = (await _query(
            "which contract has the longest termination notice period?"
        )).lower()
        assert any(
            kw in answer3
            for kw in ("180", "saas-005", "workflowmanager", "apex")
        ), f"expected 180-day notice / saas-005 / Apex in answer:\n{answer3}"
