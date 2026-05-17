"""Live integration tests for api/routers/generate.py against the real OpenAI API.

These tests build a real ``OpenAILLM`` backed by ``openai.AsyncOpenAI`` and the
``OPENAI_API_KEY`` from the repo-root ``.env``. The whole module is skipped when
the key is not set, mirroring ``tests/embeddings/test_openai_embeddings.py``.

Costs/latency: each test issues real OpenAI requests. The end-to-end chat test
may run the full agent loop with multiple tool calls.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from api.models import ChatMessage, GenerateChatRequest
from api.routers.generate import (
    _run_propose_config,
    _run_propose_extraction_schemas,
    _run_propose_workflow_schemas,
    chat,
)
from cogbase.config.config import AppConfig

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
