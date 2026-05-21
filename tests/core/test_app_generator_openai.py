"""Live integration tests for cogbase/core/app_generator.py against the real OpenAI API.

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

import pytest
import yaml

from cogbase.core.app_generator import (
    _make_record_schema,
    propose_app_config,
    _run_propose_extraction_schemas,
    _run_propose_pipeline_config,
    _run_propose_workflow_config,
    _run_propose_workflow_schemas,
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
        clause_extraction_schema = {
            "type": "object",
            "properties": {
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
        record_schemas = {
            "contract_clauses": json.dumps(
                _make_record_schema(clause_extraction_schema, id_field="clause_id")
            )
        }
        message, schemas = await _run_propose_workflow_schemas(
            llm, _CONTRACT_COMPLIANCE_CONVERSATION, record_schemas
        )
        assert schemas is not None, message
        assert "Workflow schemas validated." in message
        assert schemas, "expected at least one workflow output collection"
        for name, schema_json in schemas.items():
            schema = json.loads(schema_json)
            props = schema.get("properties", {})
            assert props, f"{name} must have properties"


# ---------------------------------------------------------------------------
# Config generation — validates the YAML actually parses as AppConfig and
# satisfies the cross-field invariants.
# ---------------------------------------------------------------------------


class TestProposePipelineConfigLive:
    async def test_generates_valid_pipeline_config_for_contract_app(self, llm):
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
        message, config_dict, record_schemas, stored_yaml = await _run_propose_pipeline_config(
            llm, _CONTRACT_CONVERSATION, extraction_schemas
        )
        assert config_dict is not None, message
        assert message.startswith("Pipeline config validated.")

        assert stored_yaml is not None
        config = AppConfig.from_yaml(stored_yaml)
        assert config.name
        assert config.pipelines, "expected at least one pipeline"

        first_step = config.pipelines[0].steps[0]
        assert getattr(first_step, "tool", None) == "chunk-embed-upsert"

        assert "contracts" in record_schemas or any("contract" in k for k in record_schemas)
        for coll_name, schema_json in record_schemas.items():
            schema = json.loads(schema_json)
            assert "doc_id" in schema.get("properties", {}), (
                f"record_schemas[{coll_name!r}] must include injected doc_id"
            )

        data = yaml.safe_load(stored_yaml)
        for sc in data.get("structured_collections", []):
            record_schema = json.loads(sc["schema"])
            assert "doc_id" in record_schema["properties"]
            assert record_schema["required"][0] == "doc_id"


class TestProposeWorkflowConfigLive:
    async def test_generates_valid_workflow_config(self, llm):
        clause_extraction_schema = {
            "type": "object",
            "properties": {
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
        extraction_schemas = {
            "contract_clauses": json.dumps(clause_extraction_schema)
        }
        p_message, pipeline_config_dict, record_schemas, _ = await _run_propose_pipeline_config(
            llm, _CONTRACT_COMPLIANCE_CONVERSATION, extraction_schemas
        )
        assert pipeline_config_dict is not None, p_message

        assert record_schemas, "expected at least one record schema from the pipeline step"
        for coll_name, schema_json in record_schemas.items():
            schema = json.loads(schema_json)
            assert "doc_id" in schema.get("properties", {}), (
                f"record_schemas[{coll_name!r}] must include doc_id"
            )

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

        message, stored_yaml = await _run_propose_workflow_config(
            llm,
            _CONTRACT_COMPLIANCE_CONVERSATION,
            pipeline_config_dict,
            record_schemas,
            workflow_schemas,
        )
        assert stored_yaml is not None, message
        assert message == "Config validated."

        config = AppConfig.from_yaml(stored_yaml)
        assert config.workflows, "expected at least one workflow"

        data = yaml.safe_load(stored_yaml)
        findings_sc = next(
            (sc for sc in data.get("structured_collections", [])
             if sc["name"] == "clause_compliance_findings"),
            None,
        )
        assert findings_sc is not None, "clause_compliance_findings must be in structured_collections"
        injected = json.loads(findings_sc["schema"])
        expected = json.loads(workflow_schemas["clause_compliance_findings"])
        assert injected == expected


# ---------------------------------------------------------------------------
# Orchestrator: verify needs_workflow branching in propose_app_config.
# ---------------------------------------------------------------------------


class TestProposeAppConfigLive:
    async def test_no_workflow_skips_workflow_steps(self, llm):
        events = []
        async for event in propose_app_config(llm, _CONTRACT_CONVERSATION, needs_workflow=False):
            events.append(event)

        result_events = [e for e in events if e["type"] == "result"]
        assert len(result_events) == 1
        result = result_events[0]
        assert result["generation_context"] == "Config generation complete.", result["generation_context"]
        assert result["config_yaml"] is not None

        progress = " ".join(e["token"] for e in events if e["type"] == "token").lower()
        assert "extraction" in progress
        assert "pipeline" in progress
        assert "workflow" not in progress

        config = AppConfig.from_yaml(result["config_yaml"])
        assert config.name
        assert config.pipelines
        assert not config.workflows

    async def test_with_workflow_runs_all_steps(self, llm):
        events = []
        async for event in propose_app_config(llm, _CONTRACT_COMPLIANCE_CONVERSATION, needs_workflow=True):
            events.append(event)

        result_events = [e for e in events if e["type"] == "result"]
        assert len(result_events) == 1
        result = result_events[0]
        assert result["generation_context"] == "Config generation complete.", result["generation_context"]
        assert result["config_yaml"] is not None

        progress = " ".join(e["token"] for e in events if e["type"] == "token").lower()
        assert "extraction" in progress
        assert "pipeline" in progress
        assert "workflow" in progress

        config = AppConfig.from_yaml(result["config_yaml"])
        assert config.name
        assert config.pipelines
        assert config.workflows
