"""Integration tests for the /applications REST endpoints.

All tests use httpx.AsyncClient pointed at the FastAPI app with dependency
overrides injected — no real LLM calls or file I/O happens.
"""

from __future__ import annotations

import asyncio
import io
import json
import textwrap
import zipfile
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import yaml

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from api.dependencies import (
    get_app_cache,
    get_skill_registry,
    get_system_resources,
    get_system_store,
)
from api.system_resources import SystemResources
from cogbase.skills.registry import SkillRegistry
from api.main import app
from api.app_cache import AppCache

from api.system_store import DocRecord, SystemStore
from cogbase.config.config import AppConfig, RecordMode
from cogbase.core.models import Chunk
from cogbase.core.query_runner import DocumentSlice, QueryResult
from cogbase.stores.structured.memory import InMemoryStructuredStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bundle(config_yaml: bytes, files: dict[str, bytes] | None = None) -> bytes:
    """Build an in-memory ZIP bundle from a config YAML and optional extra files."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("config.yaml", config_yaml)
        for name, content in (files or {}).items():
            zf.writestr(name, content)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Fixtures — lightweight dependency overrides
# ---------------------------------------------------------------------------

def _make_system_store() -> SystemStore:
    backend = InMemoryStructuredStore()
    return SystemStore(store=backend)


def _make_app_cache() -> AppCache:
    return AppCache()


_VALID_CONFIG_YAML = textwrap.dedent("""\
    name: my-contract-analyzer
    llm:
      provider: openai
      model: gpt-4o-mini
      api_key: sk-test
""").encode()

_VALID_BUNDLE = _make_bundle(_VALID_CONFIG_YAML)


def _mock_app_instance() -> MagicMock:
    """Minimal mock that satisfies the build_app / app lifecycle contract."""
    return MagicMock()


@pytest_asyncio.fixture
async def client():
    """AsyncClient with all external dependencies swapped out."""
    system_store = _make_system_store()
    await system_store.setup()
    app_cache = _make_app_cache()
    system_resources = SystemResources(structured_store=InMemoryStructuredStore())

    app.dependency_overrides[get_system_store] = lambda: system_store
    app.dependency_overrides[get_app_cache] = lambda: app_cache
    app.dependency_overrides[get_system_resources] = lambda: system_resources
    app.dependency_overrides[get_skill_registry] = lambda: SkillRegistry()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def app_overrides():
    """Like ``client`` but also exposes the underlying SystemStore for seeding test data."""
    system_store = _make_system_store()
    await system_store.setup()
    app_cache = _make_app_cache()
    system_resources = SystemResources(structured_store=InMemoryStructuredStore())

    app.dependency_overrides[get_system_store] = lambda: system_store
    app.dependency_overrides[get_app_cache] = lambda: app_cache
    app.dependency_overrides[get_system_resources] = lambda: system_resources
    app.dependency_overrides[get_skill_registry] = lambda: SkillRegistry()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield {"client": ac, "system_store": system_store}

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# _serialize_config — enum serialization
# ---------------------------------------------------------------------------

class TestSerializeConfig:
    def _minimal_config(self, **overrides) -> AppConfig:
        return AppConfig(
            name="test-app",
            llm={"provider": "openai", "model": "gpt-4o-mini"},
            **overrides,
        )

    def test_enum_serializes_as_plain_string(self):
        """RecordMode enum must appear as its string value in YAML, not as a Python object tag."""
        config_yaml = textwrap.dedent("""\
            name: test-app
            llm:
              provider: openai
              model: gpt-4o-mini
              api_key: sk-test
            structured_collections:
              - name: facts
                description: Extracted facts.
                schema: '{}'
                primary_fields: [doc_id]
            pipelines:
              - name: main
                routing_description: Documents to extract facts from.
                steps:
                  - tool: extract-structured
                    collection: facts
                    extractor:
                      type: llm
                      extraction_schema: '{}'
                      prompt: Extract facts.
                      record_mode: many
                      id_field: fact_id
                      id_template: "{doc_id}__{index:04d}"
        """)
        cfg = AppConfig.model_validate(yaml.safe_load(config_yaml))
        serialized = cfg.to_yaml()

        assert "!!python" not in serialized, "YAML must not contain Python-specific tags"
        assert "record_mode: many" in serialized

    def test_serialized_yaml_round_trips(self):
        """YAML produced by _serialize_config can be parsed back with yaml.safe_load and validates to an equivalent AppConfig."""
        config_yaml = textwrap.dedent("""\
            name: test-app
            llm:
              provider: openai
              model: gpt-4o-mini
              api_key: sk-test
            structured_collections:
              - name: facts
                description: Extracted facts.
                schema: '{}'
                primary_fields: [doc_id]
            pipelines:
              - name: main
                routing_description: Documents to extract facts from.
                steps:
                  - tool: extract-structured
                    collection: facts
                    extractor:
                      type: llm
                      extraction_schema: '{}'
                      prompt: Extract facts.
                      record_mode: one
        """)
        original = AppConfig.model_validate(yaml.safe_load(config_yaml))
        serialized = original.to_yaml()

        # safe_load must succeed (no Python-specific tags that require unsafe load)
        parsed_back = yaml.safe_load(serialized)
        restored = AppConfig.model_validate(parsed_back)

        assert restored.name == original.name
        step = restored.pipelines[0].steps[0]
        assert step.extractor.record_mode == RecordMode.ONE


# ---------------------------------------------------------------------------
# POST /applications
# ---------------------------------------------------------------------------

class TestCreateApplication:
    @pytest.mark.asyncio
    async def test_create_returns_201(self, client):
        with patch("api.routers.applications.build_app", new_callable=AsyncMock, return_value=_mock_app_instance()):
            resp = await client.post(
                "/applications",
                files={"bundle": ("bundle.zip", _VALID_BUNDLE, "application/zip")},
            )
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "my-contract-analyzer"
        assert data["status"] == "active"
        assert data["error"] is None

    @pytest.mark.asyncio
    async def test_create_stores_config_in_response(self, client):
        with patch("api.routers.applications.build_app", new_callable=AsyncMock, return_value=_mock_app_instance()):
            resp = await client.post(
                "/applications",
                files={"bundle": ("bundle.zip", _VALID_BUNDLE, "application/zip")},
            )
        assert resp.status_code == 201
        config = resp.json()["config"]
        assert config["name"] == "my-contract-analyzer"

    @pytest.mark.asyncio
    async def test_create_conflict_returns_409(self, client):
        with patch("api.routers.applications.build_app", new_callable=AsyncMock, return_value=_mock_app_instance()):
            await client.post(
                "/applications",
                files={"bundle": ("bundle.zip", _VALID_BUNDLE, "application/zip")},
            )
            resp = await client.post(
                "/applications",
                files={"bundle": ("bundle.zip", _VALID_BUNDLE, "application/zip")},
            )
        assert resp.status_code == 409
        assert "already exists" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_create_not_a_zip_returns_422(self, client):
        resp = await client.post(
            "/applications",
            files={"bundle": ("bundle.zip", b"not a zip file", "application/zip")},
        )
        assert resp.status_code == 422
        assert "ZIP" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_create_zip_missing_config_yaml_returns_422(self, client):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("other.txt", "hello")
        resp = await client.post(
            "/applications",
            files={"bundle": ("bundle.zip", buf.getvalue(), "application/zip")},
        )
        assert resp.status_code == 422
        assert "config.yaml" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_create_invalid_config_yaml_returns_422(self, client):
        bad_bundle = _make_bundle(b"not: valid: yaml: app: config\n")
        resp = await client.post(
            "/applications",
            files={"bundle": ("bundle.zip", bad_bundle, "application/zip")},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_create_records_error_status_when_build_fails(self, client):
        with patch("api.routers.applications.build_app", new_callable=AsyncMock, side_effect=RuntimeError("setup boom")):
            resp = await client.post(
                "/applications",
                files={"bundle": ("bundle.zip", _VALID_BUNDLE, "application/zip")},
            )
        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == "error"
        assert "boom" in data["error"]

    @pytest.mark.asyncio
    async def test_create_non_mapping_yaml_returns_422(self, client):
        bad_bundle = _make_bundle(b"- item1\n- item2\n")
        resp = await client.post(
            "/applications",
            files={"bundle": ("bundle.zip", bad_bundle, "application/zip")},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_file_refs_resolved_from_bundle(self, client):
        """Schema, extraction_schema, and prompt filenames in config.yaml are replaced with file contents."""
        record_schema_json = b'{"type":"object","properties":{"value":{"type":"string"},"doc_id":{"type":"string"}}}'
        extraction_schema_json = b'{"type":"object","properties":{"value":{"type":"string"}}}'
        prompt_txt = b"Extract contract fields."
        config_yaml = textwrap.dedent("""\
            name: my-contract-analyzer
            llm:
              provider: openai
              model: gpt-4o-mini
              api_key: sk-test
            structured_collections:
              - name: contract_extraction
                description: Extracted contract facts and entities for exact lookup.
                schema: record_schema.json
                primary_fields: [doc_id]
            pipelines:
              - name: main
                routing_description: Contract documents to extract facts from.
                steps:
                  - tool: extract-structured
                    collection: contract_extraction
                    extractor:
                      type: llm
                      extraction_schema: extraction_schema.json
                      prompt: extraction_prompt.txt
        """).encode()
        bundle = _make_bundle(
            config_yaml,
            files={
                "record_schema.json": record_schema_json,
                "extraction_schema.json": extraction_schema_json,
                "extraction_prompt.txt": prompt_txt,
            },
        )
        captured: list = []

        async def _capture(config, **kwargs):
            captured.append(config)
            return _mock_app_instance()

        with patch("api.routers.applications.build_app", new_callable=AsyncMock, side_effect=_capture):
            resp = await client.post(
                "/applications",
                files={"bundle": ("bundle.zip", bundle, "application/zip")},
            )

        assert resp.status_code == 201
        cfg = captured[0]
        assert cfg.structured_collections[0].schema_ == record_schema_json.decode()
        assert cfg.pipelines[0].steps[0].extractor.extraction_schema == extraction_schema_json.decode()
        assert cfg.pipelines[0].steps[0].extractor.prompt == prompt_txt.decode()


# ---------------------------------------------------------------------------
# GET /applications
# ---------------------------------------------------------------------------

class TestListApplications:
    @pytest.mark.asyncio
    async def test_list_empty(self, client):
        resp = await client.get("/applications")
        assert resp.status_code == 200
        body = resp.json()
        assert body["applications"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_list_returns_created_apps(self, client):
        bundle_a = _make_bundle(b"name: app-a\nllm:\n  model: gpt-4o-mini\n  api_key: sk-test\n")
        bundle_b = _make_bundle(b"name: app-b\nllm:\n  model: gpt-4o-mini\n  api_key: sk-test\n")
        with patch("api.routers.applications.build_app", new_callable=AsyncMock, return_value=_mock_app_instance()):
            await client.post("/applications", files={"bundle": ("a.zip", bundle_a, "application/zip")})
            await client.post("/applications", files={"bundle": ("b.zip", bundle_b, "application/zip")})
        resp = await client.get("/applications")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 2
        names = {a["name"] for a in body["applications"]}
        assert names == {"app-a", "app-b"}


# ---------------------------------------------------------------------------
# GET /applications/{app_name}
# ---------------------------------------------------------------------------

class TestGetApplication:
    @pytest.mark.asyncio
    async def test_get_existing(self, client):
        with patch("api.routers.applications.build_app", new_callable=AsyncMock, return_value=_mock_app_instance()):
            await client.post(
                "/applications",
                files={"bundle": ("bundle.zip", _VALID_BUNDLE, "application/zip")},
            )
        resp = await client.get("/applications/my-contract-analyzer")
        assert resp.status_code == 200
        assert resp.json()["name"] == "my-contract-analyzer"

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_404(self, client):
        resp = await client.get("/applications/does-not-exist")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# PATCH /applications/{app_name}
# ---------------------------------------------------------------------------

class TestUpdateApplication:
    @pytest.mark.asyncio
    async def test_update_success(self, client):
        with patch("api.routers.applications.build_app", new_callable=AsyncMock, return_value=_mock_app_instance()):
            await client.post(
                "/applications",
                files={"bundle": ("bundle.zip", _VALID_BUNDLE, "application/zip")},
            )

        updated_yaml = _VALID_CONFIG_YAML.replace(b"gpt-4o-mini", b"gpt-4o")
        updated_bundle = _make_bundle(updated_yaml)
        with patch("api.routers.applications.build_app", new_callable=AsyncMock, return_value=_mock_app_instance()):
            resp = await client.patch(
                "/applications/my-contract-analyzer",
                files={"bundle": ("bundle.zip", updated_bundle, "application/zip")},
            )
        assert resp.status_code == 200
        assert resp.json()["status"] == "active"

    @pytest.mark.asyncio
    async def test_update_nonexistent_returns_404(self, client):
        resp = await client.patch(
            "/applications/ghost",
            files={"bundle": ("bundle.zip", _VALID_BUNDLE, "application/zip")},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_update_name_conflict_returns_409(self, client):
        bundle_a = _make_bundle(b"name: app-a\nllm:\n  model: gpt-4o-mini\n  api_key: sk-test\n")
        bundle_b = _make_bundle(b"name: app-b\nllm:\n  model: gpt-4o-mini\n  api_key: sk-test\n")
        with patch("api.routers.applications.build_app", new_callable=AsyncMock, return_value=_mock_app_instance()):
            await client.post("/applications", files={"bundle": ("a.zip", bundle_a, "application/zip")})
            await client.post("/applications", files={"bundle": ("b.zip", bundle_b, "application/zip")})

        with patch("api.routers.applications.build_app", new_callable=AsyncMock, return_value=_mock_app_instance()):
            resp = await client.patch(
                "/applications/app-a",
                files={"bundle": ("bundle.zip", bundle_b, "application/zip")},
            )
        assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_update_records_error_when_setup_fails(self, client):
        with patch("api.routers.applications.build_app", new_callable=AsyncMock, return_value=_mock_app_instance()):
            await client.post(
                "/applications",
                files={"bundle": ("bundle.zip", _VALID_BUNDLE, "application/zip")},
            )

        with patch("api.routers.applications.build_app", new_callable=AsyncMock, side_effect=RuntimeError("update boom")):
            resp = await client.patch(
                "/applications/my-contract-analyzer",
                files={"bundle": ("bundle.zip", _VALID_BUNDLE, "application/zip")},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "error"
        assert "boom" in data["error"]


# ---------------------------------------------------------------------------
# DELETE /applications/{app_name}
# ---------------------------------------------------------------------------

class TestDeleteApplication:
    @pytest.mark.asyncio
    async def test_delete_returns_204(self, client):
        with patch("api.routers.applications.build_app", new_callable=AsyncMock, return_value=_mock_app_instance()):
            await client.post(
                "/applications",
                files={"bundle": ("bundle.zip", _VALID_BUNDLE, "application/zip")},
            )
        resp = await client.delete("/applications/my-contract-analyzer")
        assert resp.status_code == 204

    @pytest.mark.asyncio
    async def test_delete_removes_from_list(self, client):
        with patch("api.routers.applications.build_app", new_callable=AsyncMock, return_value=_mock_app_instance()):
            await client.post(
                "/applications",
                files={"bundle": ("bundle.zip", _VALID_BUNDLE, "application/zip")},
            )
        await client.delete("/applications/my-contract-analyzer")
        resp = await client.get("/applications")
        assert resp.json()["total"] == 0

    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_404(self, client):
        resp = await client.delete("/applications/ghost")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_removes_from_app_cache(self, client):
        app_cache = _make_app_cache()
        app.dependency_overrides[get_app_cache] = lambda: app_cache

        with patch("api.routers.applications.build_app", new_callable=AsyncMock, return_value=_mock_app_instance()):
            await client.post(
                "/applications",
                files={"bundle": ("bundle.zip", _VALID_BUNDLE, "application/zip")},
            )
        assert app_cache.get("my-contract-analyzer") is not None

        await client.delete("/applications/my-contract-analyzer")
        assert app_cache.get("my-contract-analyzer") is None


# ---------------------------------------------------------------------------
# POST /applications/{app_name}/upload_documents
# ---------------------------------------------------------------------------

_EXPLICIT_NONE = object()  # sentinel: caller explicitly wants document_store=None


def _mock_upload_app(
    results: list[dict] | None = None,
    document_store=_EXPLICIT_NONE,
) -> MagicMock:
    """Mock CogBaseApp for upload_documents tests.

    By default provides an in-memory document store mock so background ingest
    tasks (which now call load_bytes) can complete successfully in tests.
    Pass ``document_store=None`` explicitly to test the no-store path.
    """
    from dataclasses import dataclass

    @dataclass
    class _FakeIngestResult:
        doc_id: str
        success: bool
        records_extracted: int
        error: Exception | None

    if results is None:
        results = [{"doc_id": "contract", "success": True, "records_extracted": 2, "error": None}]

    fake_results = [_FakeIngestResult(**r) for r in results]
    inst = MagicMock()
    inst.ingest_documents = AsyncMock(return_value=fake_results)
    inst.name = "my-contract-analyzer"

    if document_store is _EXPLICIT_NONE:
        # Default: a store that accepts save_bytes and returns them on load_bytes.
        _storage: dict[tuple, bytes] = {}

        async def _save(collection, doc_id, content):
            _storage[(collection, doc_id)] = content

        async def _load(collection, doc_id):
            try:
                return _storage[(collection, doc_id)]
            except KeyError:
                raise KeyError(doc_id)

        store = MagicMock()
        store.save_bytes = AsyncMock(side_effect=_save)
        store.load_bytes = AsyncMock(side_effect=_load)
        inst.document_store = store
    else:
        inst.document_store = document_store

    return inst


class TestUploadDocuments:
    @pytest.mark.asyncio
    async def test_upload_txt_returns_202_with_task_ids(self, client):
        mock_app = _mock_upload_app()
        await _create_app(client, mock_app)

        with patch("api.routers.applications.parse_to_markdown", return_value="parsed text"):
            resp = await client.post(
                "/applications/my-contract-analyzer/upload_documents",
                files=[("files", ("contract.txt", b"raw text", "text/plain"))],
            )

        assert resp.status_code == 202
        data = resp.json()
        assert data["total"] == 1
        assert len(data["task_ids"]) == 1
        assert isinstance(data["task_ids"][0], str)

    @pytest.mark.asyncio
    async def test_upload_doc_id_derived_from_filename_stem(self, client):
        mock_app = _mock_upload_app(
            results=[{"doc_id": "my_contract_2024", "success": True, "records_extracted": 1, "error": None}]
        )
        await _create_app(client, mock_app)

        with patch("api.routers.applications.parse_to_markdown", return_value="text"):
            await client.post(
                "/applications/my-contract-analyzer/upload_documents",
                files=[("files", ("my contract 2024.pdf", b"bytes", "application/pdf"))],
            )

        # Drain pending tasks.
        for _ in range(5):
            await asyncio.sleep(0)

        docs = mock_app.ingest_documents.call_args[0][0]
        assert docs[0].doc_id == "my_contract_2024"

    @pytest.mark.asyncio
    async def test_upload_parse_failure_returns_failed_task(self, client):
        mock_app = _mock_upload_app(results=[])
        await _create_app(client, mock_app)

        with patch("api.routers.applications.parse_to_markdown", side_effect=RuntimeError("bad pdf")):
            resp = await client.post(
                "/applications/my-contract-analyzer/upload_documents",
                files=[("files", ("broken.pdf", b"not a pdf", "application/pdf"))],
            )

        # Returns 202 with a task_id; the task has status=failed due to parse error.
        assert resp.status_code == 202
        data = resp.json()
        assert data["total"] == 1
        assert len(data["task_ids"]) == 1
        mock_app.ingest_documents.assert_not_called()

    @pytest.mark.asyncio
    async def test_upload_metadata_merged_into_document(self, client):
        mock_app = _mock_upload_app()
        await _create_app(client, mock_app)

        with patch("api.routers.applications.parse_to_markdown", return_value="text"):
            await client.post(
                "/applications/my-contract-analyzer/upload_documents",
                files=[("files", ("contract.txt", b"text", "text/plain"))],
                data={"metadata": '{"doc_type": "contract", "client": "acme"}'},
            )

        # Drain pending tasks.
        for _ in range(5):
            await asyncio.sleep(0)

        docs = mock_app.ingest_documents.call_args[0][0]
        assert docs[0].metadata["doc_type"] == "contract"
        assert docs[0].metadata["client"] == "acme"

    @pytest.mark.asyncio
    async def test_upload_auto_metadata_always_present(self, client):
        mock_app = _mock_upload_app()
        await _create_app(client, mock_app)

        with patch("api.routers.applications.parse_to_markdown", return_value="text"):
            await client.post(
                "/applications/my-contract-analyzer/upload_documents",
                files=[("files", ("report.pdf", b"bytes", "application/pdf"))],
            )

        # Drain pending tasks.
        for _ in range(5):
            await asyncio.sleep(0)

        docs = mock_app.ingest_documents.call_args[0][0]
        assert docs[0].metadata["source_filename"] == "report.pdf"
        assert docs[0].metadata["source_format"] == "pdf"

    @pytest.mark.asyncio
    async def test_upload_caller_metadata_overrides_auto_fields(self, client):
        mock_app = _mock_upload_app()
        await _create_app(client, mock_app)

        with patch("api.routers.applications.parse_to_markdown", return_value="text"):
            await client.post(
                "/applications/my-contract-analyzer/upload_documents",
                files=[("files", ("doc.txt", b"text", "text/plain"))],
                data={"metadata": '{"source_format": "custom"}'},
            )

        # Drain pending tasks.
        for _ in range(5):
            await asyncio.sleep(0)

        docs = mock_app.ingest_documents.call_args[0][0]
        assert docs[0].metadata["source_format"] == "custom"

    @pytest.mark.asyncio
    async def test_upload_invalid_json_metadata_returns_422(self, client):
        mock_app = _mock_upload_app()
        await _create_app(client, mock_app)

        resp = await client.post(
            "/applications/my-contract-analyzer/upload_documents",
            files=[("files", ("doc.txt", b"text", "text/plain"))],
            data={"metadata": "not json"},
        )

        assert resp.status_code == 422
        assert "metadata" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_upload_non_dict_metadata_returns_422(self, client):
        mock_app = _mock_upload_app()
        await _create_app(client, mock_app)

        resp = await client.post(
            "/applications/my-contract-analyzer/upload_documents",
            files=[("files", ("doc.txt", b"text", "text/plain"))],
            data={"metadata": '["not", "an", "object"]'},
        )

        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_upload_saves_original_bytes_to_document_store(self, client):
        store = MagicMock()
        store.save_bytes = AsyncMock()
        mock_app = _mock_upload_app(document_store=store)
        await _create_app(client, mock_app)

        raw_bytes = b"original pdf bytes"
        with patch("api.routers.applications.parse_to_markdown", return_value="text"):
            await client.post(
                "/applications/my-contract-analyzer/upload_documents",
                files=[("files", ("invoice.pdf", raw_bytes, "application/pdf"))],
            )

        store.save_bytes.assert_awaited_once_with(
            "my-contract-analyzer", "originals/invoice.pdf", raw_bytes
        )

    @pytest.mark.asyncio
    async def test_upload_skips_store_save_when_no_document_store(self, client):
        mock_app = _mock_upload_app(document_store=None)
        await _create_app(client, mock_app)

        with patch("api.routers.applications.parse_to_markdown", return_value="text"):
            resp = await client.post(
                "/applications/my-contract-analyzer/upload_documents",
                files=[("files", ("doc.txt", b"text", "text/plain"))],
            )

        assert resp.status_code == 202

    @pytest.mark.asyncio
    async def test_upload_404_when_app_not_found(self, client):
        resp = await client.post(
            "/applications/nonexistent/upload_documents",
            files=[("files", ("doc.txt", b"text", "text/plain"))],
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_upload_background_task_runs(self, client):
        """Background ingest fires after successful parse."""
        mock_app = _mock_upload_app()
        await _create_app(client, mock_app)

        with patch("api.routers.applications.parse_to_markdown", return_value="text"):
            resp = await client.post(
                "/applications/my-contract-analyzer/upload_documents",
                files=[("files", ("contract.txt", b"text", "text/plain"))],
            )

        assert resp.status_code == 202

        # Drain pending tasks.
        for _ in range(5):
            await asyncio.sleep(0)

        mock_app.ingest_documents.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_upload_mixed_success_and_parse_failure(self, client):
        mock_app = _mock_upload_app(
            results=[{"doc_id": "good", "success": True, "records_extracted": 1, "error": None}]
        )
        await _create_app(client, mock_app)

        def _parse(content, filename):
            if "bad" in filename:
                raise RuntimeError("cannot parse")
            return "parsed text"

        with patch("api.routers.applications.parse_to_markdown", side_effect=_parse):
            resp = await client.post(
                "/applications/my-contract-analyzer/upload_documents",
                files=[
                    ("files", ("good.txt", b"text", "text/plain")),
                    ("files", ("bad.pdf", b"broken", "application/pdf")),
                ],
            )

        # Both files get a task ID (one failed at parse, one queued for background).
        assert resp.status_code == 202
        data = resp.json()
        assert data["total"] == 2
        assert len(data["task_ids"]) == 2

    @pytest.mark.asyncio
    async def test_upload_task_params_json_persists_doc_path_and_metadata(self, client):
        """Task record must carry doc_path + doc_metadata so a restarted node can re-run it."""
        mock_app = _mock_upload_app()
        await _create_app(client, mock_app)

        with patch("api.routers.applications.parse_to_markdown", return_value="text"):
            resp = await client.post(
                "/applications/my-contract-analyzer/upload_documents",
                files=[("files", ("invoice.pdf", b"bytes", "application/pdf"))],
                data={"metadata": '{"client": "acme"}'},
            )

        assert resp.status_code == 202
        task_id = resp.json()["task_ids"][0]

        task_resp = await client.get(f"/applications/my-contract-analyzer/tasks/{task_id}")
        assert task_resp.status_code == 200
        params = json.loads(task_resp.json()["params_json"])
        assert params["doc_path"] == "originals/invoice.pdf"
        assert params["doc_metadata"]["source_filename"] == "invoice.pdf"
        assert params["doc_metadata"]["source_format"] == "pdf"
        assert params["doc_metadata"]["client"] == "acme"

    @pytest.mark.asyncio
    async def test_upload_ingest_reads_bytes_from_document_store(self, client):
        """Background task must load bytes from document_store, not from an in-memory closure."""
        raw_bytes = b"original invoice bytes"
        mock_app = _mock_upload_app()
        await _create_app(client, mock_app)

        with patch("api.routers.applications.parse_to_markdown", return_value="text") as mock_parse:
            await client.post(
                "/applications/my-contract-analyzer/upload_documents",
                files=[("files", ("invoice.pdf", raw_bytes, "application/pdf"))],
            )
            # Drain pending background tasks.
            for _ in range(5):
                await asyncio.sleep(0)

        # parse_to_markdown must receive the bytes that came back from load_bytes (same content).
        assert mock_parse.called
        content_passed = mock_parse.call_args[0][0]
        assert content_passed == raw_bytes

    @pytest.mark.asyncio
    async def test_upload_load_bytes_failure_marks_task_failed(self, client):
        """If load_bytes raises, the task must be marked failed with a clear error message."""
        store = MagicMock()
        store.save_bytes = AsyncMock()
        store.load_bytes = AsyncMock(side_effect=KeyError("invoice.pdf not found in store"))
        mock_app = _mock_upload_app(document_store=store)
        await _create_app(client, mock_app)

        with patch("api.routers.applications.parse_to_markdown", return_value="text"):
            resp = await client.post(
                "/applications/my-contract-analyzer/upload_documents",
                files=[("files", ("invoice.pdf", b"bytes", "application/pdf"))],
            )

        task_id = resp.json()["task_ids"][0]

        # Drain pending background tasks.
        for _ in range(10):
            await asyncio.sleep(0)

        task_resp = await client.get(f"/applications/my-contract-analyzer/tasks/{task_id}")
        data = task_resp.json()
        assert data["status"] == "failed"
        assert "load" in data["error"].lower() or "not found" in data["error"].lower()
        mock_app.ingest_documents.assert_not_called()


# ---------------------------------------------------------------------------
# Helpers shared by query tests
# ---------------------------------------------------------------------------

def _make_query_result(
    answer: str = "The notice period is 60 days.",
    structured_records: list[dict] | None = None,
    chunks: list[Chunk] | None = None,
    document_slices: list[DocumentSlice] | None = None,
) -> QueryResult:
    return QueryResult(
        answer=answer,
        structured_records=structured_records or [],
        chunks=chunks or [],
        document_slices=document_slices or [],
    )


def _mock_query_app(result: QueryResult) -> MagicMock:
    """Build a mock CogBaseApp whose query_stream yields tokens then a QueryResult."""
    inst = MagicMock()

    async def _query_stream(text: str, history: list[dict] | None = None):
        _ = history
        for token in result.answer.split():
            yield token + " "
        yield result

    inst.query_stream = _query_stream
    return inst


def _parse_sse(body: str) -> list[str]:
    """Return the data payload of each SSE event (excluding blank separators)."""
    return [
        line[len("data: "):]
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


async def _create_app(client, mock_app: MagicMock) -> None:
    with patch("api.routers.applications.build_app", new_callable=AsyncMock, return_value=mock_app):
        resp = await client.post(
            "/applications",
            files={"bundle": ("bundle.zip", _VALID_BUNDLE, "application/zip")},
        )
    assert resp.status_code == 201


# ---------------------------------------------------------------------------
# POST /applications/{app_name}/query
# ---------------------------------------------------------------------------

class TestQueryApplication:
    @pytest.mark.asyncio
    async def test_returns_200_with_answer(self, client):
        result = _make_query_result("The notice period is 60 days.")
        await _create_app(client, _mock_query_app(result))

        resp = await client.post(
            "/applications/my-contract-analyzer/query",
            json={
                "text": "what is the notice period?",
                "history": [{"role": "user", "content": "summarize the contract first"}],
            },
        )

        assert resp.status_code == 200
        assert resp.json()["answer"] == result.answer

    @pytest.mark.asyncio
    async def test_includes_structured_records(self, client):
        records = [{"contract_type": "NDA", "doc_id": "c-001"}]
        result = _make_query_result(
            answer="Found 1 record(s): contract_type: NDA, doc_id: c-001",
            structured_records=records,
        )
        await _create_app(client, _mock_query_app(result))

        resp = await client.post(
            "/applications/my-contract-analyzer/query",
            json={
                "text": "list NDA contracts",
                "history": [{"role": "assistant", "content": "I found one contract already."}],
            },
        )

        data = resp.json()
        assert len(data["structured_records"]) == 1
        assert data["structured_records"][0]["contract_type"] == "NDA"

    @pytest.mark.asyncio
    async def test_includes_chunks(self, client):
        chunk = Chunk(chunk_id="doc_0_0", doc_id="doc_0", text="Payment terms are net 30.")
        result = _make_query_result(answer="Net 30 [doc_0_0].", chunks=[chunk])
        await _create_app(client, _mock_query_app(result))

        resp = await client.post(
            "/applications/my-contract-analyzer/query",
            json={"text": "payment terms?", "history": []},
        )

        data = resp.json()
        assert len(data["chunks"]) == 1
        assert data["chunks"][0]["chunk_id"] == "doc_0_0"
        assert data["chunks"][0]["doc_id"] == "doc_0"
        assert data["chunks"][0]["text"] == "Payment terms are net 30."

    @pytest.mark.asyncio
    async def test_includes_document_slices(self, client):
        doc_slice = DocumentSlice(doc_id="doc_0", offset=100, length=42, text="Relevant excerpt.")
        result = _make_query_result(answer="See the excerpt.", document_slices=[doc_slice])
        await _create_app(client, _mock_query_app(result))

        resp = await client.post(
            "/applications/my-contract-analyzer/query",
            json={"text": "show me the excerpt", "history": []},
        )

        data = resp.json()
        assert len(data["document_slices"]) == 1
        s = data["document_slices"][0]
        assert s["doc_id"] == "doc_0"
        assert s["offset"] == 100
        assert s["length"] == 42
        assert s["text"] == "Relevant excerpt."

    @pytest.mark.asyncio
    async def test_404_when_app_not_found(self, client):
        resp = await client.post(
            "/applications/nonexistent/query",
            json={"text": "q", "history": []},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_404_when_app_not_active(self, client):
        with patch("api.routers.applications.build_app", new_callable=AsyncMock, side_effect=RuntimeError("setup boom")):
            await client.post(
                "/applications",
                files={"bundle": ("bundle.zip", _VALID_BUNDLE, "application/zip")},
            )

        resp = await client.post(
            "/applications/my-contract-analyzer/query",
            json={"text": "q", "history": []},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_retries_on_first_failure(self, client):
        result = _make_query_result("Retry worked.")

        call_count = 0

        async def _flaky_stream(text: str, history: list[dict] | None = None):
            _ = history
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient failure")
            yield result.answer
            yield result

        failing_then_good = _mock_query_app(result)
        failing_then_good.query_stream = _flaky_stream

        await _create_app(client, failing_then_good)

        with patch("api.routers.applications.build_app", new_callable=AsyncMock, return_value=failing_then_good):
            resp = await client.post(
                "/applications/my-contract-analyzer/query",
                json={"text": "q", "history": []},
            )

        assert resp.status_code == 200
        assert resp.json()["answer"] == "Retry worked."


# ---------------------------------------------------------------------------
# POST /applications/{app_name}/query/stream
# ---------------------------------------------------------------------------

class TestQueryApplicationStream:
    @pytest.mark.asyncio
    async def test_returns_event_stream_content_type(self, client):
        result = _make_query_result()
        await _create_app(client, _mock_query_app(result))

        resp = await client.post(
            "/applications/my-contract-analyzer/query/stream",
            json={"text": "q", "history": []},
        )

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    @pytest.mark.asyncio
    async def test_streams_token_events(self, client):
        result = _make_query_result("Hello world.")
        await _create_app(client, _mock_query_app(result))

        resp = await client.post(
            "/applications/my-contract-analyzer/query/stream",
            json={"text": "q", "history": []},
        )

        events = _parse_sse(resp.text)
        token_events = [e for e in events if e != "[DONE]"]
        tokens = [json.loads(e) for e in token_events if "token" in json.loads(e)]
        assert len(tokens) > 0
        assembled = "".join(t["token"] for t in tokens)
        assert assembled.strip() == result.answer

    @pytest.mark.asyncio
    async def test_final_result_event_contains_answer_and_chunks(self, client):
        chunk = Chunk(chunk_id="doc_0_0", doc_id="doc_0", text="relevant passage")
        result = _make_query_result("The answer [doc_0_0].", chunks=[chunk])
        await _create_app(client, _mock_query_app(result))

        resp = await client.post(
            "/applications/my-contract-analyzer/query/stream",
            json={"text": "q", "history": []},
        )

        events = _parse_sse(resp.text)
        result_events = [json.loads(e) for e in events if e != "[DONE]" and "result" in json.loads(e)]
        assert len(result_events) == 1
        payload = result_events[0]["result"]
        assert payload["answer"] == "The answer [doc_0_0]."
        assert len(payload["chunks"]) == 1
        assert payload["chunks"][0]["chunk_id"] == "doc_0_0"

    @pytest.mark.asyncio
    async def test_ends_with_done_sentinel(self, client):
        result = _make_query_result()
        await _create_app(client, _mock_query_app(result))

        resp = await client.post(
            "/applications/my-contract-analyzer/query/stream",
            json={"text": "q", "history": []},
        )

        events = _parse_sse(resp.text)
        assert events[-1] == "[DONE]"

    @pytest.mark.asyncio
    async def test_result_includes_structured_records(self, client):
        records = [{"contract_type": "NDA", "doc_id": "c-001"}]
        result = _make_query_result(answer="Found 1 record(s).", structured_records=records)
        await _create_app(client, _mock_query_app(result))

        resp = await client.post(
            "/applications/my-contract-analyzer/query/stream",
            json={"text": "q", "history": []},
        )

        events = _parse_sse(resp.text)
        result_events = [json.loads(e) for e in events if e != "[DONE]" and "result" in json.loads(e)]
        payload = result_events[0]["result"]
        assert payload["structured_records"] == records

    @pytest.mark.asyncio
    async def test_result_includes_chunks(self, client):
        chunk = Chunk(chunk_id="doc_1_0", doc_id="doc_1", text="termination clause text", char_offset=200, char_length=50)
        result = _make_query_result(answer="See [doc_1_0] for details.", chunks=[chunk])
        await _create_app(client, _mock_query_app(result))

        resp = await client.post(
            "/applications/my-contract-analyzer/query/stream",
            json={"text": "termination clause?", "history": []},
        )

        events = _parse_sse(resp.text)
        result_events = [json.loads(e) for e in events if e != "[DONE]" and "result" in json.loads(e)]
        payload = result_events[0]["result"]
        assert len(payload["chunks"]) == 1
        c = payload["chunks"][0]
        assert c["chunk_id"] == "doc_1_0"
        assert c["doc_id"] == "doc_1"
        assert c["text"] == "termination clause text"
        assert c["char_offset"] == 200
        assert c["char_length"] == 50

    @pytest.mark.asyncio
    async def test_404_when_app_not_found(self, client):
        resp = await client.post(
            "/applications/nonexistent/query/stream",
            json={"text": "q", "history": []},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_error_event_on_stream_failure(self, client):
        inst = MagicMock()

        async def _failing_stream(text: str, history: list[dict] | None = None):
            _ = history
            raise RuntimeError("boom")
            yield  # make it an async generator

        inst.query_stream = _failing_stream
        await _create_app(client, inst)

        resp = await client.post(
            "/applications/my-contract-analyzer/query/stream",
            json={"text": "q", "history": []},
        )

        assert resp.status_code == 200
        events = _parse_sse(resp.text)
        error_events = [json.loads(e) for e in events if e != "[DONE]" and "error" in json.loads(e)]
        assert len(error_events) == 1
        assert events[-1] == "[DONE]"


# ---------------------------------------------------------------------------
# Helpers shared by collection endpoint tests
# ---------------------------------------------------------------------------


def _make_collections_bundle(
    structured_collections: list[str] | None = None,
    vector_collections: list[str] | None = None,
) -> bytes:
    lines = ["name: my-contract-analyzer", "llm:", "  provider: openai", "  model: gpt-4o-mini", "  api_key: sk-test"]
    if vector_collections:
        lines.append("vector_collections:")
        for vc in vector_collections:
            lines += [f"  - name: {vc}", f"    description: {vc} chunks", "    dimensions: 1536"]
    if structured_collections:
        lines.append("structured_collections:")
        for sc in structured_collections:
            lines += [
                f"  - name: {sc}",
                f"    description: {sc} records",
                "    schema: '{}'",
                "    primary_fields: [doc_id]",
            ]
    return _make_bundle("\n".join(lines).encode())


async def _create_collections_app(
    client,
    mock_app: MagicMock,
    structured_collections: list[str] | None = None,
    vector_collections: list[str] | None = None,
) -> None:
    bundle = _make_collections_bundle(structured_collections, vector_collections)
    with patch("api.routers.applications.build_app", new_callable=AsyncMock, return_value=mock_app):
        resp = await client.post(
            "/applications",
            files={"bundle": ("bundle.zip", bundle, "application/zip")},
        )
    assert resp.status_code == 201


def _mock_query_store_app(structured_records: list[dict] | None = None) -> MagicMock:
    """Build a mock CogBaseApp with a structured store for query_collection tests."""
    inst = MagicMock()
    runner = MagicMock()
    inst.query_runner = runner
    store = MagicMock()
    store.query = AsyncMock(return_value=structured_records or [])
    runner.structured_store = store
    return inst


# ---------------------------------------------------------------------------
# GET /applications/{app_name}/collections
# ---------------------------------------------------------------------------


class TestListCollections:
    @pytest.mark.asyncio
    async def test_returns_structured_and_vector(self, client):
        await _create_collections_app(
            client, MagicMock(),
            structured_collections=["contracts", "parties"],
            vector_collections=["doc_chunks"],
        )

        resp = await client.get("/applications/my-contract-analyzer/collections")

        assert resp.status_code == 200
        body = resp.json()
        assert set(body["structured"]) == {"contracts", "parties"}
        assert body["vector"] == ["doc_chunks"]

    @pytest.mark.asyncio
    async def test_no_structured_collections_returns_empty_structured(self, client):
        await _create_collections_app(client, MagicMock(), vector_collections=["doc_chunks"])

        resp = await client.get("/applications/my-contract-analyzer/collections")

        assert resp.status_code == 200
        body = resp.json()
        assert body["structured"] == []
        assert body["vector"] == ["doc_chunks"]

    @pytest.mark.asyncio
    async def test_no_vector_collections_returns_empty_vector(self, client):
        await _create_collections_app(client, MagicMock(), structured_collections=["contracts"])

        resp = await client.get("/applications/my-contract-analyzer/collections")

        assert resp.status_code == 200
        body = resp.json()
        assert body["structured"] == ["contracts"]
        assert body["vector"] == []

    @pytest.mark.asyncio
    async def test_no_collections_returns_empty_lists(self, client):
        await _create_app(client, MagicMock())

        resp = await client.get("/applications/my-contract-analyzer/collections")

        assert resp.status_code == 200
        body = resp.json()
        assert body["structured"] == []
        assert body["vector"] == []

    @pytest.mark.asyncio
    async def test_404_when_app_not_found(self, client):
        resp = await client.get("/applications/nonexistent/collections")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /applications/{app_name}/collections/{collection}/query
# ---------------------------------------------------------------------------


class TestQueryCollection:
    @pytest.mark.asyncio
    async def test_returns_records_for_structured_collection(self, client):
        records = [{"type": "NDA", "doc_id": "c-001"}, {"type": "NDA", "doc_id": "c-002"}]
        mock_app = _mock_query_store_app(structured_records=records)
        await _create_collections_app(client, mock_app, structured_collections=["contracts"])

        resp = await client.post(
            "/applications/my-contract-analyzer/collections/contracts/query",
            json={},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["collection"] == "contracts"
        assert body["records"] == records
        assert body["total"] == 2

    @pytest.mark.asyncio
    async def test_passes_filters_to_store(self, client):
        mock_app = _mock_query_store_app(structured_records=[{"type": "NDA", "doc_id": "c-001"}])
        await _create_collections_app(client, mock_app, structured_collections=["contracts"])

        resp = await client.post(
            "/applications/my-contract-analyzer/collections/contracts/query",
            json={"filters": [{"field": "type", "op": "=", "value": "NDA"}]},
        )

        assert resp.status_code == 200
        call_args = mock_app.query_runner.structured_store.query.call_args[0]
        filters_passed = call_args[1]
        assert len(filters_passed) == 1
        assert filters_passed[0].field == "type"
        assert filters_passed[0].value == "NDA"

    @pytest.mark.asyncio
    async def test_passes_fields_to_store(self, client):
        mock_app = _mock_query_store_app(structured_records=[{"type": "NDA"}])
        await _create_collections_app(client, mock_app, structured_collections=["contracts"])

        resp = await client.post(
            "/applications/my-contract-analyzer/collections/contracts/query",
            json={"fields": ["type"]},
        )

        assert resp.status_code == 200
        call_args = mock_app.query_runner.structured_store.query.call_args[0]
        assert call_args[2] == ["type"]

    @pytest.mark.asyncio
    async def test_empty_filters_passes_none_to_store(self, client):
        mock_app = _mock_query_store_app()
        await _create_collections_app(client, mock_app, structured_collections=["contracts"])

        resp = await client.post(
            "/applications/my-contract-analyzer/collections/contracts/query",
            json={},
        )

        assert resp.status_code == 200
        call_args = mock_app.query_runner.structured_store.query.call_args[0]
        assert call_args[1] is None  # empty filters → None
        assert call_args[2] is None  # absent fields → None

    @pytest.mark.asyncio
    async def test_vector_collection_returns_400(self, client):
        await _create_collections_app(client, MagicMock(), vector_collections=["doc_chunks"])

        resp = await client.post(
            "/applications/my-contract-analyzer/collections/doc_chunks/query",
            json={},
        )

        assert resp.status_code == 400
        assert "vector" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_unknown_collection_returns_404(self, client):
        await _create_collections_app(
            client, MagicMock(),
            structured_collections=["contracts"],
            vector_collections=["doc_chunks"],
        )

        resp = await client.post(
            "/applications/my-contract-analyzer/collections/nonexistent/query",
            json={},
        )

        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_404_when_app_not_found(self, client):
        resp = await client.post(
            "/applications/nonexistent/collections/contracts/query",
            json={},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /applications/{app_name}/workflows/{workflow_name}/docs
# ---------------------------------------------------------------------------


def _make_doc_record(app_name: str, doc_id: str, status: str = "active") -> DocRecord:
    return DocRecord(
        app_name=app_name,
        doc_id=doc_id,
        status=status,
        ingested_at="2024-01-01T00:00:00+00:00",
        metadata='{"source_filename": "' + doc_id + '.pdf"}',
    )


class TestListWorkflowDocs:
    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_workflow_records(self, app_overrides):
        client = app_overrides["client"]
        await _create_app(client, _mock_app_instance())

        resp = await client.get(
            "/applications/my-contract-analyzer/workflows/summarize/docs"
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["app_name"] == "my-contract-analyzer"
        assert body["workflow_name"] == "summarize"
        assert body["docs"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_returns_docs_with_workflow_status(self, app_overrides):
        client = app_overrides["client"]
        system_store: SystemStore = app_overrides["system_store"]
        await system_store.save_doc(_make_doc_record("my-contract-analyzer", "doc-1"))
        await system_store.upsert_doc_workflow_status(
            "my-contract-analyzer", "doc-1", "summarize", "done"
        )
        await _create_app(client, _mock_app_instance())

        resp = await client.get(
            "/applications/my-contract-analyzer/workflows/summarize/docs"
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["docs"][0]["doc_id"] == "doc-1"
        assert body["docs"][0]["workflow_status"] == "done"

    @pytest.mark.asyncio
    async def test_filters_by_status_query_param(self, app_overrides):
        client = app_overrides["client"]
        system_store: SystemStore = app_overrides["system_store"]
        for doc_id, wf_status in [("doc-1", "done"), ("doc-2", "pending"), ("doc-3", "done")]:
            await system_store.save_doc(_make_doc_record("my-contract-analyzer", doc_id))
            await system_store.upsert_doc_workflow_status(
                "my-contract-analyzer", doc_id, "summarize", wf_status
            )
        await _create_app(client, _mock_app_instance())

        resp = await client.get(
            "/applications/my-contract-analyzer/workflows/summarize/docs?status=done"
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 2
        returned_ids = {d["doc_id"] for d in body["docs"]}
        assert returned_ids == {"doc-1", "doc-3"}

    @pytest.mark.asyncio
    async def test_excludes_docs_not_in_active_doc_registry(self, app_overrides):
        client = app_overrides["client"]
        system_store: SystemStore = app_overrides["system_store"]
        # doc-1 is active; doc-2 has a workflow record but no entry in the doc registry
        await system_store.save_doc(_make_doc_record("my-contract-analyzer", "doc-1"))
        for doc_id in ["doc-1", "doc-2"]:
            await system_store.upsert_doc_workflow_status(
                "my-contract-analyzer", doc_id, "summarize", "done"
            )
        await _create_app(client, _mock_app_instance())

        resp = await client.get(
            "/applications/my-contract-analyzer/workflows/summarize/docs"
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["docs"][0]["doc_id"] == "doc-1"

    @pytest.mark.asyncio
    async def test_returns_404_when_app_not_found(self, app_overrides):
        client = app_overrides["client"]
        resp = await client.get(
            "/applications/nonexistent/workflows/summarize/docs"
        )
        assert resp.status_code == 404
        assert "nonexistent" in resp.json()["detail"]
