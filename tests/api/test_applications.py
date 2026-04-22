"""Integration tests for the /applications REST endpoints.

All tests use httpx.AsyncClient pointed at the FastAPI app with dependency
overrides injected — no real LLM calls or file I/O happens.
"""

from __future__ import annotations

import io
import json
import textwrap
import zipfile
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from api.dependencies import (
    get_app_cache,
    get_system_config,
    get_system_store,
    get_system_structured_store,
)
from api.main import app
from api.app_cache import AppCache
from api.system_config import SystemConfig
from api.system_store import SystemStore
from cogbase.engine.generation.base import GenerationResult
from cogbase.engine.retrieval.base import RetrievalResult
from cogbase.engine.router import QueryPattern, RouteResult
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


def _system_config() -> SystemConfig:
    return SystemConfig.model_validate({"system_db": {"type": "memory"}})


_VALID_CONFIG_YAML = textwrap.dedent("""\
    name: my-contract-analyzer
    llm:
      provider: openai
      model: gpt-4o-mini
""").encode()

_VALID_BUNDLE = _make_bundle(_VALID_CONFIG_YAML)


def _mock_app_instance() -> MagicMock:
    """Minimal mock that satisfies the build_app / app lifecycle contract."""
    inst = MagicMock()
    inst.setup = AsyncMock()
    return inst


@pytest_asyncio.fixture
async def client():
    """AsyncClient with all external dependencies swapped out."""
    system_store = _make_system_store()
    await system_store.setup()
    app_cache = _make_app_cache()
    system_structured_store = InMemoryStructuredStore()

    app.dependency_overrides[get_system_store] = lambda: system_store
    app.dependency_overrides[get_app_cache] = lambda: app_cache
    app.dependency_overrides[get_system_config] = lambda: _system_config()
    app.dependency_overrides[get_system_structured_store] = lambda: system_structured_store

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# POST /applications
# ---------------------------------------------------------------------------

class TestCreateApplication:
    @pytest.mark.asyncio
    async def test_create_returns_201(self, client):
        with patch("api.routers.applications.build_app", return_value=_mock_app_instance()):
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
        with patch("api.routers.applications.build_app", return_value=_mock_app_instance()):
            resp = await client.post(
                "/applications",
                files={"bundle": ("bundle.zip", _VALID_BUNDLE, "application/zip")},
            )
        assert resp.status_code == 201
        config = resp.json()["config"]
        assert config["name"] == "my-contract-analyzer"

    @pytest.mark.asyncio
    async def test_create_conflict_returns_409(self, client):
        with patch("api.routers.applications.build_app", return_value=_mock_app_instance()):
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
    async def test_create_records_error_status_when_setup_fails(self, client):
        failing_app = MagicMock()
        failing_app.setup = AsyncMock(side_effect=RuntimeError("setup boom"))
        with patch("api.routers.applications.build_app", return_value=failing_app):
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
        """Schema and prompt filenames in config.yaml are replaced with file contents."""
        schema_json = b'{"type":"object","properties":{"value":{"type":"string"}}}'
        prompt_txt = b"Extract contract fields."
        config_yaml = textwrap.dedent("""\
            name: my-contract-analyzer
            llm:
              provider: openai
              model: gpt-4o-mini
            structured_collections:
              - name: contract_extraction
                schema: extraction_schema.json
                extractor:
                  type: llm
                  prompt: extraction_prompt.txt
            pipeline:
              steps:
                - tool: extract-structured
                  collection: contract_extraction
        """).encode()
        bundle = _make_bundle(
            config_yaml,
            files={
                "extraction_schema.json": schema_json,
                "extraction_prompt.txt": prompt_txt,
            },
        )
        captured: list = []

        def _capture(config, **kwargs):
            captured.append(config)
            return _mock_app_instance()

        with patch("api.routers.applications.build_app", side_effect=_capture):
            resp = await client.post(
                "/applications",
                files={"bundle": ("bundle.zip", bundle, "application/zip")},
            )

        assert resp.status_code == 201
        sc = captured[0].structured_collections[0]
        assert sc.schema_ == schema_json.decode()
        assert sc.extractor.prompt == prompt_txt.decode()


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
        bundle_a = _make_bundle(b"name: app-a\nllm:\n  model: gpt-4o-mini\n")
        bundle_b = _make_bundle(b"name: app-b\nllm:\n  model: gpt-4o-mini\n")
        with patch("api.routers.applications.build_app", return_value=_mock_app_instance()):
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
        with patch("api.routers.applications.build_app", return_value=_mock_app_instance()):
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
        with patch("api.routers.applications.build_app", return_value=_mock_app_instance()):
            await client.post(
                "/applications",
                files={"bundle": ("bundle.zip", _VALID_BUNDLE, "application/zip")},
            )

        updated_yaml = _VALID_CONFIG_YAML.replace(b"gpt-4o-mini", b"gpt-4o")
        updated_bundle = _make_bundle(updated_yaml)
        with patch("api.routers.applications.build_app", return_value=_mock_app_instance()):
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
        bundle_a = _make_bundle(b"name: app-a\nllm:\n  model: gpt-4o-mini\n")
        bundle_b = _make_bundle(b"name: app-b\nllm:\n  model: gpt-4o-mini\n")
        with patch("api.routers.applications.build_app", return_value=_mock_app_instance()):
            await client.post("/applications", files={"bundle": ("a.zip", bundle_a, "application/zip")})
            await client.post("/applications", files={"bundle": ("b.zip", bundle_b, "application/zip")})

        with patch("api.routers.applications.build_app", return_value=_mock_app_instance()):
            resp = await client.patch(
                "/applications/app-a",
                files={"bundle": ("bundle.zip", bundle_b, "application/zip")},
            )
        assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_update_records_error_when_setup_fails(self, client):
        with patch("api.routers.applications.build_app", return_value=_mock_app_instance()):
            await client.post(
                "/applications",
                files={"bundle": ("bundle.zip", _VALID_BUNDLE, "application/zip")},
            )

        failing_app = MagicMock()
        failing_app.setup = AsyncMock(side_effect=RuntimeError("update boom"))
        with patch("api.routers.applications.build_app", return_value=failing_app):
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
        with patch("api.routers.applications.build_app", return_value=_mock_app_instance()):
            await client.post(
                "/applications",
                files={"bundle": ("bundle.zip", _VALID_BUNDLE, "application/zip")},
            )
        resp = await client.delete("/applications/my-contract-analyzer")
        assert resp.status_code == 204

    @pytest.mark.asyncio
    async def test_delete_removes_from_list(self, client):
        with patch("api.routers.applications.build_app", return_value=_mock_app_instance()):
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

        with patch("api.routers.applications.build_app", return_value=_mock_app_instance()):
            await client.post(
                "/applications",
                files={"bundle": ("bundle.zip", _VALID_BUNDLE, "application/zip")},
            )
        assert app_cache.get("my-contract-analyzer") is not None

        await client.delete("/applications/my-contract-analyzer")
        assert app_cache.get("my-contract-analyzer") is None


# ---------------------------------------------------------------------------
# POST /applications/{app_name}/ingest_documents
# ---------------------------------------------------------------------------

def _mock_ingest_app(results: list[dict] | None = None) -> MagicMock:
    """Build a mock CogBaseApp whose ingest_documents returns IngestResult-like objects."""
    from dataclasses import dataclass

    @dataclass
    class _FakeIngestResult:
        doc_id: str
        success: bool
        records_extracted: int
        error: Exception | None

    if results is None:
        results = [{"doc_id": "doc-1", "success": True, "records_extracted": 3, "error": None}]

    fake_results = [_FakeIngestResult(**r) for r in results]

    inst = MagicMock()
    inst.setup = AsyncMock()
    inst.ingest_documents = AsyncMock(return_value=fake_results)
    return inst


class TestIngestDocuments:
    @pytest.mark.asyncio
    async def test_ingest_returns_200_with_results(self, client):
        mock_app = _mock_ingest_app()
        await _create_app(client, mock_app)

        resp = await client.post(
            "/applications/my-contract-analyzer/ingest_documents",
            json={"documents": [{"doc_id": "doc-1", "text": "Contract text here."}]},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["doc_id"] == "doc-1"
        assert data["results"][0]["success"] is True
        assert data["results"][0]["records_extracted"] == 3
        assert data["results"][0]["error"] is None

    @pytest.mark.asyncio
    async def test_ingest_multiple_documents(self, client):
        results = [
            {"doc_id": "doc-1", "success": True, "records_extracted": 2, "error": None},
            {"doc_id": "doc-2", "success": True, "records_extracted": 5, "error": None},
        ]
        mock_app = _mock_ingest_app(results)
        await _create_app(client, mock_app)

        resp = await client.post(
            "/applications/my-contract-analyzer/ingest_documents",
            json={
                "documents": [
                    {"doc_id": "doc-1", "text": "First contract."},
                    {"doc_id": "doc-2", "text": "Second contract."},
                ]
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 2
        doc_ids = {r["doc_id"] for r in data["results"]}
        assert doc_ids == {"doc-1", "doc-2"}

    @pytest.mark.asyncio
    async def test_ingest_partial_failure_reported_per_document(self, client):
        results = [
            {"doc_id": "doc-ok", "success": True, "records_extracted": 1, "error": None},
            {"doc_id": "doc-bad", "success": False, "records_extracted": 0, "error": ValueError("parse error")},
        ]
        mock_app = _mock_ingest_app(results)
        await _create_app(client, mock_app)

        resp = await client.post(
            "/applications/my-contract-analyzer/ingest_documents",
            json={
                "documents": [
                    {"doc_id": "doc-ok", "text": "Good doc."},
                    {"doc_id": "doc-bad", "text": "Bad doc."},
                ]
            },
        )

        assert resp.status_code == 200
        by_id = {r["doc_id"]: r for r in resp.json()["results"]}
        assert by_id["doc-ok"]["success"] is True
        assert by_id["doc-bad"]["success"] is False
        assert "parse error" in by_id["doc-bad"]["error"]

    @pytest.mark.asyncio
    async def test_ingest_404_when_app_not_found(self, client):
        resp = await client.post(
            "/applications/nonexistent/ingest_documents",
            json={"documents": [{"doc_id": "doc-1", "text": "text"}]},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_ingest_404_when_app_not_active(self, client):
        failing = MagicMock()
        failing.setup = AsyncMock(side_effect=RuntimeError("setup boom"))
        with patch("api.routers.applications.build_app", return_value=failing):
            await client.post(
                "/applications",
                files={"bundle": ("bundle.zip", _VALID_BUNDLE, "application/zip")},
            )

        resp = await client.post(
            "/applications/my-contract-analyzer/ingest_documents",
            json={"documents": [{"doc_id": "doc-1", "text": "text"}]},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_ingest_passes_concurrency(self, client):
        mock_app = _mock_ingest_app()
        await _create_app(client, mock_app)

        await client.post(
            "/applications/my-contract-analyzer/ingest_documents",
            json={
                "documents": [{"doc_id": "doc-1", "text": "text"}],
                "concurrency": 10,
            },
        )

        _, kwargs = mock_app.ingest_documents.call_args
        assert kwargs.get("concurrency") == 10

    @pytest.mark.asyncio
    async def test_ingest_retries_on_first_failure(self, client):
        from dataclasses import dataclass

        @dataclass
        class _FakeResult:
            doc_id: str
            success: bool
            records_extracted: int
            error: Exception | None

        good_result = [_FakeResult(doc_id="doc-1", success=True, records_extracted=1, error=None)]
        call_count = 0

        async def _flaky_ingest(documents, *, concurrency=5):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient ingest failure")
            return good_result

        mock_app = _mock_ingest_app()
        mock_app.ingest_documents = _flaky_ingest
        await _create_app(client, mock_app)

        with patch("api.routers.applications.build_app", return_value=mock_app):
            resp = await client.post(
                "/applications/my-contract-analyzer/ingest_documents",
                json={"documents": [{"doc_id": "doc-1", "text": "text"}]},
            )

        assert resp.status_code == 200
        assert resp.json()["results"][0]["success"] is True

    @pytest.mark.asyncio
    async def test_ingest_with_metadata(self, client):
        mock_app = _mock_ingest_app()
        await _create_app(client, mock_app)

        resp = await client.post(
            "/applications/my-contract-analyzer/ingest_documents",
            json={
                "documents": [
                    {
                        "doc_id": "doc-meta",
                        "text": "Contract with metadata.",
                        "metadata": {"source": "upload", "version": "1"},
                    }
                ]
            },
        )

        assert resp.status_code == 200
        call_args = mock_app.ingest_documents.call_args[0][0]
        assert call_args[0].metadata == {"source": "upload", "version": "1"}


# ---------------------------------------------------------------------------
# Helpers shared by query tests
# ---------------------------------------------------------------------------

def _make_generation(
    answer: str = "The notice period is 60 days.",
    pattern: QueryPattern = QueryPattern.B,
    findings: str | None = None,
    supporting_quotes: list[str] | None = None,
) -> GenerationResult:
    route = RouteResult(pattern=pattern, semantic_query="q", structured_targets=[])
    retrieval = RetrievalResult(route=route)
    return GenerationResult(
        answer=answer,
        pattern=pattern,
        findings=findings,
        supporting_quotes=supporting_quotes or [],
        retrieval=retrieval,
    )


def _mock_query_app(generation: GenerationResult) -> MagicMock:
    """Build a mock CogBaseApp whose query_stream yields tokens then a GenerationResult."""
    inst = MagicMock()
    inst.setup = AsyncMock()

    async def _query_stream(text: str):
        for token in generation.answer.split():
            yield token + " "
        yield generation

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
    with patch("api.routers.applications.build_app", return_value=mock_app):
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
        gen = _make_generation("The notice period is 60 days.")
        await _create_app(client, _mock_query_app(gen))

        resp = await client.post(
            "/applications/my-contract-analyzer/query",
            json={"text": "what is the notice period?"},
        )

        assert resp.status_code == 200
        assert resp.json()["answer"] == gen.answer

    @pytest.mark.asyncio
    async def test_returns_correct_pattern(self, client):
        gen = _make_generation(pattern=QueryPattern.C)
        await _create_app(client, _mock_query_app(gen))

        resp = await client.post(
            "/applications/my-contract-analyzer/query",
            json={"text": "q"},
        )

        assert resp.json()["pattern"] == "C"

    @pytest.mark.asyncio
    async def test_pattern_d_returns_findings_and_quotes(self, client):
        gen = _make_generation(
            answer="[FINDINGS]\nFound it.\n[SUPPORTING_QUOTES]\n- quote",
            pattern=QueryPattern.D,
            findings="Found it.",
            supporting_quotes=["quote"],
        )
        await _create_app(client, _mock_query_app(gen))

        resp = await client.post(
            "/applications/my-contract-analyzer/query",
            json={"text": "summarise"},
        )

        data = resp.json()
        assert data["findings"] == "Found it."
        assert data["supporting_quotes"] == ["quote"]

    @pytest.mark.asyncio
    async def test_404_when_app_not_found(self, client):
        resp = await client.post(
            "/applications/nonexistent/query",
            json={"text": "q"},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_404_when_app_not_active(self, client):
        failing = MagicMock()
        failing.setup = AsyncMock(side_effect=RuntimeError("setup boom"))
        with patch("api.routers.applications.build_app", return_value=failing):
            await client.post(
                "/applications",
                files={"bundle": ("bundle.zip", _VALID_BUNDLE, "application/zip")},
            )

        resp = await client.post(
            "/applications/my-contract-analyzer/query",
            json={"text": "q"},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_retries_on_first_failure(self, client):
        gen = _make_generation("Retry worked.")

        call_count = 0

        async def _flaky_stream(text: str):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient failure")
            yield gen.answer
            yield gen

        failing_then_good = _mock_query_app(gen)
        failing_then_good.query_stream = _flaky_stream

        await _create_app(client, failing_then_good)

        with patch("api.routers.applications.build_app", return_value=failing_then_good):
            resp = await client.post(
                "/applications/my-contract-analyzer/query",
                json={"text": "q"},
            )

        assert resp.status_code == 200
        assert resp.json()["answer"] == "Retry worked."


# ---------------------------------------------------------------------------
# POST /applications/{app_name}/query/stream
# ---------------------------------------------------------------------------

class TestQueryApplicationStream:
    @pytest.mark.asyncio
    async def test_returns_event_stream_content_type(self, client):
        gen = _make_generation()
        await _create_app(client, _mock_query_app(gen))

        resp = await client.post(
            "/applications/my-contract-analyzer/query/stream",
            json={"text": "q"},
        )

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    @pytest.mark.asyncio
    async def test_streams_token_events(self, client):
        gen = _make_generation("Hello world.")
        await _create_app(client, _mock_query_app(gen))

        resp = await client.post(
            "/applications/my-contract-analyzer/query/stream",
            json={"text": "q"},
        )

        events = _parse_sse(resp.text)
        token_events = [e for e in events if e != "[DONE]"]
        tokens = [json.loads(e) for e in token_events if "token" in json.loads(e)]
        assert len(tokens) > 0
        assembled = "".join(t["token"] for t in tokens)
        assert assembled.strip() == gen.answer

    @pytest.mark.asyncio
    async def test_final_result_event_contains_answer_and_pattern(self, client):
        gen = _make_generation("The answer.", pattern=QueryPattern.C)
        await _create_app(client, _mock_query_app(gen))

        resp = await client.post(
            "/applications/my-contract-analyzer/query/stream",
            json={"text": "q"},
        )

        events = _parse_sse(resp.text)
        result_events = [json.loads(e) for e in events if e != "[DONE]" and "result" in json.loads(e)]
        assert len(result_events) == 1
        result = result_events[0]["result"]
        assert result["answer"] == "The answer."
        assert result["pattern"] == "C"

    @pytest.mark.asyncio
    async def test_ends_with_done_sentinel(self, client):
        gen = _make_generation()
        await _create_app(client, _mock_query_app(gen))

        resp = await client.post(
            "/applications/my-contract-analyzer/query/stream",
            json={"text": "q"},
        )

        events = _parse_sse(resp.text)
        assert events[-1] == "[DONE]"

    @pytest.mark.asyncio
    async def test_pattern_d_result_includes_findings_and_quotes(self, client):
        gen = _make_generation(
            pattern=QueryPattern.D,
            findings="Key finding.",
            supporting_quotes=["verbatim excerpt"],
        )
        await _create_app(client, _mock_query_app(gen))

        resp = await client.post(
            "/applications/my-contract-analyzer/query/stream",
            json={"text": "q"},
        )

        events = _parse_sse(resp.text)
        result_events = [json.loads(e) for e in events if e != "[DONE]" and "result" in json.loads(e)]
        result = result_events[0]["result"]
        assert result["findings"] == "Key finding."
        assert result["supporting_quotes"] == ["verbatim excerpt"]

    @pytest.mark.asyncio
    async def test_404_when_app_not_found(self, client):
        resp = await client.post(
            "/applications/nonexistent/query/stream",
            json={"text": "q"},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_error_event_on_stream_failure(self, client):
        inst = MagicMock()
        inst.setup = AsyncMock()

        async def _failing_stream(text: str):
            raise RuntimeError("boom")
            yield  # make it an async generator

        inst.query_stream = _failing_stream
        await _create_app(client, inst)

        resp = await client.post(
            "/applications/my-contract-analyzer/query/stream",
            json={"text": "q"},
        )

        assert resp.status_code == 200
        events = _parse_sse(resp.text)
        error_events = [json.loads(e) for e in events if e != "[DONE]" and "error" in json.loads(e)]
        assert len(error_events) == 1
        assert events[-1] == "[DONE]"
