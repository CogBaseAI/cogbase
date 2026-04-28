"""Unit tests for api/system_config.py."""

from __future__ import annotations

import textwrap
from unittest.mock import patch

import pytest

from api.system_config import SystemConfig


class TestSystemConfigFromYaml:
    def test_empty_yaml_gives_defaults(self):
        cfg = SystemConfig.from_yaml("")
        assert cfg.system_db.type == "sqlite"
        assert cfg.structured_store is None
        assert cfg.vector_store is None
        assert cfg.document_store is None

    def test_full_yaml(self):
        yaml_text = textwrap.dedent("""\
            system_db:
              type: sqlite
              path: ./data/system.db
            structured_store:
              type: sqlite
              path: ./data/app.db
            vector_store:
              type: faiss
              dim: 768
            document_store:
              type: local
              path: ./data/documents
        """)
        cfg = SystemConfig.from_yaml(yaml_text)
        assert cfg.system_db.type == "sqlite"
        assert cfg.system_db.path == "./data/system.db"
        assert cfg.structured_store.type == "sqlite"
        assert cfg.structured_store.path == "./data/app.db"
        assert cfg.vector_store.type == "faiss"
        assert cfg.vector_store.dim == 768
        assert cfg.document_store.type == "local"
        assert cfg.document_store.path == "./data/documents"

    def test_system_db_postgres(self):
        yaml_text = textwrap.dedent("""\
            system_db:
              type: postgres
              url: postgresql://user:pass@localhost/cogbase
        """)
        cfg = SystemConfig.from_yaml(yaml_text)
        assert cfg.system_db.type == "postgres"
        assert cfg.system_db.url == "postgresql://user:pass@localhost/cogbase"

    def test_system_db_memory(self):
        yaml_text = textwrap.dedent("""\
            system_db:
              type: memory
        """)
        cfg = SystemConfig.from_yaml(yaml_text)
        assert cfg.system_db.type == "memory"

    def test_structured_store_only(self):
        yaml_text = textwrap.dedent("""\
            structured_store:
              type: memory
        """)
        cfg = SystemConfig.from_yaml(yaml_text)
        assert cfg.structured_store.type == "memory"
        assert cfg.vector_store is None

    def test_non_mapping_raises(self):
        with pytest.raises(ValueError, match="mapping"):
            SystemConfig.from_yaml("- item1\n- item2\n")


class TestSystemConfigLoad:
    def test_load_defaults_when_no_file_no_env(self, monkeypatch):
        monkeypatch.delenv("COGBASE_CONFIG", raising=False)
        monkeypatch.delenv("COGBASE_SYSTEM_DB", raising=False)
        with patch("api.system_config.Path.exists", return_value=False):
            cfg = SystemConfig.load()
        assert cfg.system_db.type == "sqlite"
        assert cfg.system_db.path == "./cogbase_system.db"
        assert cfg.structured_store is None
        assert cfg.vector_store is None
        assert cfg.document_store is None

    def test_load_respects_cogbase_system_db_env_var(self, monkeypatch):
        monkeypatch.delenv("COGBASE_CONFIG", raising=False)
        monkeypatch.setenv("COGBASE_SYSTEM_DB", "/custom/path.db")
        with patch("api.system_config.Path.exists", return_value=False):
            cfg = SystemConfig.load()
        assert cfg.system_db.type == "sqlite"
        assert cfg.system_db.path == "/custom/path.db"

    def test_load_from_explicit_path(self, tmp_path):
        config_file = tmp_path / "system.yaml"
        config_file.write_text(textwrap.dedent("""\
            system_db:
              type: memory
            structured_store:
              type: memory
        """))
        cfg = SystemConfig.load(path=str(config_file))
        assert cfg.system_db.type == "memory"
        assert cfg.structured_store.type == "memory"

    def test_load_from_cogbase_config_env_var(self, tmp_path, monkeypatch):
        config_file = tmp_path / "system.yaml"
        config_file.write_text(textwrap.dedent("""\
            system_db:
              type: memory
        """))
        monkeypatch.setenv("COGBASE_CONFIG", str(config_file))
        cfg = SystemConfig.load()
        assert cfg.system_db.type == "memory"

    def test_load_from_default_yaml_path(self, monkeypatch):
        """When ./cogbase_system.yaml exists, load() reads it."""
        monkeypatch.delenv("COGBASE_CONFIG", raising=False)
        monkeypatch.delenv("COGBASE_SYSTEM_DB", raising=False)

        yaml_content = textwrap.dedent("""\
            system_db:
              type: memory
            structured_store:
              type: memory
        """)

        from pathlib import Path
        original_exists = Path.exists

        def fake_exists(self):
            # pathlib normalises "./foo.yaml" → "foo.yaml"
            if self.name == "cogbase_system.yaml":
                return True
            return original_exists(self)

        def fake_read_text(self, *args, **kwargs):
            return yaml_content

        with patch("api.system_config.Path.exists", fake_exists), \
             patch("api.system_config.Path.read_text", fake_read_text):
            cfg = SystemConfig.load()

        assert cfg.system_db.type == "memory"
        assert cfg.structured_store.type == "memory"

    def test_load_explicit_path_beats_env(self, tmp_path, monkeypatch):
        explicit = tmp_path / "explicit.yaml"
        explicit.write_text("system_db:\n  type: memory\n")

        env_file = tmp_path / "env.yaml"
        env_file.write_text("system_db:\n  type: sqlite\n  path: ./env.db\n")
        monkeypatch.setenv("COGBASE_CONFIG", str(env_file))

        cfg = SystemConfig.load(path=str(explicit))
        assert cfg.system_db.type == "memory"

    def test_system_db_injected_from_yaml_when_present(self, tmp_path):
        config_file = tmp_path / "cfg.yaml"
        config_file.write_text("system_db:\n  type: sqlite\n  path: ./custom.db\n")
        cfg = SystemConfig.load(path=str(config_file))
        assert cfg.system_db.path == "./custom.db"
