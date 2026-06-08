"""System-level configuration — loaded once at service startup from a YAML file.

The system config defines service-wide defaults for the structured store and
vector store backends.  Applications posted to ``POST /applications`` only need
to declare their LLM, embedding, chunker, and pack settings; the store backends
are injected automatically from this config.

Configuration resolution order:
1. Path passed to ``SystemConfig.load(path=...)``
2. ``COGBASE_CONFIG`` environment variable
3. ``./cogbase_system.yaml`` if the file exists
4. Built-in defaults (in-memory stores; system DB from ``COGBASE_SYSTEM_DB``)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from cogbase.config.models import EmbeddingConfig, LLMConfig
from cogbase.config.stores import (
    DocumentStoreConfig,
    LogStoreConfig,
    StructuredStoreConfig,
    VectorStoreConfig,
)


class SystemConfig(BaseModel):
    """Top-level system configuration.

    Fields:
        system_db:        Store backend for the application registry (metadata
                          about all registered applications).  Supports the same
                          backends as application stores: ``sqlite``,
                          ``postgres``, or ``memory``.  Defaults to SQLite at
                          the path from ``COGBASE_SYSTEM_DB`` env var or
                          ``./cogbase_system.db``.
        structured_store: Shared backend for all application structured data.
                          When ``None``, each application falls back to an
                          isolated in-memory store (data lost on restart).
        vector_store:     Shared backend settings used when creating
                          per-application vector store instances.  When
                          ``None``, applications without an explicit
                          ``embedding`` + ``chunker`` config run in
                          structured-only mode.
        document_store:   Default document store for full document text.
                          Applications can override it with their own
                          ``document_store`` config.
        log_store:        Default append-only log store backing the episodic
                          memory NDJSON log.  When ``None``, episodic logging
                          runs without a durable backend.
    """

    system_db: StructuredStoreConfig = StructuredStoreConfig(
        type="sqlite", path="./cogbase_system.db"
    )
    structured_store: StructuredStoreConfig | None = None
    vector_store: VectorStoreConfig | None = None
    document_store: DocumentStoreConfig | None = None
    log_store: LogStoreConfig | None = None
    llm: LLMConfig | None = None
    embedding: EmbeddingConfig | None = None
    skills_dir: str | None = None  # directory containing <skill_name>/SKILL.md files

    @classmethod
    def from_yaml(cls, yaml_text: str) -> "SystemConfig":
        data: Any = yaml.safe_load(yaml_text) or {}
        if not isinstance(data, dict):
            raise ValueError("System config YAML must be a mapping at the top level")
        return cls.model_validate(data)

    @classmethod
    def load(cls, path: str | None = None) -> "SystemConfig":
        """Load system config from a file, env vars, and built-in defaults.

        Args:
            path: Explicit path to a system config YAML file.  When ``None``
                  the ``COGBASE_CONFIG`` env var and then
                  ``./cogbase_system.yaml`` are tried in order.

        Returns:
            A fully resolved ``SystemConfig`` instance.
        """
        config_path = path or os.environ.get("COGBASE_CONFIG")
        if config_path is None:
            for candidate in ("./cogbase_system.yaml", "./api/example_system_config.yaml"):
                if Path(candidate).exists():
                    config_path = candidate
                    break

        if config_path and Path(config_path).exists():
            raw: dict = yaml.safe_load(Path(config_path).read_text()) or {}
        else:
            raw = {}

        # When system_db is absent from the YAML, fall back to the
        # COGBASE_SYSTEM_DB env var (treated as a SQLite path) or the default.
        if "system_db" not in raw:
            db_path = os.environ.get("COGBASE_SYSTEM_DB", "./cogbase_system.db")
            raw["system_db"] = {"type": "sqlite", "path": db_path}

        return cls.model_validate(raw)
