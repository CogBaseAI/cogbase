"""System-level shared resources — injected into each application at build time."""

from __future__ import annotations

from dataclasses import dataclass

from cogbase.embeddings.base import EmbeddingBase
from cogbase.llms.base import LLMBase
from cogbase.stores import DocumentStoreBase, StructuredStoreBase, VectorStoreBase


@dataclass
class SystemResources:
    """Shared runtime resources available to all applications as fallback defaults.

    Each field is optional.  When an application config declares its own
    ``llm``, ``embedding``, or store backend, that takes precedence.  When
    it omits one, the system-level resource is used instead.
    """

    structured_store: StructuredStoreBase | None = None
    vector_store: VectorStoreBase | None = None
    document_store: DocumentStoreBase | None = None
    llm: LLMBase | None = None
    embedder: EmbeddingBase | None = None
