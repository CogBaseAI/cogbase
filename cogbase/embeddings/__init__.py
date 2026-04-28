"""Chunk embedding backends for CogBase.

This module is shared by both the ingestion pipeline and the retrieval engine.
Implement ``EmbeddingBase`` to plug in a custom embedding backend.
"""

from cogbase.embeddings.base import EmbeddingBase
from cogbase.embeddings.factory import build_embedding

__all__ = ["EmbeddingBase", "build_embedding"]
