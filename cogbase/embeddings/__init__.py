"""Chunk embedding backends for CogBase.

This module is shared by both the ingestion pipeline and the retrieval engine.
Implement ``EmbeddingBase`` to plug in a custom embedding backend.
"""

from cogbase.embeddings.base import EmbeddingBase, OpenAIEmbedding, SentenceTransformersEmbedding

__all__ = ["EmbeddingBase", "OpenAIEmbedding", "SentenceTransformersEmbedding"]
