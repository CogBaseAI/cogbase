"""Chunk embedding backends for CogBase.

This module is shared by both the ingestion pipeline and the retrieval engine.
Implement ``EmbeddingBase`` to plug in a custom embedding backend.
"""

from cogbase.embeddings.base import EmbeddingBase
from cogbase.embeddings.huggingface import SentenceTransformersEmbedding
from cogbase.embeddings.openai import OpenAIEmbedding

__all__ = ["EmbeddingBase", "OpenAIEmbedding", "SentenceTransformersEmbedding"]
