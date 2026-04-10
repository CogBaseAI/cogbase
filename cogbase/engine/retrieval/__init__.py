"""Retrieval layer — executes RouteResult instructions against the stores."""

from cogbase.engine.retrieval.base import RetrievalResult, RetrieverBase
from cogbase.engine.retrieval.structured import StructuredRetriever
from cogbase.engine.retrieval.vector import VectorRetriever
from cogbase.engine.retrieval.hybrid import HybridRetriever

__all__ = [
    "RetrievalResult",
    "RetrieverBase",
    "StructuredRetriever",
    "VectorRetriever",
    "HybridRetriever",
]
