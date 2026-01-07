"""Corpus management - document loading, indexing, and search."""

from source.corpus.agent_annotator import AgentAnnotator
from source.corpus.annotator import CorpusAnnotator
from source.corpus.context import TokenBudget, get_budget, reset_budget
from source.corpus.index import (
    CorpusIndex,
    create_tfidf_index,
    create_vector_index,
    create_hybrid_index,
)
from source.corpus.loader import DocumentLoader
from source.corpus.metadata import MetadataExtractor

__all__ = [
    "AgentAnnotator",
    "CorpusAnnotator",
    "CorpusIndex",
    "DocumentLoader",
    "MetadataExtractor",
    "TokenBudget",
    "get_budget",
    "reset_budget",
    # Factory functions
    "create_tfidf_index",
    "create_vector_index",
    "create_hybrid_index",
]
