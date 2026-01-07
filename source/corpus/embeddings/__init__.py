"""Embeddings subsystem for semantic search."""

from source.corpus.embeddings.chunker import Chunk, DocumentChunker
from source.corpus.embeddings.hybrid import BM25Index, HybridIndex
from source.corpus.embeddings.providers import (
    ABCEmbeddingProvider,
    OpenAIEmbeddingProvider,
    SentenceTransformerProvider,
)
from source.corpus.embeddings.reranker import (
    ABCReranker,
    BM25Reranker,
    CrossEncoderReranker,
    create_reranker,
)
from source.corpus.embeddings.vector_index import SearchResult, VectorIndex

__all__ = [
    # Providers
    "ABCEmbeddingProvider",
    "SentenceTransformerProvider",
    "OpenAIEmbeddingProvider",
    # Chunking
    "DocumentChunker",
    "Chunk",
    # Indices
    "VectorIndex",
    "HybridIndex",
    "BM25Index",
    "SearchResult",
    # Re-ranking
    "ABCReranker",
    "CrossEncoderReranker",
    "BM25Reranker",
    "create_reranker",
]
