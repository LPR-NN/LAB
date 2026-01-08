"""
Corpus index facade - unified interface for TF-IDF and vector search.

This module provides backward-compatible CorpusIndex that can use either:
- TF-IDF (legacy, fast, keyword-based)
- Vector search (semantic, requires embeddings)
- Hybrid search (best of both worlds)

Optional re-ranking support for improved relevance (no API cost).

Usage:
    # Legacy TF-IDF (default, no extra dependencies)
    index = CorpusIndex(documents)

    # Vector search
    from source.corpus.embeddings import SentenceTransformerProvider
    provider = SentenceTransformerProvider(name="intfloat/multilingual-e5-base")
    index = CorpusIndex(documents, embedding_provider=provider)

    # Hybrid search with re-ranking (best quality)
    from source.corpus.embeddings import create_reranker
    reranker = create_reranker(model_name="BAAI/bge-reranker-base")
    index = CorpusIndex(documents, embedding_provider=provider, mode="hybrid", reranker=reranker)
"""

from dataclasses import dataclass, field
from typing import Any, Literal

from source.contracts.corpus import CorpusDocument

# Import legacy TF-IDF index
from source.corpus.tfidf_index import CorpusIndex as TFIDFIndex


@dataclass
class CorpusIndex:
    """
    Unified corpus index supporting multiple search backends.

    Provides backward-compatible API while allowing upgrade to vector search.

    Modes:
    - "tfidf": Legacy TF-IDF search (default, no extra deps)
    - "vector": Pure vector similarity search
    - "hybrid": Combines vector + BM25 (recommended)

    Optional re-ranking for improved relevance (runs locally, no API cost).

    Args:
        documents: List of documents to index (optional, can add later)
        embedding_provider: Embedding provider for vector modes (optional)
        mode: Search mode - "tfidf", "vector", or "hybrid"
        alpha: Weight for vector similarity in hybrid mode (0-1)
        cache_dir: Directory for embedding cache (optional)
        use_rrf: Use Reciprocal Rank Fusion in hybrid mode
        reranker: Optional re-ranker for improved relevance (local cross-encoder)
    """

    documents: list[CorpusDocument] | None = None
    embedding_provider: Any | None = None  # ABCEmbeddingProvider
    mode: Literal["tfidf", "vector", "hybrid"] = "tfidf"
    alpha: float = 0.7
    cache_dir: str | None = None
    use_rrf: bool = False
    reranker: Any | None = None  # ABCReranker - optional re-ranker

    _tfidf_index: TFIDFIndex | None = None
    _vector_index: Any | None = None
    _hybrid_index: Any | None = None
    _documents: dict[str, CorpusDocument] = field(default_factory=dict)
    _by_citation: dict[str, CorpusDocument] = field(default_factory=dict)
    _by_type: dict[str, list[CorpusDocument]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize the appropriate index based on mode."""
        # Auto-detect mode if embedding provider is given
        if self.embedding_provider is not None and self.mode == "tfidf":
            self.mode = "hybrid"

        # Initialize index
        if self.mode == "tfidf":
            self._init_tfidf()
        elif self.mode == "vector":
            self._init_vector()
        else:
            self._init_hybrid()

        # Add initial documents in batch for efficiency
        if self.documents:
            self._add_documents_batch(self.documents)

    def _init_tfidf(self) -> None:
        """Initialize TF-IDF index."""
        self._tfidf_index = TFIDFIndex()

    def _init_vector(self) -> None:
        """Initialize vector index."""
        if self.embedding_provider is None:
            raise ValueError("embedding_provider is required for vector mode")

        from source.corpus.embeddings import DocumentChunker, VectorIndex

        self._vector_index = VectorIndex(
            embedding_provider=self.embedding_provider,
            chunker=DocumentChunker(),
            cache_dir=self.cache_dir,
        )

    def _init_hybrid(self) -> None:
        """Initialize hybrid index."""
        if self.embedding_provider is None:
            raise ValueError("embedding_provider is required for hybrid mode")

        from source.corpus.embeddings import DocumentChunker, HybridIndex

        self._hybrid_index = HybridIndex(
            embedding_provider=self.embedding_provider,
            chunker=DocumentChunker(),
            alpha=self.alpha,
            cache_dir=self.cache_dir,
            use_rrf=self.use_rrf,
        )

    def _add_documents_batch(self, documents: list[CorpusDocument]) -> None:
        """Add multiple documents efficiently using batch operations."""
        if not documents:
            return

        # Store in lookup dicts
        for doc in documents:
            self._documents[doc.doc_id] = doc
            self._by_citation[doc.citation_key] = doc
            if doc.doc_type not in self._by_type:
                self._by_type[doc.doc_type] = []
            self._by_type[doc.doc_type].append(doc)

        # Add to appropriate index in batch
        if self._tfidf_index is not None:
            for doc in documents:
                self._tfidf_index.add_document(doc)
            self._tfidf_index._rebuild_idf()
        elif self._vector_index is not None:
            self._vector_index.add_documents(documents)
        elif self._hybrid_index is not None:
            self._hybrid_index.add_documents(documents)

    def add_document(self, document: CorpusDocument) -> None:
        """Add a document to the index."""
        # Store in lookup dicts
        self._documents[document.doc_id] = document
        self._by_citation[document.citation_key] = document

        if document.doc_type not in self._by_type:
            self._by_type[document.doc_type] = []
        self._by_type[document.doc_type].append(document)

        # Add to appropriate index
        if self._tfidf_index is not None:
            self._tfidf_index.add_document(document)
            self._tfidf_index._rebuild_idf()
        elif self._vector_index is not None:
            self._vector_index.add_document(document)
        elif self._hybrid_index is not None:
            self._hybrid_index.add_document(document)

    def search(
        self,
        query: str,
        doc_types: list[str] | None = None,
        top_k: int = 10,
        min_priority: int | None = None,
    ) -> list[tuple[CorpusDocument, float]]:
        """
        Search for documents matching the query.

        This is the main search method, backward-compatible with TF-IDF API.

        Args:
            query: Search query
            doc_types: Filter by document types (optional)
            top_k: Maximum number of results
            min_priority: Minimum priority level (optional)

        Returns:
            List of (document, score) tuples, sorted by score descending
        """
        if self._tfidf_index is not None:
            return self._tfidf_index.search(
                query=query,
                doc_types=doc_types,
                top_k=top_k,
                min_priority=min_priority,
            )

        elif self._vector_index is not None:
            results = self._vector_index.search(
                query=query,
                doc_types=doc_types,
                top_k=top_k,
                min_priority=min_priority,
                include_documents=True,
            )
            return [(r.document, r.score) for r in results if r.document is not None]

        elif self._hybrid_index is not None:
            results = self._hybrid_index.search(
                query=query,
                doc_types=doc_types,
                top_k=top_k,
                min_priority=min_priority,
            )
            return [(r.document, r.score) for r in results if r.document is not None]

        return []

    def search_chunks(
        self,
        query: str,
        doc_types: list[str] | None = None,
        top_k: int = 10,
        min_priority: int | None = None,
    ) -> list[Any]:  # list[SearchResult]
        """
        Search and return chunks instead of full documents.

        Only available in vector/hybrid mode.
        If reranker is configured, results are re-ranked for better relevance.

        Returns:
            List of SearchResult objects with chunk info
        """
        # Fetch more results if we have a reranker (to rerank top candidates)
        fetch_k = top_k * 3 if self.reranker else top_k

        if self._vector_index is not None:
            results = self._vector_index.search(
                query=query,
                doc_types=doc_types,
                top_k=fetch_k,
                min_priority=min_priority,
            )
        elif self._hybrid_index is not None:
            results = self._hybrid_index.search(
                query=query,
                doc_types=doc_types,
                top_k=fetch_k,
                min_priority=min_priority,
            )
        else:
            # Fallback for TF-IDF: wrap documents as pseudo-chunks
            from source.corpus.embeddings.chunker import Chunk
            from source.corpus.embeddings.vector_index import SearchResult

            doc_results = self.search(query, doc_types, top_k, min_priority)
            results = [
                SearchResult(
                    chunk=Chunk(
                        chunk_id=f"{doc.doc_id}:0",
                        doc_id=doc.doc_id,
                        citation_key=doc.citation_key,
                        doc_type=doc.doc_type,
                        priority=doc.priority,
                        title=doc.metadata.title,
                        section=None,
                        content=doc.content,
                        start_char=0,
                        end_char=len(doc.content),
                    ),
                    score=score,
                    document=doc,
                )
                for doc, score in doc_results
            ]

        # Apply re-ranking if configured
        if self.reranker and results:
            results = self.reranker.rerank(query, results, top_k=top_k)

        return results[:top_k]

    def get_by_id(self, doc_id: str) -> CorpusDocument | None:
        """Get document by ID."""
        return self._documents.get(doc_id)

    def get_by_citation(self, citation_key: str) -> CorpusDocument | None:
        """Get document by citation key."""
        return self._by_citation.get(citation_key)

    def get_by_type(self, doc_type: str) -> list[CorpusDocument]:
        """Get all documents of a specific type."""
        return self._by_type.get(doc_type, [])

    def get_by_priority(self, min_priority: int) -> list[CorpusDocument]:
        """Get all documents with at least the specified priority."""
        return [doc for doc in self._documents.values() if doc.priority >= min_priority]

    def get_active_documents(self) -> list[CorpusDocument]:
        """Get all active (non-superseded) documents."""
        return [doc for doc in self._documents.values() if doc.is_active]

    def all_documents(self) -> list[CorpusDocument]:
        """Get all documents in the index."""
        return list(self._documents.values())

    def find_similar(
        self,
        document: CorpusDocument,
        top_k: int = 5,
        exclude_self: bool = True,
    ) -> list[tuple[CorpusDocument, float]]:
        """
        Find documents similar to the given document.

        In vector/hybrid mode, this is much more accurate.

        Args:
            document: Reference document
            top_k: Number of similar documents to return
            exclude_self: Whether to exclude the document itself

        Returns:
            List of (document, similarity_score) tuples
        """
        # Use document content as query
        query = f"{document.metadata.title} {document.content[:1000]}"
        results = self.search(query, top_k=top_k + (1 if exclude_self else 0))

        if exclude_self:
            results = [
                (doc, score) for doc, score in results if doc.doc_id != document.doc_id
            ]

        return results[:top_k]

    def get_statistics(self) -> dict[str, Any]:
        """Get index statistics."""
        base_stats = {
            "total_documents": len(self._documents),
            "documents_by_type": {
                doc_type: len(docs) for doc_type, docs in self._by_type.items()
            },
            "active_documents": len(self.get_active_documents()),
            "mode": self.mode,
        }

        if self._tfidf_index is not None:
            tfidf_stats = self._tfidf_index.get_statistics()
            base_stats["unique_terms"] = tfidf_stats.get("unique_terms", 0)

        elif self._vector_index is not None:
            vector_stats = self._vector_index.get_statistics()
            base_stats.update(vector_stats)

        elif self._hybrid_index is not None:
            hybrid_stats = self._hybrid_index.get_statistics()
            base_stats.update(hybrid_stats)

        return base_stats


# Factory functions for convenience


def create_tfidf_index(documents: list[CorpusDocument]) -> CorpusIndex:
    """Create a TF-IDF based index (fast, no dependencies)."""
    return CorpusIndex(documents=documents, mode="tfidf")


def create_vector_index(
    documents: list[CorpusDocument],
    model_name: str = "intfloat/multilingual-e5-base",
    cache_dir: str | None = None,
) -> CorpusIndex:
    """
    Create a vector-based index (semantic search).

    Requires: sentence-transformers
    """
    from source.corpus.embeddings import SentenceTransformerProvider

    provider = SentenceTransformerProvider(name=model_name)
    return CorpusIndex(
        documents=documents,
        embedding_provider=provider,
        mode="vector",
        cache_dir=cache_dir,
    )


def create_hybrid_index(
    documents: list[CorpusDocument],
    model_name: str = "intfloat/multilingual-e5-base",
    alpha: float = 0.7,
    cache_dir: str | None = None,
    use_rrf: bool = False,
    reranker_model: str | None = None,
) -> CorpusIndex:
    """
    Create a hybrid index (vector + BM25).

    This is the recommended mode for best results.

    Args:
        documents: Documents to index
        model_name: Sentence transformer model name
        alpha: Weight for vector similarity (0-1, higher = more vector)
        cache_dir: Directory for embedding cache
        use_rrf: Use Reciprocal Rank Fusion instead of weighted average
        reranker_model: Optional cross-encoder model for re-ranking (improves quality, local)

    Requires: sentence-transformers
    """
    from source.corpus.embeddings import SentenceTransformerProvider

    provider = SentenceTransformerProvider(name=model_name)

    # Create reranker if model specified
    reranker = None
    if reranker_model:
        from source.corpus.embeddings import create_reranker

        reranker = create_reranker(model_name=reranker_model)

    return CorpusIndex(
        documents=documents,
        embedding_provider=provider,
        mode="hybrid",
        alpha=alpha,
        cache_dir=cache_dir,
        use_rrf=use_rrf,
        reranker=reranker,
    )


__all__ = [
    "CorpusIndex",
    "create_tfidf_index",
    "create_vector_index",
    "create_hybrid_index",
]
