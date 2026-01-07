"""Hybrid search combining vector similarity and BM25 keyword matching."""

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from source.contracts.corpus import CorpusDocument
from source.corpus.embeddings.chunker import Chunk, DocumentChunker
from source.corpus.embeddings.providers import ABCEmbeddingProvider
from source.corpus.embeddings.vector_index import SearchResult, VectorIndex

# BM25 parameters
BM25_K1 = 1.5
BM25_B = 0.75


def tokenize_russian(text: str) -> list[str]:
    """Simple tokenization for Russian text."""
    # Extract words (both Cyrillic and Latin)
    tokens = re.findall(r"\b[\wа-яёА-ЯЁ]+\b", text.lower())
    # Filter short tokens
    return [t for t in tokens if len(t) > 2]


@dataclass
class BM25Index:
    """
    Simple BM25 index for keyword search.

    Used as part of hybrid search to combine with vector similarity.
    """

    _doc_freqs: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _doc_lengths: dict[str, int] = field(default_factory=dict)
    _doc_terms: dict[str, dict[str, int]] = field(default_factory=dict)
    _avg_doc_length: float = 0.0
    _num_docs: int = 0

    def add_document(self, doc_id: str, text: str) -> None:
        """Add a document to the BM25 index."""
        tokens = tokenize_russian(text)

        if not tokens:
            return

        # Count term frequencies
        term_freqs: dict[str, int] = defaultdict(int)
        for token in tokens:
            term_freqs[token] += 1

        # Store document info
        self._doc_terms[doc_id] = dict(term_freqs)
        self._doc_lengths[doc_id] = len(tokens)

        # Update document frequencies
        for term in term_freqs:
            self._doc_freqs[term] += 1

        # Update average document length
        self._num_docs += 1
        total_length = sum(self._doc_lengths.values())
        self._avg_doc_length = (
            total_length / self._num_docs if self._num_docs > 0 else 0
        )

    def score(self, doc_id: str, query: str) -> float:
        """
        Calculate BM25 score for a document given a query.

        Args:
            doc_id: Document ID
            query: Query text

        Returns:
            BM25 score (higher is better)
        """
        if doc_id not in self._doc_terms:
            return 0.0

        query_tokens = tokenize_russian(query)
        if not query_tokens:
            return 0.0

        doc_terms = self._doc_terms[doc_id]
        doc_length = self._doc_lengths[doc_id]

        score = 0.0

        for term in query_tokens:
            if term not in doc_terms:
                continue

            # Term frequency in document
            tf = doc_terms[term]

            # Document frequency
            df = self._doc_freqs.get(term, 0)

            # IDF component
            idf = math.log((self._num_docs - df + 0.5) / (df + 0.5) + 1)

            # BM25 score component
            numerator = tf * (BM25_K1 + 1)
            denominator = tf + BM25_K1 * (
                1 - BM25_B + BM25_B * (doc_length / self._avg_doc_length)
            )

            score += idf * (numerator / denominator)

        return score

    def search(
        self,
        query: str,
        doc_ids: list[str] | None = None,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """
        Search for documents matching the query.

        Args:
            query: Query text
            doc_ids: Optional list of doc IDs to search within
            top_k: Maximum number of results

        Returns:
            List of (doc_id, score) tuples sorted by score descending
        """
        candidates = doc_ids if doc_ids else list(self._doc_terms.keys())

        scores: list[tuple[str, float]] = []
        for doc_id in candidates:
            score = self.score(doc_id, query)
            if score > 0:
                scores.append((doc_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


@dataclass
class HybridIndex:
    """
    Hybrid search index combining vector similarity and BM25.

    The final score is computed as:
        score = alpha * vector_score + (1 - alpha) * bm25_score

    Where both scores are normalized to [0, 1] range.

    Features:
    - Semantic understanding from embeddings
    - Keyword matching from BM25
    - Tunable balance between the two
    - Reciprocal Rank Fusion (RRF) as alternative scoring
    """

    embedding_provider: ABCEmbeddingProvider
    chunker: DocumentChunker = field(default_factory=DocumentChunker)
    alpha: float = 0.7  # Weight for vector similarity (0.7 = 70% vector, 30% BM25)
    cache_dir: str | None = None
    use_rrf: bool = False  # Use Reciprocal Rank Fusion instead of weighted average

    _vector_index: VectorIndex | None = None
    _bm25_index: BM25Index = field(default_factory=BM25Index)
    _documents: dict[str, CorpusDocument] = field(default_factory=dict)
    _chunks: dict[str, Chunk] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize vector index."""
        self._vector_index = VectorIndex(
            embedding_provider=self.embedding_provider,
            chunker=self.chunker,
            cache_dir=self.cache_dir,
        )

    def add_document(self, document: CorpusDocument) -> int:
        """
        Add a document to both indices.

        Args:
            document: Document to add

        Returns:
            Number of chunks created
        """
        self._documents[document.doc_id] = document

        # Add to vector index
        assert self._vector_index is not None
        num_chunks = self._vector_index.add_document(document)

        # Add chunks to BM25 index
        chunks = self.chunker.chunk_document(document)
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk
            self._bm25_index.add_document(chunk.chunk_id, chunk.content)

        return num_chunks

    def add_documents(self, documents: list[CorpusDocument]) -> int:
        """Add multiple documents to the index."""
        # First add all documents to vector index in batch (efficient)
        assert self._vector_index is not None

        for doc in documents:
            self._documents[doc.doc_id] = doc

        total_chunks = self._vector_index.add_documents(documents)

        # Then add to BM25 index
        for doc in documents:
            chunks = self.chunker.chunk_document(doc)
            for chunk in chunks:
                self._chunks[chunk.chunk_id] = chunk
                self._bm25_index.add_document(chunk.chunk_id, chunk.content)

        return total_chunks

    def search(
        self,
        query: str,
        top_k: int = 10,
        doc_types: list[str] | None = None,
        min_priority: int | None = None,
        alpha: float | None = None,
    ) -> list[SearchResult]:
        """
        Hybrid search combining vector and BM25.

        Args:
            query: Search query
            top_k: Maximum number of results
            doc_types: Filter by document types
            min_priority: Minimum priority level
            alpha: Override default alpha (vector weight)

        Returns:
            List of SearchResult objects
        """
        if alpha is None:
            alpha = self.alpha

        # Get more results from each index to ensure good coverage
        fetch_k = top_k * 3

        # Vector search
        assert self._vector_index is not None
        vector_results = self._vector_index.search(
            query=query,
            top_k=fetch_k,
            doc_types=doc_types,
            min_priority=min_priority,
            include_documents=False,
        )

        # Get chunk IDs that match filters
        filtered_chunk_ids: list[str] | None = None
        if doc_types or min_priority is not None:
            filtered_chunk_ids = [r.chunk.chunk_id for r in vector_results]

        # BM25 search
        bm25_results = self._bm25_index.search(
            query=query,
            doc_ids=filtered_chunk_ids,
            top_k=fetch_k,
        )

        # Combine scores
        if self.use_rrf:
            combined = self._combine_rrf(vector_results, bm25_results)
        else:
            combined = self._combine_weighted(vector_results, bm25_results, alpha)

        # Sort by combined score and take top_k
        combined.sort(key=lambda x: x[1], reverse=True)

        results: list[SearchResult] = []

        for chunk_id, score in combined[: top_k * 2]:  # Get more to allow dedup
            chunk = self._chunks.get(chunk_id)
            if chunk is None:
                continue

            results.append(
                SearchResult(
                    chunk=chunk,
                    score=score,
                    document=self._documents.get(chunk.doc_id),
                )
            )

            if len(results) >= top_k:
                break

        return results

    def _combine_weighted(
        self,
        vector_results: list[SearchResult],
        bm25_results: list[tuple[str, float]],
        alpha: float,
    ) -> list[tuple[str, float]]:
        """Combine scores using weighted average."""
        scores: dict[str, dict[str, float]] = defaultdict(
            lambda: {"vector": 0.0, "bm25": 0.0}
        )

        # Normalize vector scores
        if vector_results:
            max_vector = max(r.score for r in vector_results)
            for r in vector_results:
                norm_score = r.score / max_vector if max_vector > 0 else 0
                scores[r.chunk.chunk_id]["vector"] = norm_score

        # Normalize BM25 scores
        if bm25_results:
            max_bm25 = max(s for _, s in bm25_results)
            for chunk_id, score in bm25_results:
                norm_score = score / max_bm25 if max_bm25 > 0 else 0
                scores[chunk_id]["bm25"] = norm_score

        # Compute combined scores
        combined: list[tuple[str, float]] = []
        for chunk_id, s in scores.items():
            final_score = alpha * s["vector"] + (1 - alpha) * s["bm25"]
            combined.append((chunk_id, final_score))

        return combined

    def _combine_rrf(
        self,
        vector_results: list[SearchResult],
        bm25_results: list[tuple[str, float]],
        k: int = 60,
    ) -> list[tuple[str, float]]:
        """
        Combine using Reciprocal Rank Fusion (RRF).

        RRF score = sum(1 / (k + rank_i)) for each ranking

        This is more robust to score distribution differences.
        """
        rrf_scores: dict[str, float] = defaultdict(float)

        # Add RRF scores from vector results
        for rank, r in enumerate(vector_results, start=1):
            rrf_scores[r.chunk.chunk_id] += 1 / (k + rank)

        # Add RRF scores from BM25 results
        for rank, (chunk_id, _) in enumerate(bm25_results, start=1):
            rrf_scores[chunk_id] += 1 / (k + rank)

        return list(rrf_scores.items())

    def get_document(self, doc_id: str) -> CorpusDocument | None:
        """Get a document by ID."""
        return self._documents.get(doc_id)

    def get_document_by_citation(self, citation_key: str) -> CorpusDocument | None:
        """Get a document by citation key."""
        for doc in self._documents.values():
            if doc.citation_key == citation_key:
                return doc
        return None

    def get_statistics(self) -> dict[str, Any]:
        """Get index statistics."""
        assert self._vector_index is not None
        vector_stats = self._vector_index.get_statistics()

        return {
            **vector_stats,
            "bm25_documents": self._bm25_index._num_docs,
            "bm25_unique_terms": len(self._bm25_index._doc_freqs),
            "hybrid_alpha": self.alpha,
            "use_rrf": self.use_rrf,
        }

    @classmethod
    def from_documents(
        cls,
        documents: list[CorpusDocument],
        embedding_provider: ABCEmbeddingProvider,
        chunker: DocumentChunker | None = None,
        alpha: float = 0.7,
        cache_dir: str | None = None,
        use_rrf: bool = False,
    ) -> "HybridIndex":
        """
        Create a HybridIndex from a list of documents.

        Args:
            documents: Documents to index
            embedding_provider: Embedding provider
            chunker: Document chunker
            alpha: Weight for vector similarity
            cache_dir: Directory for embedding cache
            use_rrf: Use Reciprocal Rank Fusion

        Returns:
            Initialized HybridIndex
        """
        index = cls(
            embedding_provider=embedding_provider,
            chunker=chunker or DocumentChunker(),
            alpha=alpha,
            cache_dir=cache_dir,
            use_rrf=use_rrf,
        )
        index.add_documents(documents)
        return index


__all__ = ["HybridIndex", "BM25Index"]
