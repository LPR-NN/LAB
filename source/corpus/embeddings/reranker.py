"""
Re-ranking module using cross-encoder models.

Cross-encoders are more accurate than bi-encoders (embeddings) for relevance scoring,
but slower. Used as a final step to re-rank top-k results from initial retrieval.

This is a **cost-effective** improvement:
- Runs locally (no API calls)
- Small models (~100MB) with good multilingual support
- Only processes top-k results (10-30 documents), not entire corpus
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from source.corpus.embeddings.vector_index import SearchResult


class ABCReranker(ABC):
    """Abstract base class for re-rankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """
        Re-rank search results by relevance to query.

        Args:
            query: Search query
            results: Initial search results to re-rank
            top_k: Return only top-k results (None = all)

        Returns:
            Re-ranked results sorted by relevance (highest first)
        """
        ...

    @abstractmethod
    def score(self, query: str, document: str) -> float:
        """
        Score a single query-document pair.

        Args:
            query: Search query
            document: Document text

        Returns:
            Relevance score (higher = more relevant)
        """
        ...


@dataclass
class CrossEncoderReranker(ABCReranker):
    """
    Re-ranker using cross-encoder models from sentence-transformers.

    Cross-encoders process query and document together, capturing their interaction.
    This is more accurate than bi-encoder similarity but slower.

    Recommended models for Russian/multilingual:
    - cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, English-focused)
    - BAAI/bge-reranker-base (good multilingual)
    - DiTy/cross-encoder-russian-msmarco (Russian-specific)

    Usage:
        reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-base")
        reranked = reranker.rerank(query, initial_results, top_k=5)
    """

    model_name: str = "BAAI/bge-reranker-base"
    device: str | None = None  # auto-detect: cuda if available, else cpu
    batch_size: int = 16

    _model: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Lazy initialization of the model."""
        pass  # Model loaded on first use

    def _load_model(self) -> None:
        """Load the cross-encoder model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for cross-encoder reranking. "
                "Install with: pip install sentence-transformers"
            ) from e

        self._model = CrossEncoder(
            self.model_name,
            device=self.device,
        )

    def score(self, query: str, document: str) -> float:
        """Score a single query-document pair."""
        self._load_model()
        scores = self._model.predict([(query, document)])
        return float(scores[0])

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """
        Re-rank search results using cross-encoder.

        Args:
            query: Search query
            results: Initial search results
            top_k: Return only top-k results

        Returns:
            Re-ranked results
        """
        if not results:
            return []

        self._load_model()

        # Prepare query-document pairs
        pairs = [(query, r.chunk.content) for r in results]

        # Get cross-encoder scores
        scores = self._model.predict(pairs, batch_size=self.batch_size)

        # Combine results with new scores
        scored_results = list(zip(results, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # Build re-ranked results with updated scores
        reranked: list[SearchResult] = []
        for result, new_score in scored_results:
            reranked.append(
                SearchResult(
                    chunk=result.chunk,
                    score=float(new_score),
                    document=result.document,
                )
            )

        if top_k is not None:
            return reranked[:top_k]
        return reranked


@dataclass
class BM25Reranker(ABCReranker):
    """
    Simple BM25-based re-ranker (no ML model required).

    Uses BM25 scoring to re-rank results. Useful when:
    - You don't want ML dependencies
    - You need fast re-ranking
    - Keyword matching is important

    Less accurate than cross-encoder but zero dependencies.
    """

    k1: float = 1.5
    b: float = 0.75

    def score(self, query: str, document: str) -> float:
        """Score using BM25-like term matching."""
        import re
        from collections import Counter

        # Tokenize
        query_tokens = set(re.findall(r"\b[\wа-яёА-ЯЁ]+\b", query.lower()))
        doc_tokens = re.findall(r"\b[\wа-яёА-ЯЁ]+\b", document.lower())

        if not query_tokens or not doc_tokens:
            return 0.0

        doc_freq = Counter(doc_tokens)
        doc_len = len(doc_tokens)

        score = 0.0
        for term in query_tokens:
            if term in doc_freq:
                tf = doc_freq[term]
                # Simplified BM25 (without IDF, as we don't have corpus stats)
                score += (tf * (self.k1 + 1)) / (tf + self.k1)

        # Normalize by query length
        return score / len(query_tokens)

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Re-rank using BM25 scoring."""
        if not results:
            return []

        scored = [(r, self.score(query, r.chunk.content)) for r in results]
        scored.sort(key=lambda x: x[1], reverse=True)

        reranked = [
            SearchResult(chunk=r.chunk, score=s, document=r.document) for r, s in scored
        ]

        if top_k is not None:
            return reranked[:top_k]
        return reranked


# Factory function
def create_reranker(
    model_name: str | None = None,
    use_cross_encoder: bool = True,
) -> ABCReranker:
    """
    Create a re-ranker instance.

    Args:
        model_name: Cross-encoder model name (if using cross-encoder)
        use_cross_encoder: If False, use BM25 re-ranker (no ML)

    Returns:
        Configured re-ranker
    """
    if not use_cross_encoder:
        return BM25Reranker()

    return CrossEncoderReranker(model_name=model_name or "BAAI/bge-reranker-base")


__all__ = [
    "ABCReranker",
    "CrossEncoderReranker",
    "BM25Reranker",
    "create_reranker",
]
