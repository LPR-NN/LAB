from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from source.constants import DEFAULT_TOKEN_BUDGET

if TYPE_CHECKING:
    from source.corpus.index import CorpusIndex

_corpus_context: ContextVar["CorpusIndex | None"] = ContextVar("corpus", default=None)


def set_corpus(corpus: "CorpusIndex") -> None:
    _corpus_context.set(corpus)


def get_corpus() -> "CorpusIndex":
    corpus = _corpus_context.get()
    if corpus is None:
        raise RuntimeError("Corpus not initialized")
    return corpus


# ============================================================================
# Search Cache - saves API calls by caching search results
# ============================================================================


@dataclass
class SearchCache:
    """
    LRU cache for search results.

    Stores results keyed by (query, doc_types, top_k) to avoid redundant searches.
    Also tracks similar queries to block near-duplicate searches.
    """

    max_size: int = 100
    _cache: dict[tuple[str, str | None, int], Any] = field(default_factory=dict)
    _query_history: list[str] = field(default_factory=list)
    _similar_threshold: float = 0.8

    def _normalize_query(self, query: str) -> str:
        """Normalize query for comparison (lowercase, sorted words)."""
        words = sorted(set(query.lower().split()))
        return " ".join(words)

    def _is_similar(self, query1: str, query2: str) -> bool:
        """Check if two queries are too similar (high word overlap)."""
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        if not words1 or not words2:
            return False
        overlap = len(words1 & words2) / max(len(words1), len(words2))
        return overlap >= self._similar_threshold

    def get(
        self,
        query: str,
        doc_types: str | None,
        top_k: int,
    ) -> Any | None:
        """Get cached result if exists."""
        key = (query, doc_types, top_k)
        return self._cache.get(key)

    def put(
        self,
        query: str,
        doc_types: str | None,
        top_k: int,
        result: Any,
    ) -> None:
        """Store result in cache."""
        # Evict oldest if at capacity
        if len(self._cache) >= self.max_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        key = (query, doc_types, top_k)
        self._cache[key] = result
        self._query_history.append(query)

    def find_similar_cached(
        self,
        query: str,
        doc_types: str | None,
        top_k: int,
    ) -> Any | None:
        """
        Find cached result from a similar query.

        Returns cached result if a previous query with high word overlap exists.
        """
        for cached_key, result in self._cache.items():
            cached_query, cached_types, cached_k = cached_key
            if cached_types == doc_types and cached_k >= top_k:
                if self._is_similar(query, cached_query):
                    return result
        return None

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._query_history.clear()

    def stats(self) -> dict[str, int]:
        """Get cache statistics."""
        return {
            "cached_queries": len(self._cache),
            "total_queries": len(self._query_history),
        }


_search_cache: ContextVar[SearchCache] = ContextVar(
    "search_cache", default=SearchCache()
)


def get_search_cache() -> SearchCache:
    """Get the search cache for current context."""
    return _search_cache.get()


def reset_search_cache() -> None:
    """Reset the search cache."""
    _search_cache.set(SearchCache())


@dataclass
class TokenBudget:
    limit: int = DEFAULT_TOKEN_BUDGET
    used: int = 0

    @staticmethod
    def estimate(text: str) -> int:
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        unicode_chars = len(text) - ascii_chars
        return (ascii_chars // 4) + (unicode_chars // 2)

    def consume(self, text: str) -> tuple[bool, int]:
        tokens = self.estimate(text)
        if self.used + tokens > self.limit:
            return False, tokens
        self.used += tokens
        return True, tokens

    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    def reset(self) -> None:
        self.used = 0


_token_budget: ContextVar[TokenBudget] = ContextVar(
    "token_budget", default=TokenBudget()
)


def get_budget() -> TokenBudget:
    return _token_budget.get()


def reset_budget(limit: int = DEFAULT_TOKEN_BUDGET) -> None:
    _token_budget.set(TokenBudget(limit=limit))
