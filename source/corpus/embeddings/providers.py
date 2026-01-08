"""Embedding providers abstraction for pluggable embedding models."""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import combinators

logger = logging.getLogger(__name__)
from combinators import lift as L
from combinators import rate_limit
from combinators.concurrency import RateLimitPolicy
from kungfu import Error, LazyCoroResult, Ok


class ABCEmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name for identification."""
        ...

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        ...

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query text.

        Some models have different embeddings for queries vs documents.
        Override this if your model requires it.

        Args:
            query: Query text to embed

        Returns:
            Embedding vector
        """
        return self.embed([query])[0]

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """
        Embed documents (passages).

        Some models have different embeddings for queries vs documents.
        Override this if your model requires it.

        Args:
            documents: Document texts to embed

        Returns:
            List of embedding vectors
        """
        return self.embed(documents)

    async def embed_documents_async(
        self,
        documents: list[str],
        concurrency: int = 5,
    ) -> list[list[float]]:
        """
        Embed documents asynchronously with parallel batch processing.

        Default implementation runs sync embed_documents in thread pool.
        Override for providers with native async support (e.g., OpenAI).

        Args:
            documents: Document texts to embed
            concurrency: Number of concurrent API calls (for batched providers)

        Returns:
            List of embedding vectors
        """
        return await asyncio.to_thread(self.embed_documents, documents)


@dataclass
class SentenceTransformerProvider(ABCEmbeddingProvider):
    """
    Embedding provider using sentence-transformers library.

    Supports any model from HuggingFace that works with sentence-transformers.

    Recommended models for Russian:
    - intfloat/multilingual-e5-large (best quality)
    - intfloat/multilingual-e5-base (good balance)
    - BAAI/bge-m3 (long context support)
    - sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (fast)

    E5 models require specific prefixes:
    - Query: "query: <text>"
    - Document: "passage: <text>"
    """

    name: str = "intfloat/multilingual-e5-base"
    device: str | None = None
    normalize: bool = True
    _model: Any = None
    _dimension: int | None = None

    def __post_init__(self) -> None:
        """Lazy initialization - model is loaded on first use."""
        pass

    def _load_model(self) -> None:
        """Load the model if not already loaded."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerProvider. "
                "Install it with: uv add sentence-transformers"
            ) from e

        logger.info(f"Loading embedding model: {self.name}...")
        self._model = SentenceTransformer(
            self.name,
            device=self.device,
        )
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded: dim={self._dimension}, device={self._model.device}")

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        self._load_model()
        assert self._dimension is not None
        return self._dimension

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self.name

    def _is_e5_model(self) -> bool:
        """Check if this is an E5 model requiring prefixes."""
        return "e5" in self.name.lower()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts without any prefix."""
        self._load_model()
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a query.

        For E5 models, adds "query: " prefix.
        """
        self._load_model()

        if self._is_e5_model():
            query = f"query: {query}"

        embeddings = self._model.encode(
            [query],
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return embeddings[0].tolist()

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """
        Embed documents.

        For E5 models, adds "passage: " prefix.
        """
        self._load_model()

        if self._is_e5_model():
            documents = [f"passage: {doc}" for doc in documents]

        logger.info(f"Embedding {len(documents)} chunks...")
        embeddings = self._model.encode(
            documents,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=len(documents) > 50,  # Show progress for large batches
        )
        logger.info("Embedding complete")
        return embeddings.tolist()


@dataclass
class OpenAIEmbeddingProvider(ABCEmbeddingProvider):
    """
    Embedding provider using OpenAI API.

    Models:
    - text-embedding-3-small (1536 dims, cheap)
    - text-embedding-3-large (3072 dims, best quality)
    - text-embedding-ada-002 (1536 dims, legacy)
    """

    name: str = "text-embedding-3-small"
    api_key: str | None = None
    _client: Any = None
    _dimensions: dict[str, int] = None  # type: ignore

    def __post_init__(self) -> None:
        self._dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai is required for OpenAIEmbeddingProvider. "
                "Install it with: uv add openai"
            ) from e

        self._client = OpenAI(api_key=self.api_key)
        return self._client

    @property
    def dimension(self) -> int:
        return self._dimensions.get(self.name, 1536)

    @property
    def model_name(self) -> str:
        return self.name

    def embed(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        response = client.embeddings.create(
            model=self.name,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def embed_query(self, query: str) -> list[float]:
        return self.embed([query])[0]

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        # OpenAI has a limit of 8191 tokens per request
        # Batch in groups of 100 to be safe
        all_embeddings: list[list[float]] = []
        batch_size = 100

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            embeddings = self.embed(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    def _make_embed_batch_interp(
        self, batch: list[str]
    ) -> LazyCoroResult[list[list[float]], str]:
        """Wrap embed() call as Interp for parallel processing."""
        return L.catching_async(
            lambda: asyncio.to_thread(self.embed, batch),
            on_error=str,
        )

    async def embed_documents_async(
        self,
        documents: list[str],
        concurrency: int = 5,
        rate_per_second: float = 50.0,
    ) -> list[list[float]]:
        """
        Embed documents with parallel API calls.

        Processes batches concurrently with rate limiting to avoid hitting
        OpenAI rate limits. Provides 2-5x speedup for large document sets.

        Args:
            documents: Document texts to embed
            concurrency: Number of concurrent API calls
            rate_per_second: Max API calls per second (OpenAI limits vary by tier)

        Returns:
            List of embedding vectors
        """
        if not documents:
            return []

        # Split into batches
        batch_size = 100
        batches = [
            documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
        ]

        if len(batches) == 1:
            # Single batch - no need for parallel processing
            return await asyncio.to_thread(self.embed, batches[0])

        # Rate limit policy for OpenAI API
        rate_policy = RateLimitPolicy(
            max_per_second=rate_per_second,
            burst=min(concurrency, 10),
        )

        def make_rate_limited_interp(
            batch: list[str],
        ) -> LazyCoroResult[list[list[float]], str]:
            return rate_limit(
                self._make_embed_batch_interp(batch),
                policy=rate_policy,
            )

        # Process batches in parallel with rate limiting
        batch_result = await combinators.batch_all(
            items=batches,
            handler=make_rate_limited_interp,
            concurrency=concurrency,
        )

        # Flatten results
        all_embeddings: list[list[float]] = []
        for result in batch_result.unwrap():
            match result:
                case Ok(embeddings):
                    all_embeddings.extend(embeddings)
                case Error(_):
                    pass

        return all_embeddings


__all__ = [
    "ABCEmbeddingProvider",
    "SentenceTransformerProvider",
    "OpenAIEmbeddingProvider",
]
