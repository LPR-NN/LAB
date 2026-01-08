"""Vector index using numpy for semantic search.

Simple in-memory vector store without external database dependencies.
For small-to-medium corpora (<50k chunks), this is faster and simpler
than ChromaDB or pgvector.
"""

import hashlib
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from source.contracts.corpus import CorpusDocument
from source.corpus.embeddings.chunker import Chunk, DocumentChunker
from source.corpus.embeddings.providers import ABCEmbeddingProvider


@dataclass
class SearchResult:
    """Result from vector search."""

    chunk: Chunk
    score: float
    document: CorpusDocument | None = None


@dataclass
class VectorIndex:
    """
    In-memory vector index for semantic document search.

    Uses numpy for cosine similarity search. No external database required.

    Features:
    - Fast cosine similarity search
    - Metadata filtering (doc_type, priority)
    - Automatic document chunking
    - Optional pickle cache for embeddings
    """

    embedding_provider: ABCEmbeddingProvider
    chunker: DocumentChunker = field(default_factory=DocumentChunker)
    cache_dir: str | None = None

    _documents: dict[str, CorpusDocument] = field(default_factory=dict)
    _chunks: list[Chunk] = field(default_factory=list)
    _embeddings: NDArray[np.float32] | None = field(default=None, repr=False)
    _chunk_id_to_idx: dict[str, int] = field(default_factory=dict)
    _cache_loaded: bool = field(default=False, repr=False)

    def _get_cache_path(self) -> Path | None:
        """Get path for embedding cache file."""
        if not self.cache_dir:
            return None

        # Create cache key from model name
        model_hash = hashlib.md5(
            self.embedding_provider.model_name.encode()
        ).hexdigest()[:8]
        return Path(self.cache_dir) / f"embeddings_{model_hash}.pkl"

    def _load_cache(self) -> bool:
        """Try to load embeddings from cache. Returns True if successful."""
        import logging

        logger = logging.getLogger(__name__)
        cache_path = self._get_cache_path()
        if not cache_path or not cache_path.exists():
            logger.info(f"No embedding cache found at {cache_path}")
            return False

        try:
            logger.info(f"Loading embeddings from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                cached: dict[str, Any] = pickle.load(f)

            # Verify cache is for same model
            if cached.get("model") != self.embedding_provider.model_name:
                logger.warning(
                    f"Cache model mismatch: {cached.get('model')} != {self.embedding_provider.model_name}"
                )
                return False

            self._embeddings = cached["embeddings"]
            self._chunks = cached["chunks"]
            self._chunk_id_to_idx = {c.chunk_id: i for i, c in enumerate(self._chunks)}

            logger.info(f"Loaded {len(self._chunks)} chunks from cache")
            return True
        except Exception as e:
            logger.warning(f"Failed to load embedding cache: {e}")
            return False

    def _save_cache(self) -> None:
        """Save embeddings to cache."""
        import logging

        logger = logging.getLogger(__name__)
        cache_path = self._get_cache_path()
        if not cache_path:
            return

        cache_path.parent.mkdir(parents=True, exist_ok=True)

        cached = {
            "model": self.embedding_provider.model_name,
            "embeddings": self._embeddings,
            "chunks": self._chunks,
        }

        with open(cache_path, "wb") as f:
            pickle.dump(cached, f)

        logger.info(
            f"Saved {len(self._chunks)} chunk embeddings to cache: {cache_path}"
        )

    def add_document(self, document: CorpusDocument) -> int:
        """
        Add a document to the index.

        Document is chunked and each chunk is embedded.

        Args:
            document: Document to add

        Returns:
            Number of chunks created
        """
        self._documents[document.doc_id] = document

        chunks = self.chunker.chunk_document(document)
        if not chunks:
            return 0

        # Get embeddings for new chunks
        texts = [c.content for c in chunks]
        new_embeddings: NDArray[np.float32] = np.array(
            self.embedding_provider.embed_documents(texts),
            dtype=np.float32,
        )

        # Normalize for cosine similarity
        norms: NDArray[np.float32] = np.linalg.norm(
            new_embeddings, axis=1, keepdims=True
        )
        new_embeddings = new_embeddings / np.maximum(norms, 1e-10)

        # Add to index
        start_idx = len(self._chunks)
        for i, chunk in enumerate(chunks):
            self._chunk_id_to_idx[chunk.chunk_id] = start_idx + i
            self._chunks.append(chunk)

        if self._embeddings is None:
            self._embeddings = new_embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, new_embeddings])

        return len(chunks)

    def add_documents(self, documents: list[CorpusDocument]) -> int:
        """
        Add multiple documents to the index.

        Batches embedding computation for efficiency.
        Loads from cache if available.

        Args:
            documents: List of documents to add

        Returns:
            Total number of chunks created
        """
        # Try to load from cache on first call
        if not self._cache_loaded and self.cache_dir:
            self._cache_loaded = True
            if self._load_cache():
                # Cache loaded - restore documents dict
                for doc in documents:
                    self._documents[doc.doc_id] = doc
                return len(self._chunks)

        # Chunk all documents first
        all_chunks: list[Chunk] = []
        for doc in documents:
            self._documents[doc.doc_id] = doc
            chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(chunks)

        if not all_chunks:
            return 0

        # Batch embed all chunks
        texts = [c.content for c in all_chunks]
        embeddings: NDArray[np.float32] = np.array(
            self.embedding_provider.embed_documents(texts),
            dtype=np.float32,
        )

        # Normalize
        norms: NDArray[np.float32] = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-10)

        # Add to index
        start_idx = len(self._chunks)
        for i, chunk in enumerate(all_chunks):
            self._chunk_id_to_idx[chunk.chunk_id] = start_idx + i
            self._chunks.append(chunk)

        if self._embeddings is None:
            self._embeddings = embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, embeddings])

        # Save cache if configured
        if self.cache_dir:
            self._save_cache()

        return len(all_chunks)

    def search(
        self,
        query: str,
        top_k: int = 10,
        doc_types: list[str] | None = None,
        min_priority: int | None = None,
        include_documents: bool = False,
    ) -> list[SearchResult]:
        """
        Search for chunks matching the query.

        Args:
            query: Search query
            top_k: Maximum number of results
            doc_types: Filter by document types (optional)
            min_priority: Minimum priority level (optional)
            include_documents: Include full document in results

        Returns:
            List of SearchResult objects sorted by relevance
        """
        if self._embeddings is None or len(self._chunks) == 0:
            return []

        # Embed and normalize query
        query_vec: NDArray[np.float32] = np.array(
            self.embedding_provider.embed_query(query),
            dtype=np.float32,
        )
        query_norm = np.linalg.norm(query_vec)
        query_vec = query_vec / max(query_norm, 1e-10)

        # Cosine similarity (embeddings are already normalized)
        similarities: NDArray[np.float32] = self._embeddings @ query_vec

        # Apply filters
        mask: NDArray[np.bool_] = np.ones(len(self._chunks), dtype=np.bool_)

        if doc_types:
            doc_type_set = set(doc_types)
            mask &= np.array([c.doc_type in doc_type_set for c in self._chunks])

        if min_priority is not None:
            mask &= np.array([c.priority >= min_priority for c in self._chunks])

        # Apply mask to similarities
        filtered_similarities: NDArray[np.float32] = np.where(
            mask, similarities, np.float32(-np.inf)
        )

        # Get top-k indices
        if top_k >= len(self._chunks):
            top_indices: NDArray[np.intp] = np.argsort(filtered_similarities)[::-1]
        else:
            # Use argpartition for efficiency when k << n
            top_indices = np.argpartition(filtered_similarities, -top_k)[-top_k:]
            top_indices = top_indices[
                np.argsort(filtered_similarities[top_indices])[::-1]
            ]

        # Build results
        results: list[SearchResult] = []
        for idx in top_indices:
            if filtered_similarities[idx] == -np.inf:
                break

            chunk = self._chunks[idx]
            document = None
            if include_documents:
                document = self._documents.get(chunk.doc_id)

            results.append(
                SearchResult(
                    chunk=chunk,
                    score=float(filtered_similarities[idx]),
                    document=document,
                )
            )

        return results

    def search_by_document(
        self,
        doc_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        Search within a specific document.

        Args:
            doc_id: Document ID to search within
            query: Search query
            top_k: Maximum number of results

        Returns:
            List of SearchResult objects
        """
        if self._embeddings is None or len(self._chunks) == 0:
            return []

        # Embed query
        query_vec: NDArray[np.float32] = np.array(
            self.embedding_provider.embed_query(query),
            dtype=np.float32,
        )
        query_norm = np.linalg.norm(query_vec)
        query_vec = query_vec / max(query_norm, 1e-10)

        # Similarities
        similarities: NDArray[np.float32] = self._embeddings @ query_vec

        # Filter by doc_id
        mask: NDArray[np.bool_] = np.array([c.doc_id == doc_id for c in self._chunks])
        filtered_similarities: NDArray[np.float32] = np.where(
            mask, similarities, np.float32(-np.inf)
        )

        # Top-k
        top_indices: NDArray[np.intp] = np.argsort(filtered_similarities)[::-1][:top_k]

        results: list[SearchResult] = []
        for idx in top_indices:
            if filtered_similarities[idx] == -np.inf:
                break

            chunk = self._chunks[idx]
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=float(filtered_similarities[idx]),
                    document=self._documents.get(chunk.doc_id),
                )
            )

        return results

    def get_document(self, doc_id: str) -> CorpusDocument | None:
        """Get a document by ID."""
        return self._documents.get(doc_id)

    def get_document_by_citation(self, citation_key: str) -> CorpusDocument | None:
        """Get a document by citation key."""
        for doc in self._documents.values():
            if doc.citation_key == citation_key:
                return doc
        return None

    def get_chunks_for_document(self, doc_id: str) -> list[Chunk]:
        """Get all chunks for a document."""
        return [chunk for chunk in self._chunks if chunk.doc_id == doc_id]

    def get_statistics(self) -> dict[str, Any]:
        """Get index statistics."""
        type_counts: dict[str, int] = {}
        for doc in self._documents.values():
            doc_type = doc.doc_type
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1

        return {
            "total_documents": len(self._documents),
            "total_chunks": len(self._chunks),
            "documents_by_type": type_counts,
            "embedding_model": self.embedding_provider.model_name,
            "embedding_dimension": self.embedding_provider.dimension,
        }

    def clear(self) -> None:
        """Clear all documents from the index."""
        self._documents.clear()
        self._chunks.clear()
        self._embeddings = None
        self._chunk_id_to_idx.clear()

    @classmethod
    def from_documents(
        cls,
        documents: list[CorpusDocument],
        embedding_provider: ABCEmbeddingProvider,
        chunker: DocumentChunker | None = None,
        cache_dir: str | None = None,
    ) -> "VectorIndex":
        """
        Create a VectorIndex from a list of documents.

        Args:
            documents: Documents to index
            embedding_provider: Embedding provider to use
            chunker: Document chunker (optional, uses defaults)
            cache_dir: Directory for embedding cache (optional)

        Returns:
            Initialized VectorIndex with all documents indexed
        """
        index = cls(
            embedding_provider=embedding_provider,
            chunker=chunker or DocumentChunker(),
            cache_dir=cache_dir,
        )
        index.add_documents(documents)
        return index


__all__ = ["VectorIndex", "SearchResult"]
