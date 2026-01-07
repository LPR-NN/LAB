"""Corpus index for document search and retrieval."""

import math
import re
from collections import defaultdict
from dataclasses import dataclass

from source.contracts.corpus import CorpusDocument

TITLE_BOOST = 3.0
TAGS_BOOST = 2.0
CONTENT_BOOST = 1.0


RUSSIAN_SUFFIXES = (
    "ого",
    "его",
    "ому",
    "ему",
    "ым",
    "им",
    "ой",
    "ей",
    "ую",
    "юю",
    "ая",
    "яя",
    "ое",
    "ее",
    "ые",
    "ие",
    "ых",
    "их",
    "ам",
    "ям",
    "ами",
    "ями",
    "ах",
    "ях",
    "ов",
    "ев",
    "ей",
    "ин",
    "ина",
    "ину",
    "ином",
    "иной",
    "ины",
    "иных",
    "ия",
    "ие",
    "ию",
    "ией",
    "ии",
    "ий",
    "ия",
)


def stem_russian(word: str) -> str:
    """Simple Russian stemmer — removes common suffixes."""
    if len(word) <= 4:
        return word
    for suffix in RUSSIAN_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def tokenize(text: str) -> list[str]:
    """Tokenize and stem text."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [stem_russian(t) for t in tokens if len(t) > 2]


@dataclass
class FieldWeights:
    """Weighted term frequencies from different fields."""

    title: dict[str, float]
    tags: dict[str, float]
    content: dict[str, float]

    def get_boosted_tf(self, term: str) -> float:
        """Get boosted TF for a term across all fields."""
        return (
            self.title.get(term, 0.0) * TITLE_BOOST
            + self.tags.get(term, 0.0) * TAGS_BOOST
            + self.content.get(term, 0.0) * CONTENT_BOOST
        )

    def all_terms(self) -> set[str]:
        """Get all unique terms across fields."""
        return set(self.title) | set(self.tags) | set(self.content)


def compute_tf(tokens: list[str]) -> dict[str, float]:
    """Compute term frequency."""
    tf: dict[str, int] = defaultdict(int)
    for token in tokens:
        tf[token] += 1

    total = len(tokens) or 1
    return {term: count / total for term, count in tf.items()}


class CorpusIndex:
    """
    Index for searching and retrieving corpus documents.

    Implements TF-IDF based search with field boosting (title, tags, content).
    """

    def __init__(self, documents: list[CorpusDocument] | None = None):
        self._documents: dict[str, CorpusDocument] = {}
        self._by_citation: dict[str, CorpusDocument] = {}
        self._by_type: dict[str, list[CorpusDocument]] = defaultdict(list)

        self._doc_fields: dict[str, FieldWeights] = {}
        self._idf: dict[str, float] = {}
        self._doc_count = 0

        if documents:
            for doc in documents:
                self.add_document(doc)
            self._rebuild_idf()

    def add_document(self, document: CorpusDocument) -> None:
        """Add a document to the index."""
        doc_id = document.doc_id

        self._documents[doc_id] = document
        self._by_citation[document.citation_key] = document
        self._by_type[document.doc_type].append(document)

        title_tokens = tokenize(document.metadata.title)
        tags_text = " ".join(document.metadata.tags) if document.metadata.tags else ""
        tags_tokens = tokenize(tags_text)
        content_tokens = tokenize(document.content)

        self._doc_fields[doc_id] = FieldWeights(
            title=compute_tf(title_tokens),
            tags=compute_tf(tags_tokens),
            content=compute_tf(content_tokens),
        )

        self._doc_count += 1

    def _rebuild_idf(self) -> None:
        """Rebuild IDF scores after adding documents."""
        doc_freq: dict[str, int] = defaultdict(int)

        for fields in self._doc_fields.values():
            for term in fields.all_terms():
                doc_freq[term] += 1

        self._idf = {
            term: math.log(self._doc_count / (df + 1)) + 1
            for term, df in doc_freq.items()
        }

    def _compute_tfidf_score(self, doc_id: str, query_tokens: list[str]) -> float:
        """Compute TF-IDF score with field boosting."""
        if doc_id not in self._doc_fields:
            return 0.0

        fields = self._doc_fields[doc_id]
        score = 0.0

        for token in query_tokens:
            boosted_tf = fields.get_boosted_tf(token)
            if boosted_tf > 0:
                score += boosted_tf * self._idf.get(token, 1.0)

        return score

    def search(
        self,
        query: str,
        doc_types: list[str] | None = None,
        top_k: int = 10,
        min_priority: int | None = None,
    ) -> list[tuple[CorpusDocument, float]]:
        """
        Search for documents matching the query.

        Args:
            query: Search query
            doc_types: Filter by document types (optional)
            top_k: Maximum number of results
            min_priority: Minimum priority level (optional)

        Returns:
            List of (document, score) tuples, sorted by score descending
        """
        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        # Filter documents
        candidates: list[str]
        if doc_types:
            candidates = [
                doc.doc_id
                for doc_type in doc_types
                for doc in self._by_type.get(doc_type, [])
            ]
        else:
            candidates = list(self._documents.keys())

        # Apply priority filter
        if min_priority is not None:
            candidates = [
                doc_id
                for doc_id in candidates
                if self._documents[doc_id].priority >= min_priority
            ]

        # Score documents
        scores: list[tuple[str, float]] = []
        for doc_id in candidates:
            score = self._compute_tfidf_score(doc_id, query_tokens)
            if score > 0:
                scores.append((doc_id, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Return top_k with documents
        results: list[tuple[CorpusDocument, float]] = []
        for doc_id, score in scores[:top_k]:
            results.append((self._documents[doc_id], score))

        return results

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

        Args:
            document: Reference document
            top_k: Number of similar documents to return
            exclude_self: Whether to exclude the document itself

        Returns:
            List of (document, similarity_score) tuples
        """
        # Use document content as query
        query = f"{document.metadata.title} {document.content}"
        results = self.search(query, top_k=top_k + (1 if exclude_self else 0))

        if exclude_self:
            results = [
                (doc, score) for doc, score in results if doc.doc_id != document.doc_id
            ]

        return results[:top_k]

    def get_statistics(self) -> dict[str, int | dict[str, int]]:
        """Get index statistics."""
        type_counts: dict[str, int] = {
            doc_type: len(docs) for doc_type, docs in self._by_type.items()
        }

        return {
            "total_documents": len(self._documents),
            "unique_terms": len(self._idf),
            "documents_by_type": type_counts,
            "active_documents": len(self.get_active_documents()),
        }


__all__ = ["CorpusIndex"]
