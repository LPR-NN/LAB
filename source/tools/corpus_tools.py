"""Tools for searching and citing the normative corpus."""

import re

from funcai import tool
from funcai.agents.tool import Tool
from source.contracts.tool_results import (
    BudgetExceeded,
    CitationError,
    CitationResult,
    DocumentResult,
    SearchResult,
)
from source.corpus.context import get_budget, get_corpus, get_search_cache


def _execute_search(
    query: str,
    doc_types: str | None,
    top_k: int,
) -> list[SearchResult]:
    """Execute the actual search (internal, no caching)."""
    types_list: list[str] | None = None
    if doc_types:
        types_list = [t.strip() for t in doc_types.split(",")]

    corpus = get_corpus()
    chunk_results = corpus.search_chunks(query, doc_types=types_list, top_k=top_k)

    if not chunk_results:
        return []

    max_score = max(r.score for r in chunk_results)
    normalizer = max_score if max_score > 0 else 1.0

    MAX_PREVIEW = 1000

    return [
        SearchResult(
            doc_id=r.chunk.doc_id,
            title=r.chunk.title or "",
            doc_type=r.chunk.doc_type,
            citation_key=r.chunk.citation_key,
            priority=r.chunk.priority,
            relevance_score=round(min(r.score / normalizer, 1.0), 4),
            preview=r.chunk.content[:MAX_PREVIEW]
            + ("..." if len(r.chunk.content) > MAX_PREVIEW else ""),
            section=r.chunk.section,
        )
        for r in chunk_results
    ]


@tool("Search the normative corpus for documents matching the query")
def search_corpus(
    query: str,
    doc_types: str | None = None,
    top_k: int = 5,
) -> list[SearchResult]:
    """
    Search the normative corpus for relevant documents using semantic search.

    The search uses embeddings to understand meaning, not just keyword matching.
    You can write natural language questions or describe what you're looking for.

    Returns relevant excerpts (chunks ~1500 chars) from documents, not just previews.

    Results are cached - similar queries return cached results to save API calls.

    Args:
        query: Search query - natural language question or description.
               Good: "правила исключения членов за нарушение этики"
               Good: "какие санкции применяются за оскорбление"
               Bad: "исключение член санкция" (keywords work but full sentences better)
        doc_types: Comma-separated document types to filter (e.g., "charter,regulations")
        top_k: Maximum number of results to return

    Returns:
        List of relevant excerpts with document info and relevance scores
    """
    cache = get_search_cache()

    # Check exact cache hit
    cached = cache.get(query, doc_types, top_k)
    if cached is not None:
        return cached

    # Check for similar query (blocks near-duplicate searches)
    similar_cached = cache.find_similar_cached(query, doc_types, top_k)
    if similar_cached is not None:
        return similar_cached

    # Execute actual search
    results = _execute_search(query, doc_types, top_k)

    # Cache the results
    cache.put(query, doc_types, top_k, results)

    return results


@tool("Get the full content of a document by its ID or citation key")
def get_document(doc_id: str) -> DocumentResult | BudgetExceeded | None:
    """
    Retrieve a document by ID or citation key.

    Args:
        doc_id: Document ID (e.g., "CHARTER-1") or citation key (e.g., "UST-1")

    Returns:
        Full document with metadata and content, BudgetExceeded if token limit reached, or None if not found
    """
    corpus = get_corpus()
    doc = corpus.get_by_id(doc_id) or corpus.get_by_citation(doc_id)

    if doc is None:
        return None

    budget = get_budget()
    ok, tokens = budget.consume(doc.content)

    if not ok:
        return BudgetExceeded(
            requested_tokens=tokens,
            remaining_tokens=budget.remaining(),
        )

    return DocumentResult(
        doc_id=doc.doc_id,
        title=doc.metadata.title,
        doc_type=doc.doc_type,
        citation_key=doc.citation_key,
        priority=doc.priority,
        status=doc.metadata.status,
        effective_date=str(doc.metadata.effective_date)
        if doc.metadata.effective_date
        else None,
        version=doc.metadata.version,
        content=doc.content,
    )


@tool("Search for a specific section within a document by section number")
def search_document_section(doc_id: str, section: str) -> str | None:
    """
    Search for a specific numbered section within a document.

    Returns a focused excerpt around the section (about 1500 chars).
    For full document, use get_document instead.

    Args:
        doc_id: Document ID or citation key
        section: Section number (e.g., "2.3.2.2", "3.1.4")

    Returns:
        Section content with surrounding context if found, None otherwise
    """
    corpus = get_corpus()
    doc = corpus.get_by_id(doc_id) or corpus.get_by_citation(doc_id)
    if doc is None:
        return None

    content = doc.content

    pattern = rf"(?:^|\n)({re.escape(section)})\s*\n"
    match = re.search(pattern, content)

    if match:
        start_idx = match.start()
        section_num = match.group(1)

        parts = section_num.split(".")
        if len(parts) >= 2:
            prefix = ".".join(parts[:-1]) + "."
            next_section_pattern = r"\n(\d+(?:\.\d+)*)\s*\n"

            remaining = content[match.end() :]
            next_match = None
            for m in re.finditer(next_section_pattern, remaining):
                candidate = m.group(1)
                if candidate.startswith(prefix) or not candidate.startswith(
                    parts[0] + "."
                ):
                    next_match = m
                    break

            if next_match:
                end_idx = match.end() + next_match.start()
            else:
                end_idx = min(len(content), match.end() + 1500)
        else:
            end_idx = min(len(content), match.end() + 1500)

        context_start = max(0, start_idx - 200)
        result = content[context_start:end_idx].strip()

        if context_start > 0:
            result = "..." + result
        if end_idx < len(content):
            result = result + "..."

        return result

    idx = content.find(section)
    if idx != -1:
        start = max(0, idx - 200)
        end = min(len(content), idx + 1300)
        return f"...{content[start:end]}..."

    return None


@tool("Create a formal citation for a document fragment")
def cite_fragment(
    doc_id: str,
    section: str,
    quoted_text: str,
) -> CitationResult | CitationError:
    """
    Create a formal citation for a document fragment.

    Args:
        corpus: The corpus index to use
        doc_id: Document ID or citation key
        section: Section reference (e.g., "Article 5.2")
        quoted_text: Exact text being cited

    Returns:
        Formatted citation object
    """
    corpus = get_corpus()
    doc = corpus.get_by_id(doc_id) or corpus.get_by_citation(doc_id)

    if doc is None:
        return CitationError(
            error=f"Document {doc_id} not found",
            doc_id=doc_id,
            section=section,
            quoted_text=quoted_text,
        )

    # Verify the quoted text exists in the document
    text_found = quoted_text.lower() in doc.content.lower()

    return CitationResult(
        doc_id=doc.doc_id,
        citation_key=doc.citation_key,
        title=doc.metadata.title,
        section=section,
        quoted_text=quoted_text,
        verified=text_found,
        formatted_citation=f"[{doc.citation_key}, {section}]",
        full_citation=f'{doc.metadata.title}, {section}: "{quoted_text}"',
    )


def make_corpus_tools() -> list[Tool]:
    """
    Create corpus tools bound to a specific corpus instance.

    Returns:
        List of funcai Tool objects
    """

    return [
        search_corpus,
        get_document,
        search_document_section,
        cite_fragment,
    ]


__all__ = [
    "search_corpus",
    "get_document",
    "search_document_section",
    "cite_fragment",
    "make_corpus_tools",
]
