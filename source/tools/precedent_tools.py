"""Tools for working with precedents."""

from funcai import tool
from funcai.agents.tool import Tool
from source.contracts.tool_results import (
    BudgetExceeded,
    PrecedentComparisonError,
    PrecedentComparisonResult,
    PrecedentInfo,
    PrecedentResult,
)
from source.corpus.context import get_budget, get_corpus

DECISION_INDICATORS = (
    "исключить",
    "взыскание",
    "предупреждение",
    "наложить",
    "отклонить",
    "удовлетворить",
    "отменить",
    "санкци",
    "нарушени",
    "установил",
    "постановил",
)

RESOLUTION_MARKERS = (
    "рассмотрев",
    "установил",
    "постановил",
    "решил",
    "принял решение",
    "федеральный комитет",
    "этический комитет",
)


def _is_mostly_english(text: str) -> bool:
    """Check if text is predominantly English (Latin characters)."""
    import re

    latin_words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    cyrillic_words = re.findall(r"\b[а-яёА-ЯЁ]{3,}\b", text.lower())

    if not cyrillic_words:
        return True
    return len(latin_words) > len(cyrillic_words) * 2


def _is_actual_decision(content: str) -> bool:
    """Check if document contains actual decision with resolution."""
    content_lower = content.lower()

    if len(content) < 300:
        return False

    indicator_count = sum(1 for ind in DECISION_INDICATORS if ind in content_lower)
    resolution_count = sum(1 for m in RESOLUTION_MARKERS if m in content_lower)

    has_structure = resolution_count >= 1 and indicator_count >= 1
    has_substance = len(content) > 800 or (
        "решение" in content_lower and indicator_count >= 2
    )

    return has_structure and has_substance


@tool("Find similar precedent cases based on facts and case type")
def find_precedents(
    facts: str,
    case_type: str,
    top_k: int = 3,
) -> list[PrecedentResult]:
    """
    Find precedent cases similar to the current situation.

    Uses semantic search (embeddings) to find similar cases by meaning.
    Write detailed facts - the more context, the better the matches.

    Args:
        facts: DETAILED description of the facts (in Russian for best results).
               Good: "Член партии опубликовал в публичном телеграм-канале посты
                      с оскорблениями в адрес председателя регионального отделения,
                      назвав его 'вором' и 'предателем'. Посты набрали 5000 просмотров."
               Bad: "оскорбление председателя" (too short, loses context)
        case_type: Type of case (ethics, discipline, arbitration, etc.)
        top_k: Number of precedents to return

    Returns:
        List of relevant precedent cases with similarity scores
    """
    corpus = get_corpus()

    # Warn if facts are in English — corpus is Russian, search works better in Russian
    if _is_mostly_english(facts):
        return [
            PrecedentResult(
                doc_id="ERROR",
                citation_key="ERROR",
                title="ERROR: Facts should be written in RUSSIAN",
                doc_type="error",
                similarity_score=0.0,
                summary=(
                    "The corpus is in Russian. Semantic search works better "
                    "when the query language matches the corpus. "
                    "Please rewrite your facts in RUSSIAN for best results. "
                    "Example: instead of 'Member published tweets criticizing party leadership', "
                    "write 'Член партии опубликовал твиты с критикой руководства партии'."
                ),
                effective_date=None,
            )
        ]

    # Build semantic query: case type + full facts description
    # Vector search understands meaning, no need to extract keywords
    query = f"{case_type}: {facts}"

    results = corpus.search(
        query=query,
        doc_types=["precedent", "decision"],
        top_k=top_k * 3,
    )

    if not results:
        return []

    from source.contracts.corpus import CorpusDocument

    filtered_results: list[tuple[CorpusDocument, float]] = []
    for doc, score in results:
        if _is_actual_decision(doc.content):
            filtered_results.append((doc, score))

    if not filtered_results:
        filtered_results = list(results[:top_k])

    max_score = max(score for _, score in filtered_results) if filtered_results else 1.0
    normalizer = max_score if max_score > 0 else 1.0

    precedents: list[PrecedentResult] = []
    for doc, score in filtered_results[:top_k]:
        content = doc.content
        # Extract meaningful summary: first ~800 chars or until natural break
        MAX_SUMMARY = 800
        summary_end = min(MAX_SUMMARY, len(content))
        # Try to find a natural break point
        for marker in ["\n\nРЕШЕНИЕ", "\n\nПОСТАНОВИЛ", "\n## "]:
            idx = content.find(marker)
            if 300 < idx < MAX_SUMMARY + 200:
                summary_end = idx
                break
        summary = content[:summary_end].strip()
        if len(content) > summary_end:
            summary += "..."

        normalized_score = min(score / normalizer, 1.0)

        precedents.append(
            PrecedentResult(
                doc_id=doc.doc_id,
                citation_key=doc.citation_key,
                title=doc.metadata.title,
                doc_type=doc.doc_type,
                similarity_score=round(normalized_score, 4),
                summary=summary,
                effective_date=str(doc.metadata.effective_date)
                if doc.metadata.effective_date
                else None,
            )
        )

    return precedents


@tool("Compare current case facts with a specific precedent")
def compare_with_precedent(
    current_facts: str,
    precedent_doc_id: str,
) -> PrecedentComparisonResult | PrecedentComparisonError | BudgetExceeded:
    """
    Compare the current case with a specific precedent.

    Args:
        current_facts: Facts of the current case
        precedent_doc_id: ID of the precedent to compare with

    Returns:
        Comparison analysis, error if not found, or BudgetExceeded if token limit reached
    """
    corpus = get_corpus()

    precedent = corpus.get_by_id(precedent_doc_id) or corpus.get_by_citation(
        precedent_doc_id
    )

    if precedent is None:
        return PrecedentComparisonError(
            error=f"Precedent {precedent_doc_id} not found",
            precedent_doc_id=precedent_doc_id,
        )

    budget = get_budget()
    ok, tokens = budget.consume(precedent.content)

    if not ok:
        return BudgetExceeded(
            requested_tokens=tokens,
            remaining_tokens=budget.remaining(),
        )

    return PrecedentComparisonResult(
        precedent=PrecedentInfo(
            doc_id=precedent.doc_id,
            citation_key=precedent.citation_key,
            title=precedent.metadata.title,
            content=precedent.content,
        ),
        current_facts=current_facts,
        comparison_guidance="""
To compare these cases, you MUST analyze:

1. SEVERITY: Compare specific acts (words/actions, their target, context)
2. REACH: Audience size — public platform vs private chat, spread/virality
3. HARM: Documented consequences — media coverage, third-party reactions, reputational damage
4. CONTEXT: Was target a public figure? Was there provocation? Retraction?
5. AGGRAVATING: Prior sanctions, refusal to apologize, continued behavior
6. MITIGATING: First offense, apology, deletion of content
7. NORMS APPLIED: Which norms were invoked in the precedent? Same or different?
8. SANCTION: What was the outcome? Why that specific sanction?

Conclude with explicit comparison:
- "Дело [X] сопоставимо: [matching factors]" → precedent applies
- "Дело [X] отличается: [differences]" → precedent distinguished/inapplicable
""",
    )


def make_precedent_tools() -> list[Tool]:
    """
    Create precedent tools bound to a specific corpus.

    Args:
        corpus: The corpus index to use

    Returns:
        List of funcai Tool objects
    """
    return [
        find_precedents,
        compare_with_precedent,
    ]


__all__ = [
    "find_precedents",
    "compare_with_precedent",
    "make_precedent_tools",
]
