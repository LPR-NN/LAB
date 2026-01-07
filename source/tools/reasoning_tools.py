"""Tools for legal reasoning and norm conflict resolution."""

from functools import partial
from typing import Literal

from funcai import tool
from funcai.agents.tool import Tool

from source.corpus.context import get_corpus
from source.reasoning.priority import PriorityResolver
from source.contracts.tool_results import (
    NormInfo,
    ConflictInfo,
    NormConflictResult,
    NormSummary,
    PriorityResolutionResult,
    FoundNorm,
    ReasoningVerificationResult,
)


# Priority by document type (higher = more authoritative)
DOC_TYPE_PRIORITY = {
    "charter": 100,
    "regulations": 80,
    "code": 70,
    "decision": 50,
    "precedent": 40,
    "clarification": 30,
}


@tool("Check for conflicts between applicable norms")
def check_norm_conflicts(norm_ids: str) -> NormConflictResult:
    """
    Check if there are conflicts between the specified norms.

    Args:
        corpus: The corpus index to use
        norm_ids: Comma-separated list of document IDs or citation keys

    Returns:
        Analysis of potential conflicts between norms
    """
    corpus = get_corpus()
    ids = [n.strip() for n in norm_ids.split(",")]
    norms: list[NormInfo] = []

    for norm_id in ids:
        doc = corpus.get_by_id(norm_id) or corpus.get_by_citation(norm_id)
        if doc:
            norms.append(
                NormInfo(
                    doc_id=doc.doc_id,
                    citation_key=doc.citation_key,
                    title=doc.metadata.title,
                    doc_type=doc.doc_type,
                    priority=doc.priority,
                    effective_date=str(doc.metadata.effective_date)
                    if doc.metadata.effective_date
                    else None,
                    status=doc.metadata.status,
                )
            )

    if len(norms) < 2:
        return NormConflictResult(
            conflict_detected=False,
            reason="Need at least 2 norms to check for conflicts",
            norms_found=norms,
            resolution_guidance="Need at least 2 norms to check for conflicts.",
        )

    # Check for potential conflicts
    conflicts: list[ConflictInfo] = []

    for i, norm_a in enumerate(norms):
        for norm_b in norms[i + 1 :]:
            # Same type but different priorities might conflict
            if norm_a.doc_type == norm_b.doc_type:
                if norm_a.priority != norm_b.priority:
                    conflicts.append(
                        ConflictInfo(
                            norm_a=norm_a.doc_id,
                            norm_b=norm_b.doc_id,
                            conflict_type="priority_mismatch",
                            description=f"Same document type but different priorities: {norm_a.priority} vs {norm_b.priority}",
                        )
                    )

            # Check for supersession
            if norm_a.status == "superseded" or norm_b.status == "superseded":
                conflicts.append(
                    ConflictInfo(
                        norm_a=norm_a.doc_id,
                        norm_b=norm_b.doc_id,
                        conflict_type="superseded_norm",
                        description="One of the norms has been superseded",
                    )
                )

    resolution_guidance = (
        """
If conflicts exist, apply resolution rules:
1. LEX SUPERIOR: Higher authority norm prevails (charter > regulations > code > decision)
2. LEX SPECIALIS: More specific norm prevails over general
3. LEX POSTERIOR: Later norm prevails (same authority level)
"""
        if conflicts
        else "No conflicts detected between specified norms."
    )

    return NormConflictResult(
        conflict_detected=len(conflicts) > 0,
        conflicts=conflicts,
        norms_analyzed=norms,
        resolution_guidance=resolution_guidance,
    )


@tool("Apply priority rules to resolve conflict between two norms")
def apply_priority_rules(
    norm_a_id: str,
    norm_b_id: str,
    conflict_type: Literal["lex_superior", "lex_posterior", "lex_specialis"],
) -> PriorityResolutionResult:
    """
    Apply legal priority rules to resolve a norm conflict.

    Args:
        corpus: The corpus index to use
        norm_a_id: First norm's document ID or citation key
        norm_b_id: Second norm's document ID or citation key
        conflict_type: Type of conflict (lex_superior, lex_specialis, lex_posterior)

    Returns:
        Resolution with the prevailing norm and justification
    """
    corpus = get_corpus()
    norm_a = corpus.get_by_id(norm_a_id) or corpus.get_by_citation(norm_a_id)
    norm_b = corpus.get_by_id(norm_b_id) or corpus.get_by_citation(norm_b_id)

    # Handle missing norms - create dummy summaries for error response
    if norm_a is None or norm_b is None:
        missing: list[str] = []
        if norm_a is None:
            missing.append(norm_a_id)
        if norm_b is None:
            missing.append(norm_b_id)

        # Create placeholder summaries for error case
        norm_a_summary = NormSummary(
            doc_id=norm_a_id,
            citation_key=norm_a_id,
            doc_type="unknown",
            priority=0,
            effective_date=None,
        )
        norm_b_summary = NormSummary(
            doc_id=norm_b_id,
            citation_key=norm_b_id,
            doc_type="unknown",
            priority=0,
            effective_date=None,
        )

        return PriorityResolutionResult(
            norm_a=norm_a_summary,
            norm_b=norm_b_summary,
            conflict_type=conflict_type,
            resolved=False,
            reason=f"Norm(s) not found: {', '.join(missing)}",
            error=f"Norm(s) not found: {', '.join(missing)}",
        )

    norm_a_summary = NormSummary(
        doc_id=norm_a.doc_id,
        citation_key=norm_a.citation_key,
        doc_type=norm_a.doc_type,
        priority=norm_a.priority,
        effective_date=str(norm_a.metadata.effective_date)
        if norm_a.metadata.effective_date
        else None,
    )
    norm_b_summary = NormSummary(
        doc_id=norm_b.doc_id,
        citation_key=norm_b.citation_key,
        doc_type=norm_b.doc_type,
        priority=norm_b.priority,
        effective_date=str(norm_b.metadata.effective_date)
        if norm_b.metadata.effective_date
        else None,
    )

    normalized_type = conflict_type.lower().replace("lex_", "")

    if normalized_type == "specialis":
        return PriorityResolutionResult(
            norm_a=norm_a_summary,
            norm_b=norm_b_summary,
            conflict_type=conflict_type,
            resolved=False,
            reason="Lex specialis requires content analysis to determine which norm is more specific. The more specific norm should prevail.",
            guidance="Analyze both norms to determine which addresses the specific situation more directly. The norm with narrower scope prevails.",
        )

    if normalized_type not in ("superior", "posterior"):
        return PriorityResolutionResult(
            norm_a=norm_a_summary,
            norm_b=norm_b_summary,
            conflict_type=conflict_type,
            resolved=False,
            reason=f"Unknown conflict type: {conflict_type}. Use: lex_superior, lex_specialis, or lex_posterior",
        )

    try:
        prevailing, reason = PriorityResolver.resolve(norm_a, norm_b, normalized_type)
        return PriorityResolutionResult(
            norm_a=norm_a_summary,
            norm_b=norm_b_summary,
            conflict_type=conflict_type,
            resolved=True,
            prevailing_norm=prevailing.doc_id,
            prevailing_citation=prevailing.citation_key,
            reason=reason,
        )
    except ValueError as e:
        return PriorityResolutionResult(
            norm_a=norm_a_summary,
            norm_b=norm_b_summary,
            conflict_type=conflict_type,
            resolved=False,
            reason=str(e),
        )


@tool("Verify that a reasoning chain is logically sound")
def verify_reasoning_chain(
    facts: str,
    norms: str,
    conclusion: str,
) -> ReasoningVerificationResult:
    """
    Verify that the conclusion follows logically from facts and norms.

    Args:
        corpus: The corpus index to use
        facts: Comma-separated list of established facts
        norms: Comma-separated list of applicable norm IDs
        conclusion: The proposed conclusion

    Returns:
        Verification result with analysis
    """
    corpus = get_corpus()
    fact_list = [f.strip() for f in facts.split(",") if f.strip()]
    norm_ids = [n.strip() for n in norms.split(",") if n.strip()]

    # Look up norms
    found_norms: list[FoundNorm] = []
    for norm_id in norm_ids:
        doc = corpus.get_by_id(norm_id) or corpus.get_by_citation(norm_id)
        if doc:
            found_norms.append(
                FoundNorm(
                    doc_id=doc.doc_id,
                    citation_key=doc.citation_key,
                    title=doc.metadata.title,
                )
            )

    return ReasoningVerificationResult(
        facts=fact_list,
        norms=found_norms,
        proposed_conclusion=conclusion,
        verification_checklist=[
            "1. Each fact must be supported by evidence",
            "2. Each norm must be applicable to the facts",
            "3. The conclusion must follow from applying the norm to the facts",
            "4. No logical gaps or unsupported leaps",
            "5. Consider alternative interpretations",
        ],
        status="requires_analysis",
        note="This tool provides the framework for verification. The agent must perform the actual logical analysis.",
    )


def make_reasoning_tools() -> list[Tool]:
    """
    Create reasoning tools bound to a specific corpus.

    Args:
        corpus: The corpus index to use

    Returns:
        List of funcai Tool objects
    """

    return [
        check_norm_conflicts,
        apply_priority_rules,
        verify_reasoning_chain,
    ]


__all__ = [
    "check_norm_conflicts",
    "apply_priority_rules",
    "verify_reasoning_chain",
    "make_reasoning_tools",
]
