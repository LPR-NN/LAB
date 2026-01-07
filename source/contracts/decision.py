"""Decision contract - output structure for committee decisions."""

from typing import Literal
from pydantic import BaseModel, Field


class FindingOfFact(BaseModel):
    """An established fact derived from evidence."""

    fact: str = Field(description="Statement of the established fact")
    evidence_refs: list[str] = Field(
        description="References to evidence (attachment paths or citation keys)"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence level in this fact (0.0 to 1.0)"
    )


class ApplicableNorm(BaseModel):
    """A norm/rule from the corpus that applies to this case."""

    doc_id: str = Field(description="Document identifier")
    citation_key: str = Field(description="How to cite this document")
    section: str = Field(description="Specific section or article")
    priority: int = Field(description="Priority level (higher = more authoritative)")
    relevance: str = Field(description="Why this norm applies to the case")


class ReasoningStep(BaseModel):
    """A single step in the reasoning chain: fact -> norm -> conclusion."""

    fact: str = Field(description="The fact being analyzed")
    norm_or_precedent: str = Field(
        description="The applicable norm or precedent (with citation)"
    )
    conclusion: str = Field(description="The conclusion derived from applying the norm")


class Citation(BaseModel):
    """A specific citation from the normative corpus."""

    doc_id: str = Field(description="Document identifier")
    section: str = Field(description="Section, article, or paragraph reference")
    quoted_text: str = Field(description="Exact text being cited")


class Decision(BaseModel):
    """
    Output decision from the AI Committee Member.

    This contract defines the structure of decisions produced by the system.
    Every decision must include proper reasoning and citations.
    """

    verdict: Literal[
        "decision", "sanction", "clarification", "refusal", "needs_more_info"
    ] = Field(description="Type of verdict being issued")

    verdict_summary: str = Field(
        description="One-sentence summary of the decision for quick reference"
    )

    findings_of_fact: list[FindingOfFact] = Field(
        description="Established facts based on evidence"
    )

    applicable_norms: list[ApplicableNorm] = Field(
        description="List of norms that apply, ordered by priority"
    )

    reasoning: list[ReasoningStep] = Field(
        description="Step-by-step justification: fact -> norm -> conclusion"
    )

    citations: list[Citation] = Field(description="All citations used in the decision")

    uncertainty: str = Field(
        description="What remains unclear and what data could change the outcome"
    )

    minority_view: str | None = Field(
        default=None,
        description="Alternative interpretation if norms allow for reasonable disagreement",
    )

    recommended_next_steps: list[str] | None = Field(
        default=None,
        description="Suggested actions or what needs to be clarified/verified",
    )

    missing_information: list[str] | None = Field(
        default=None,
        description="If verdict=needs_more_info, specific questions that need answers",
    )
