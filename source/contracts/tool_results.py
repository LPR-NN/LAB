"""Typed result contracts for tools in source/tools/."""

from pydantic import BaseModel, Field

# =============================================================================
# validation_tools.py results
# =============================================================================


class ValidationResult(BaseModel):
    """Result of validate_request tool."""

    valid: bool = Field(description="Whether the request is valid")
    error: str | None = Field(
        default=None, description="Error message if JSON is invalid"
    )
    missing_fields: list[str] = Field(
        default_factory=list, description="List of missing required fields"
    )
    warnings: list[str] = Field(
        default_factory=list, description="List of warnings for recommended fields"
    )
    suggestions: list[str] = Field(
        default_factory=list, description="Suggestions for improving the request"
    )
    completeness_score: int = Field(
        default=0, ge=0, le=100, description="Completeness score (0-100)"
    )


class ReadinessChecklist(BaseModel):
    """Checklist for decision readiness."""

    facts_established: bool = Field(description="Whether facts are established")
    norms_identified: bool = Field(description="Whether norms are identified")
    no_critical_gaps: bool = Field(description="Whether there are no critical gaps")


class DecisionReadinessResult(BaseModel):
    """Result of check_decision_readiness tool."""

    ready: bool = Field(description="Whether ready to make a decision")
    facts_count: int = Field(ge=0, description="Number of established facts")
    norms_count: int = Field(ge=0, description="Number of identified norms")
    gaps: list[str] = Field(default_factory=list, description="Identified gaps")
    recommendation: str = Field(description="Recommendation on how to proceed")
    checklist: ReadinessChecklist = Field(description="Readiness checklist")


# =============================================================================
# corpus_tools.py results
# =============================================================================


class SearchResult(BaseModel):
    """A single search result from the corpus."""

    doc_id: str = Field(description="Document identifier")
    title: str = Field(description="Document title")
    doc_type: str = Field(description="Type of document")
    citation_key: str = Field(description="Citation key for referencing")
    priority: int = Field(description="Priority level")
    relevance_score: float = Field(ge=0.0, le=1.0, description="Relevance score")
    preview: str = Field(description="Relevant excerpt from document (chunk)")
    section: str | None = Field(default=None, description="Section header if available")


class DocumentResult(BaseModel):
    """Full document retrieved from corpus."""

    doc_id: str = Field(description="Document identifier")
    title: str = Field(description="Document title")
    doc_type: str = Field(description="Type of document")
    citation_key: str = Field(description="Citation key for referencing")
    priority: int = Field(description="Priority level")
    status: str = Field(description="Document status")
    effective_date: str | None = Field(default=None, description="Effective date")
    version: str | None = Field(default=None, description="Document version")
    content: str = Field(description="Full document content")


class CitationResult(BaseModel):
    """Result of cite_fragment tool."""

    doc_id: str = Field(description="Document identifier")
    citation_key: str = Field(description="Citation key")
    title: str = Field(description="Document title")
    section: str = Field(description="Section reference")
    quoted_text: str = Field(description="Quoted text")
    verified: bool = Field(description="Whether quote was found in document")
    formatted_citation: str = Field(description="Short formatted citation")
    full_citation: str = Field(description="Full citation with quote")


class CitationError(BaseModel):
    """Error result when document not found for citation."""

    error: str = Field(description="Error message")
    doc_id: str = Field(description="Requested document ID")
    section: str = Field(description="Requested section")
    quoted_text: str = Field(description="Requested quote")


# =============================================================================
# precedent_tools.py results
# =============================================================================


class PrecedentResult(BaseModel):
    """A single precedent search result."""

    doc_id: str = Field(description="Document identifier")
    citation_key: str = Field(description="Citation key for referencing")
    title: str = Field(description="Precedent title")
    doc_type: str = Field(description="Type of document")
    similarity_score: float = Field(ge=0.0, le=1.0, description="Similarity score")
    summary: str = Field(description="Case summary")
    effective_date: str | None = Field(default=None, description="Effective date")


class PrecedentInfo(BaseModel):
    """Information about a precedent for comparison."""

    doc_id: str = Field(description="Document identifier")
    citation_key: str = Field(description="Citation key")
    title: str = Field(description="Precedent title")
    content: str = Field(description="Full precedent content")


class PrecedentComparisonResult(BaseModel):
    """Result of compare_with_precedent tool."""

    precedent: PrecedentInfo = Field(description="Precedent information")
    current_facts: str = Field(description="Facts of the current case")
    comparison_guidance: str = Field(description="Guidance for comparison")


class PrecedentComparisonError(BaseModel):
    """Error result when precedent not found."""

    error: str = Field(description="Error message")
    precedent_doc_id: str = Field(description="Requested precedent ID")


# =============================================================================
# reasoning_tools.py results
# =============================================================================


class NormInfo(BaseModel):
    """Information about a norm for conflict analysis."""

    doc_id: str = Field(description="Document identifier")
    citation_key: str = Field(description="Citation key")
    title: str = Field(description="Document title")
    doc_type: str = Field(description="Type of document")
    priority: int = Field(description="Priority level")
    effective_date: str | None = Field(default=None, description="Effective date")
    status: str = Field(description="Document status")


class ConflictInfo(BaseModel):
    """Information about a detected conflict between norms."""

    norm_a: str = Field(description="First norm document ID")
    norm_b: str = Field(description="Second norm document ID")
    conflict_type: str = Field(description="Type of conflict")
    description: str = Field(description="Conflict description")


class NormConflictResult(BaseModel):
    """Result of check_norm_conflicts tool."""

    conflict_detected: bool = Field(description="Whether conflicts were detected")
    reason: str | None = Field(default=None, description="Reason if not enough norms")
    conflicts: list[ConflictInfo] = Field(
        default_factory=list, description="List of detected conflicts"
    )
    norms_found: list[NormInfo] = Field(
        default_factory=list, description="Norms found (when less than 2)"
    )
    norms_analyzed: list[NormInfo] = Field(
        default_factory=list, description="Norms analyzed (when 2 or more)"
    )
    resolution_guidance: str = Field(description="Guidance for resolving conflicts")


class NormSummary(BaseModel):
    """Summary of a norm for priority resolution."""

    doc_id: str = Field(description="Document identifier")
    citation_key: str = Field(description="Citation key")
    doc_type: str = Field(description="Type of document")
    priority: int = Field(description="Priority level")
    effective_date: str | None = Field(default=None, description="Effective date")


class PriorityResolutionResult(BaseModel):
    """Result of apply_priority_rules tool."""

    norm_a: NormSummary = Field(description="First norm summary")
    norm_b: NormSummary = Field(description="Second norm summary")
    conflict_type: str = Field(description="Type of conflict being resolved")
    resolved: bool = Field(description="Whether conflict was resolved")
    prevailing_norm: str | None = Field(
        default=None, description="ID of prevailing norm"
    )
    prevailing_citation: str | None = Field(
        default=None, description="Citation key of prevailing norm"
    )
    reason: str = Field(description="Reason for resolution or guidance")
    guidance: str | None = Field(default=None, description="Additional guidance")
    error: str | None = Field(default=None, description="Error if norms not found")


class FoundNorm(BaseModel):
    """A norm found during reasoning verification."""

    doc_id: str = Field(description="Document identifier")
    citation_key: str = Field(description="Citation key")
    title: str = Field(description="Document title")


class ReasoningVerificationResult(BaseModel):
    """Result of verify_reasoning_chain tool."""

    facts: list[str] = Field(description="List of facts")
    norms: list[FoundNorm] = Field(description="List of found norms")
    proposed_conclusion: str = Field(description="The proposed conclusion")
    verification_checklist: list[str] = Field(description="Verification checklist")
    status: str = Field(description="Verification status")
    note: str = Field(description="Note about verification")


# =============================================================================
# attachment_tools.py results
# =============================================================================


class AttachmentContent(BaseModel):
    """Content of an attachment file."""

    file_path: str = Field(description="Path to the attachment file")
    content: str | None = Field(
        default=None, description="File content if read successfully"
    )
    error: str | None = Field(default=None, description="Error message if read failed")


# =============================================================================
# Common error results
# =============================================================================


class BudgetExceeded(BaseModel):
    """Returned when token budget is exceeded."""

    error: str = Field(default="Token budget exceeded")
    requested_tokens: int = Field(description="Tokens requested by this operation")
    remaining_tokens: int = Field(description="Remaining tokens in budget")
    hint: str = Field(
        default="Formulate your decision with documents already retrieved, or use search_corpus for more targeted queries."
    )


# =============================================================================
# Command/LLM error types for FP-style error handling
# =============================================================================


class CommandError(BaseModel):
    """Error from command execution."""

    message: str = Field(description="Error message")
    command: str = Field(description="Command that failed")

    def __str__(self) -> str:
        return f"[{self.command}] {self.message}"


# Re-export LLMError from source.llm for convenience
# (LLMError is defined in source/llm/call.py with retry info)
