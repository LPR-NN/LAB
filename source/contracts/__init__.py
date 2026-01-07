"""Data contracts for the AI Committee Member system."""

from source.contracts.request import Request, Party, Attachment
from source.contracts.decision import (
    Decision,
    FindingOfFact,
    ApplicableNorm,
    ReasoningStep,
    Citation,
)
from source.contracts.corpus import CorpusMetadata, CorpusDocument
from source.contracts.audit import DecisionPackage, CorpusDocumentRef, CitedFragment
from source.contracts.tool_results import (
    # Validation tools
    ValidationResult,
    ReadinessChecklist,
    DecisionReadinessResult,
    # Corpus tools
    SearchResult,
    DocumentResult,
    CitationResult,
    CitationError,
    # Precedent tools
    PrecedentResult,
    PrecedentInfo,
    PrecedentComparisonResult,
    PrecedentComparisonError,
    # Reasoning tools
    NormInfo,
    ConflictInfo,
    NormConflictResult,
    NormSummary,
    PriorityResolutionResult,
    FoundNorm,
    ReasoningVerificationResult,
)

__all__ = [
    # Request
    "Request",
    "Party",
    "Attachment",
    # Decision
    "Decision",
    "FindingOfFact",
    "ApplicableNorm",
    "ReasoningStep",
    "Citation",
    # Corpus
    "CorpusMetadata",
    "CorpusDocument",
    # Audit
    "DecisionPackage",
    "CorpusDocumentRef",
    "CitedFragment",
    # Tool Results - Validation
    "ValidationResult",
    "ReadinessChecklist",
    "DecisionReadinessResult",
    # Tool Results - Corpus
    "SearchResult",
    "DocumentResult",
    "CitationResult",
    "CitationError",
    # Tool Results - Precedent
    "PrecedentResult",
    "PrecedentInfo",
    "PrecedentComparisonResult",
    "PrecedentComparisonError",
    # Tool Results - Reasoning
    "NormInfo",
    "ConflictInfo",
    "NormConflictResult",
    "NormSummary",
    "PriorityResolutionResult",
    "FoundNorm",
    "ReasoningVerificationResult",
]
