"""Audit contracts - decision package for reproducibility."""

import datetime
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolCallRecord(BaseModel):
    """Record of a single tool call made by the agent."""

    tool_name: str = Field(description="Name of the tool called")
    arguments: dict[str, Any] = Field(description="Arguments passed to the tool")
    result_preview: str = Field(description="Preview of the result (truncated)")
    success: bool = Field(description="Whether the call succeeded")
    error: str | None = Field(default=None, description="Error message if failed")
    duration_ms: float = Field(description="Execution time in milliseconds")
    timestamp: datetime.datetime = Field(description="When the call was made (UTC)")
    cached: bool = Field(default=False, description="Whether result was from cache")


class ToolCallHistory(BaseModel):
    """Complete history of tool calls during decision making."""

    calls: list[ToolCallRecord] = Field(default_factory=list)
    total_calls: int = Field(default=0, description="Total number of tool calls")
    total_duration_ms: float = Field(
        default=0.0, description="Total execution time in ms"
    )
    cache_hits: int = Field(default=0, description="Number of cache hits")


class CorpusDocumentRef(BaseModel):
    """Reference to a corpus document used in a decision."""

    doc_id: str = Field(description="Document identifier")
    version: str | None = Field(description="Document version")
    content_hash: str = Field(description="SHA-256 hash at time of decision")


class CitedFragment(BaseModel):
    """A specific fragment cited in the decision."""

    doc_id: str = Field(description="Source document identifier")
    section: str = Field(description="Section reference")
    quoted_text: str = Field(description="Exact quoted text")
    context: str | None = Field(
        default=None, description="Surrounding context if relevant"
    )


class DecisionPackage(BaseModel):
    """
    Complete audit package for a decision.

    Contains everything needed to reproduce and verify a decision:
    - Original request
    - Corpus snapshot (documents used)
    - Citations made
    - Final decision
    - Model and system information
    """

    package_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique package identifier",
    )

    timestamp: datetime.datetime = Field(
        default_factory=datetime.datetime.now(datetime.timezone.utc),
        description="When decision was made (UTC)",
    )

    # Original input
    request_json: str = Field(
        description="Original request serialized as JSON for exact reproduction"
    )

    # Corpus state
    corpus_snapshot: list[CorpusDocumentRef] = Field(
        description="Documents from corpus that were available/used"
    )

    # What was actually cited
    cited_fragments: list[CitedFragment] = Field(
        description="Specific fragments that were cited"
    )

    # Tool call history (for debugging and analysis)
    tool_calls: ToolCallHistory | None = Field(
        default=None,
        description="History of tool calls made during decision process",
    )

    # Output
    decision_json: str = Field(description="Final decision serialized as JSON")

    # System information
    agent_version: str = Field(description="Version of the AI Committee Member")
    model_id: str = Field(description="Exact model identifier (not alias)")
    provider: str = Field(description="LLM provider used")

    mode: Literal["deterministic", "creative"] = Field(
        default="deterministic", description="Processing mode"
    )

    temperature: float = Field(default=0.0, description="Temperature setting used")

    # Integrity
    content_hash: str | None = Field(
        default=None, description="SHA-256 hash of the package (computed after save)"
    )
