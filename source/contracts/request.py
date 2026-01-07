"""Request contract - input data for decision making."""

from typing import Literal

from pydantic import BaseModel, Field


class Party(BaseModel):
    """A party involved in the case."""

    name: str = Field(description="Name or pseudonym of the party")
    role: Literal[
        "complainant", "respondent", "witness", "affected_party", "representative"
    ] = Field(description="Role in the case")
    description: str | None = Field(
        default=None, description="Additional information about the party"
    )


class Attachment(BaseModel):
    """An attachment (evidence, statement, etc.) related to the case."""

    file_path: str = Field(description="Path to the markdown file")
    integrity_hash: str | None = Field(
        default=None, description="SHA-256 hash for integrity verification"
    )
    relevance_note: str = Field(
        description="Brief explanation of why this attachment is relevant (1-2 sentences)"
    )
    attachment_type: Literal[
        "statement", "evidence", "transcript", "document", "correspondence", "other"
    ] = Field(default="document", description="Type of attachment")


class Request(BaseModel):
    """
    Input request for the AI Committee Member.

    This contract defines the structure for submitting cases to the system.
    """

    case_id: str | None = Field(
        default=None,
        description="Unique identifier for the case (used for grouping decisions from different models)",
    )

    query: str = Field(description="What needs to be decided - the main question")

    case_type: Literal[
        "ethics",
        "discipline",
        "arbitration",
        "procedure_interpretation",
        "moderation",
        "conflict",
    ] = Field(description="Type of case being submitted")

    org_profile: str | None = Field(
        default=None,
        description="Brief description of the organization and its procedures",
    )

    jurisdiction: list[str] = Field(
        default=["charter", "regulations", "decisions", "clarifications"],
        description="Priority order of normative documents (higher = more authoritative)",
    )

    parties: list[Party] = Field(
        default_factory=list, description="Parties involved in the case"
    )

    requested_remedy: Literal["decision", "sanction", "clarification", "refusal"] = (
        Field(description="What outcome is being requested")
    )

    time_context: str | None = Field(
        default=None,
        description="Relevant temporal context (deadlines, dates of events, etc.)",
    )

    attachments: list[Attachment] = Field(
        default_factory=list,
        description="Supporting documents and evidence in markdown format",
    )

    additional_context: str | None = Field(
        default=None, description="Any additional context relevant to the decision"
    )
