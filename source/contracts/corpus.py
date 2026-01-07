"""Corpus document contracts - normative base structure."""

from datetime import date
from typing import Literal
from pydantic import BaseModel, Field


class CorpusMetadata(BaseModel):
    """
    Metadata for a corpus document.

    Can be provided via YAML frontmatter or extracted by LLM.
    """

    doc_id: str = Field(description="Unique document identifier")
    title: str = Field(description="Document title")

    doc_type: Literal[
        "charter",
        "regulations",
        "code",
        "decision",
        "precedent",
        "clarification",
    ] = Field(description="Type of normative document")

    effective_date: date | None = Field(
        default=None, description="Date when document became effective"
    )

    version: str | None = Field(default=None, description="Document version")

    status: Literal["active", "superseded", "draft"] = Field(
        default="active", description="Current status of the document"
    )

    supersedes: list[str] | None = Field(
        default=None, description="List of doc_ids this document supersedes"
    )

    superseded_by: str | None = Field(
        default=None, description="doc_id of document that supersedes this one"
    )

    citation_key: str = Field(description="Short key for citing this document")

    priority: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Priority level (higher = more authoritative)",
    )

    tags: list[str] = Field(
        default_factory=list, description="Tags for categorization and search"
    )


class CorpusDocument(BaseModel):
    """A complete document from the normative corpus."""

    metadata: CorpusMetadata = Field(description="Document metadata")
    content: str = Field(description="Full document content in markdown")
    content_hash: str = Field(description="SHA-256 hash of content for integrity")
    file_path: str = Field(description="Original file path")

    @property
    def doc_id(self) -> str:
        return self.metadata.doc_id

    @property
    def citation_key(self) -> str:
        return self.metadata.citation_key

    @property
    def doc_type(self) -> str:
        return self.metadata.doc_type

    @property
    def priority(self) -> int:
        return self.metadata.priority

    @property
    def is_active(self) -> bool:
        return self.metadata.status == "active"
