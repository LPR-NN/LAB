"""Metadata extraction from documents - YAML frontmatter or LLM-based."""

import hashlib
import re
from pathlib import Path
from typing import Any

import yaml
from funcai.core.provider import ABCAIProvider

from source.contracts.corpus import CorpusMetadata

# Priority mapping for document types
DOC_TYPE_PRIORITIES: dict[str, int] = {
    "charter": 100,
    "regulations": 80,
    "code": 70,
    "decision": 50,
    "precedent": 40,
    "clarification": 30,
}


def parse_yaml_frontmatter(content: str) -> tuple[dict[str, Any] | None, str]:
    """
    Parse YAML frontmatter from markdown content.

    Returns:
        (metadata_dict, content_without_frontmatter) or (None, original_content)
    """
    # Pattern: starts with ---, then YAML content, then ---
    pattern = r"^---\s*\n(.*?)\n---\s*\n(.*)$"
    match = re.match(pattern, content, re.DOTALL)

    if not match:
        return None, content

    try:
        yaml_content = match.group(1)
        body_content = match.group(2)
        metadata = yaml.safe_load(yaml_content)

        if not isinstance(metadata, dict):
            return None, content

        return metadata, body_content
    except yaml.YAMLError:
        return None, content


def compute_content_hash(content: str) -> str:
    """Compute SHA-256 hash of content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def generate_doc_id(file_path: Path) -> str:
    """Generate document ID from file path."""
    # Use parent folder + stem as ID
    parts = file_path.parts[-2:] if len(file_path.parts) >= 2 else file_path.parts
    stem = file_path.stem
    folder = parts[0] if len(parts) > 1 else ""

    if folder:
        return f"{folder}-{stem}".upper().replace("_", "-").replace(" ", "-")
    return stem.upper().replace("_", "-").replace(" ", "-")


def generate_citation_key(file_path: Path, doc_type: str) -> str:
    """Generate citation key for a document."""
    stem = file_path.stem
    # Use abbreviation based on doc_type + number if present
    prefix_map = {
        "charter": "UST",
        "regulations": "REG",
        "code": "COD",
        "decision": "DEC",
        "precedent": "PRE",
        "clarification": "CLR",
    }
    prefix = prefix_map.get(doc_type, "DOC")

    # Extract number if present in filename
    numbers = re.findall(r"\d+", stem)
    if numbers:
        return f"{prefix}-{numbers[0]}"

    # Otherwise use hash
    short_hash = hashlib.md5(stem.encode()).hexdigest()[:6].upper()
    return f"{prefix}-{short_hash}"


class MetadataExtractor:
    """Extract metadata from documents using YAML frontmatter or LLM."""

    def __init__(self, provider: ABCAIProvider[Any] | None = None):
        """
        Initialize extractor.

        Args:
            provider: LLM provider for extraction when frontmatter is missing.
                     If None, will use heuristics only.
        """
        self.provider = provider

    def extract_from_frontmatter(
        self, content: str, file_path: Path
    ) -> CorpusMetadata | None:
        """
        Extract metadata from YAML frontmatter.

        Returns:
            CorpusMetadata if frontmatter is valid, None otherwise.

        Raises:
            ValueError: If frontmatter is present but missing required fields.
        """
        metadata_dict, _ = parse_yaml_frontmatter(content)

        if metadata_dict is None:
            return None

        required_fields = ["title", "doc_type"]
        missing = [f for f in required_fields if f not in metadata_dict]
        if missing:
            raise ValueError(
                f"Missing required frontmatter fields: {', '.join(missing)}"
            )

        valid_doc_types = {
            "charter",
            "regulations",
            "code",
            "decision",
            "precedent",
            "clarification",
        }
        if metadata_dict["doc_type"] not in valid_doc_types:
            raise ValueError(
                f"Invalid doc_type '{metadata_dict['doc_type']}'. "
                f"Must be one of: {', '.join(sorted(valid_doc_types))}"
            )

        # Fill in defaults from file path
        if "doc_id" not in metadata_dict:
            metadata_dict["doc_id"] = generate_doc_id(file_path)

        if "citation_key" not in metadata_dict:
            doc_type = metadata_dict.get("doc_type", "document")
            metadata_dict["citation_key"] = generate_citation_key(file_path, doc_type)

        if "priority" not in metadata_dict:
            doc_type = metadata_dict.get("doc_type", "")
            metadata_dict["priority"] = DOC_TYPE_PRIORITIES.get(doc_type, 0)

        return CorpusMetadata.model_validate(metadata_dict)


__all__ = [
    "MetadataExtractor",
    "parse_yaml_frontmatter",
    "compute_content_hash",
    "generate_doc_id",
    "generate_citation_key",
]
