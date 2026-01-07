"""Document loader for the corpus."""

from pathlib import Path
from typing import Any

import combinators
from combinators import lift as L
from kungfu import Error, LazyCoroResult, Ok

from funcai.core.provider import ABCAIProvider
from source.contracts.corpus import CorpusDocument
from source.corpus.constants import CORPUS_EXTENSIONS
from source.corpus.metadata import (
    MetadataExtractor,
    compute_content_hash,
    parse_yaml_frontmatter,
)


class CorpusLoadError(Exception):
    """Raised when corpus loading fails due to invalid documents."""


class DocumentLoader:
    """
    Load documents from the corpus directory.

    Supports:
    - YAML frontmatter for metadata
    - LLM-based metadata extraction as fallback
    - Heuristic extraction if no LLM available
    """

    def __init__(
        self,
        corpus_path: Path,
        provider: ABCAIProvider[Any] | None = None,
    ):
        """
        Initialize document loader.

        Args:
            corpus_path: Path to the corpus directory
            provider: LLM provider for metadata extraction (optional)
        """
        self.corpus_path = Path(corpus_path)
        self.extractor = MetadataExtractor(provider)

    def _find_markdown_files(self) -> list[Path]:
        """Find all markdown files in the corpus directory."""
        files: list[Path] = []

        for ext in CORPUS_EXTENSIONS:
            files.extend(self.corpus_path.rglob(ext))

        return sorted(files)

    def load_document_sync(self, file_path: Path) -> CorpusDocument:
        """
        Load a single document synchronously (without LLM extraction).

        Uses only YAML frontmatter and heuristics.

        Raises:
            CorpusLoadError: If document has no frontmatter or invalid metadata.
        """
        content = file_path.read_text(encoding="utf-8")

        # Check for frontmatter presence
        if not content.startswith("---"):
            raise CorpusLoadError(
                f"Document {file_path} has no YAML frontmatter. "
                "All corpus documents must start with '---' followed by YAML metadata."
            )

        # Extract metadata from frontmatter
        try:
            metadata = self.extractor.extract_from_frontmatter(content, file_path)
        except ValueError as e:
            raise CorpusLoadError(f"Invalid frontmatter in {file_path}: {e}") from e

        if metadata is None:
            raise CorpusLoadError(
                f"Failed to parse YAML frontmatter in {file_path}. Check YAML syntax."
            )

        # Remove frontmatter from content for storage
        _, body = parse_yaml_frontmatter(content)

        return CorpusDocument(
            metadata=metadata,
            content=body.strip(),
            content_hash=compute_content_hash(content),
            file_path=str(file_path),
        )

    async def load_document(self, file_path: Path) -> CorpusDocument:
        """
        Load a single document with full metadata extraction.

        Raises:
            CorpusLoadError: If document has no frontmatter or invalid metadata.
        """
        return self.load_document_sync(file_path)

    def load_corpus_sync(self, strict: bool = True) -> list[CorpusDocument]:
        """
        Load all documents from corpus synchronously.

        Does not use LLM extraction - only YAML frontmatter and heuristics.

        Args:
            strict: If True, raise error on first invalid document.
                   If False, skip invalid documents with warning.
        """
        files = self._find_markdown_files()
        documents: list[CorpusDocument] = []
        errors: list[tuple[Path, str]] = []

        for file_path in files:
            try:
                doc = self.load_document_sync(file_path)
                documents.append(doc)
            except Exception as e:
                if strict:
                    raise CorpusLoadError(
                        f"Failed to load {file_path}: {e}\n"
                        "All corpus documents must have valid YAML frontmatter with 'title' and 'doc_type' fields."
                    ) from e
                errors.append((file_path, str(e)))

        if errors and not strict:
            for path, err in errors:
                print(f"Warning: Failed to load {path}: {err}")

        return documents

    def _make_load_interp(self, file_path: Path) -> LazyCoroResult[CorpusDocument, str]:
        """Wrap load_document as Interp for use with combinators.batch_all."""
        return L.catching_async(
            lambda: self.load_document(file_path),
            on_error=str,
        )

    async def load_corpus(
        self, batch_size: int = 5, strict: bool = True
    ) -> list[CorpusDocument]:
        """
        Load all documents from corpus with parallel processing.

        Uses combinators.batch_all for:
        - Controlled concurrency
        - Unified error handling
        - No partial failures in strict mode

        Args:
            batch_size: Number of documents to process concurrently
            strict: If True, raise error on first invalid document.

        Returns:
            List of loaded documents
        """
        files = self._find_markdown_files()

        if not files:
            return []

        # Use batch_all for collecting all results (never fails early)
        batch_result = await combinators.batch_all(
            items=files,
            handler=self._make_load_interp,
            concurrency=batch_size,
        )

        documents: list[CorpusDocument] = []
        errors: list[tuple[Path, str]] = []

        for file_path, result in zip(files, batch_result.unwrap()):
            match result:
                case Ok(doc):
                    documents.append(doc)
                case Error(e):
                    if strict:
                        raise CorpusLoadError(
                            f"Failed to load {file_path}: {e}\n"
                            "All corpus documents must have valid YAML frontmatter."
                        )
                    errors.append((file_path, e))

        if errors and not strict:
            for path, err in errors:
                print(f"Warning: Failed to load {path}: {err}")

        return documents

    def get_document_by_path(self, relative_path: str) -> CorpusDocument | None:
        """Load a specific document by relative path."""
        full_path = self.corpus_path / relative_path

        if not full_path.exists():
            return None

        return self.load_document_sync(full_path)


__all__ = ["DocumentLoader", "CorpusLoadError"]
