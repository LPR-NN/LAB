"""Simple corpus annotation with LLM metadata extraction."""

import asyncio
import re
from datetime import date
from pathlib import Path
from typing import Any, Literal

import combinators
import ftfy
import yaml
from combinators import lift as L
from combinators import rate_limit
from combinators.concurrency import RateLimitPolicy
from kungfu import Error, LazyCoroResult, Ok
from pydantic import BaseModel, Field

from funcai.core import message
from funcai.core.dialogue import Dialogue
from funcai.core.provider import ABCAIProvider
from source.contracts.corpus import CorpusMetadata
from source.corpus.constants import CORPUS_EXTENSIONS
from source.corpus.metadata import (
    DOC_TYPE_PRIORITIES,
    generate_citation_key,
    generate_doc_id,
    parse_yaml_frontmatter,
)

MOJIBAKE_PATTERN = re.compile(r"[Ð-Ñ][°-¿À-ÿ]")


def _is_mojibake(text: str) -> bool:
    """
    Detect if text contains mojibake (UTF-8 decoded as Latin-1/cp1252).

    Typical pattern: Cyrillic UTF-8 bytes interpreted as Latin-1 produce
    sequences like "Ð ÐµÑˆÐµÐ½Ð¸Ðµ" instead of "Решение".
    """
    matches = MOJIBAKE_PATTERN.findall(text)
    return len(matches) > 5


def fix_encoding(text: str) -> tuple[str, bool]:
    """
    Fix text encoding using ftfy library.

    Handles various encoding issues including:
    - UTF-8 decoded as Latin-1/cp1252 (mojibake)
    - Double-encoded UTF-8
    - Mixed encoding issues

    Returns:
        Tuple of (fixed_text, was_fixed)
    """
    if not _is_mojibake(text):
        return text, False

    fixed = ftfy.fix_text(text)

    if fixed != text:
        return fixed, True

    return text, False


class ExtractedMetadata(BaseModel):
    """Schema for LLM extraction - minimal fields."""

    title: str = Field(description="Document title (short, descriptive)")
    doc_type: Literal[
        "charter",
        "regulations",
        "code",
        "decision",
        "precedent",
        "clarification",
    ] = Field(description="Type of normative document")
    effective_date: date | None = Field(
        default=None,
        description="Date when document became effective (if mentioned)",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="3-5 tags for categorization",
    )


EXTRACTION_PROMPT = """\
Проанализируй нормативный документ и извлеки метаданные.

Правила:
- title: Краткое описательное название на русском
- doc_type: Выбери подходящий тип (charter, regulations, code, decision, precedent, clarification)
- effective_date: Только если дата явно указана в документе (YYYY-MM-DD)
- tags: 3-5 релевантных тегов для поиска/категоризации

Пример 1:
Документ: "31 марта 2025 года состоялось заседание ЭК. Повестка: 1. Рассмотрение жалобы Иванова"
Ответ: {{"title": "Заседание ЭК 31 марта 2025: рассмотрение жалоб", "doc_type": "decision", "effective_date": "2025-03-31", "tags": ["заседание ЭК", "жалобы", "рассмотрение жалоб"]}}

Пример 2:
Документ: "УСТАВ Либертарианской партии России. Глава 1. Общие положения..."
Ответ: {{"title": "Устав Либертарианской партии России", "doc_type": "charter", "effective_date": null, "tags": ["устав", "ЛПР", "основной документ"]}}

Содержимое документа:
---
{content}
---

Извлеки метаданные в JSON формате."""


class CorpusAnnotator:
    """
    Annotate corpus documents with LLM-generated metadata.

    Простая логика:
    1. Найти файлы без frontmatter
    2. Для каждого: один LLM-вызов → ExtractedMetadata
    3. Записать YAML frontmatter в файл
    """

    def __init__(self, provider: ABCAIProvider[Any]):
        self.provider = provider

    def find_unannotated(self, corpus_path: Path) -> list[Path]:
        """Find markdown files without YAML frontmatter."""
        unannotated: list[Path] = []

        for ext in CORPUS_EXTENSIONS:
            for file_path in corpus_path.rglob(ext):
                content = file_path.read_text(encoding="utf-8")
                metadata, _ = parse_yaml_frontmatter(content)

                if metadata is None:
                    unannotated.append(file_path)

        return sorted(unannotated)

    def find_all_files(self, corpus_path: Path) -> list[Path]:
        """Find all markdown files in corpus."""
        files: list[Path] = []

        for ext in CORPUS_EXTENSIONS:
            for file_path in corpus_path.rglob(ext):
                files.append(file_path)

        return sorted(files)

    async def find_unannotated_async(
        self, corpus_path: Path, concurrency: int = 20
    ) -> list[Path]:
        """
        Find markdown files without YAML frontmatter using parallel file reading.

        Faster than sync version for large corpora (100+ files).

        Args:
            corpus_path: Path to corpus directory
            concurrency: Number of files to read in parallel

        Returns:
            Sorted list of paths to unannotated files
        """
        # Collect all file paths first (fast, no I/O)
        all_files: list[Path] = []
        for ext in CORPUS_EXTENSIONS:
            all_files.extend(corpus_path.rglob(ext))

        if not all_files:
            return []

        async def check_file(path: Path) -> bool:
            """Check if file lacks frontmatter. Returns True if unannotated."""
            content = await asyncio.to_thread(path.read_text, encoding="utf-8")
            metadata, _ = parse_yaml_frontmatter(content)
            return metadata is None

        def make_check_interp(path: Path) -> LazyCoroResult[bool, str]:
            return L.catching_async(lambda: check_file(path), on_error=str)

        # Check all files in parallel
        batch_result = await combinators.batch_all(
            items=all_files,
            handler=make_check_interp,
            concurrency=concurrency,
        )

        # Filter unannotated files
        unannotated: list[Path] = []
        for path, result in zip(all_files, batch_result.unwrap()):
            match result:
                case Ok():
                    unannotated.append(path)
                case Error(_):
                    pass

        return sorted(unannotated)

    async def extract_metadata(
        self, content: str, file_path: Path
    ) -> CorpusMetadata | None:
        """
        Extract metadata from document content using LLM.

        Один вызов = минимальная стоимость.
        """
        # Strip existing frontmatter if present
        _, body = parse_yaml_frontmatter(content)
        content_for_llm = body.strip()

        dialogue = Dialogue(
            [
                message.system(
                    text="Ты классификатор документов. Отвечай точно и конкретно, используя русский язык для заголовков и тегов."
                ),
                message.user(text=EXTRACTION_PROMPT.format(content=content_for_llm)),
            ]
        )

        result = await dialogue.interpret(self.provider, ExtractedMetadata)

        match result:
            case Ok(extracted):
                doc_id = generate_doc_id(file_path)
                doc_type = extracted.doc_type
                citation_key = generate_citation_key(file_path, doc_type)
                priority = DOC_TYPE_PRIORITIES.get(doc_type, 0)

                return CorpusMetadata(
                    doc_id=doc_id,
                    title=extracted.title,
                    doc_type=doc_type,
                    effective_date=extracted.effective_date,
                    citation_key=citation_key,
                    priority=priority,
                    tags=extracted.tags,
                )
            case Error(e):
                print(f"Failed to extract metadata from {file_path}: {e}")
                return None

    def write_frontmatter(self, file_path: Path, metadata: CorpusMetadata) -> None:
        """Write YAML frontmatter to file, replacing existing if present."""
        content = file_path.read_text(encoding="utf-8")

        # Remove existing frontmatter if present
        _, body = parse_yaml_frontmatter(content)
        content = body.strip()

        # Build frontmatter dict
        fm_dict: dict[str, Any] = {
            "doc_id": metadata.doc_id,
            "title": metadata.title,
            "doc_type": metadata.doc_type,
            "citation_key": metadata.citation_key,
            "priority": metadata.priority,
        }

        if metadata.effective_date:
            fm_dict["effective_date"] = metadata.effective_date.isoformat()

        if metadata.tags:
            fm_dict["tags"] = metadata.tags

        # Format YAML
        yaml_str = yaml.dump(
            fm_dict,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )

        # Prepend frontmatter
        new_content = f"---\n{yaml_str}---\n\n{content}"
        file_path.write_text(new_content, encoding="utf-8")

    async def annotate_file(self, file_path: Path) -> bool:
        """
        Annotate a single file.

        Returns True if successful, False otherwise.
        """
        print(f"  → Processing: {file_path.name}")
        try:
            content = file_path.read_text(encoding="utf-8")

            fixed_content, was_fixed = fix_encoding(content)
            if was_fixed:
                file_path.write_text(fixed_content, encoding="utf-8")
                content = fixed_content
                print(f"    Fixed encoding: {file_path.name}")

            metadata = await self.extract_metadata(content, file_path)

            if metadata is None:
                print(f"  ✗ Failed: {file_path.name} (no metadata)")
                return False

            self.write_frontmatter(file_path, metadata)
            print(f"  ✓ Done: {file_path.name}")
            return True

        except Exception as e:
            print(f"  ✗ Error: {file_path.name}: {e}")
            return False

    def _make_annotate_interp(self, file_path: Path) -> LazyCoroResult[bool, str]:
        """Wrap annotate_file as Interp for use with combinators.batch."""
        return L.catching_async(
            lambda: self.annotate_file(file_path),
            on_error=str,
        )

    def fix_corpus_encoding(
        self,
        corpus_path: Path,
        dry_run: bool = False,
    ) -> tuple[int, int]:
        """
        Fix mojibake encoding in all corpus files.

        This repairs files where UTF-8 content was incorrectly saved as Latin-1.

        Args:
            corpus_path: Path to corpus directory
            dry_run: If True, only report what would be fixed

        Returns:
            (fixed_count, skipped_count)
        """
        fixed = 0
        skipped = 0

        for ext in CORPUS_EXTENSIONS:
            for file_path in corpus_path.rglob(ext):
                try:
                    content = file_path.read_text(encoding="utf-8")
                    fixed_content, was_fixed = fix_encoding(content)

                    if was_fixed:
                        if dry_run:
                            print(f"  Would fix: {file_path.name}")
                        else:
                            file_path.write_text(fixed_content, encoding="utf-8")
                            print(f"  ✓ Fixed: {file_path.name}")
                        fixed += 1
                    else:
                        skipped += 1

                except Exception as e:
                    print(f"  ✗ Error reading {file_path.name}: {e}")
                    skipped += 1

        print(f"\nEncoding fix: {fixed} files fixed, {skipped} already OK.")
        return fixed, skipped

    async def annotate_corpus(
        self,
        corpus_path: Path,
        batch_size: int = 5,
        dry_run: bool = False,
        force: bool = False,
    ) -> tuple[int, int]:
        """
        Annotate all unannotated files in corpus.

        Uses parallel file scanning for faster discovery on large corpora.

        Args:
            corpus_path: Path to corpus directory
            batch_size: Number of concurrent LLM calls
            dry_run: If True, only print what would be done
            force: If True, re-annotate all files (including already annotated)

        Returns:
            (success_count, failure_count)
        """
        if force:
            files = self.find_all_files(corpus_path)
        else:
            # Use async version for parallel file scanning
            files = await self.find_unannotated_async(corpus_path)

        if not files:
            print("No files to annotate.")
            return 0, 0

        if force:
            print(f"Found {len(files)} files (force re-annotation).")
        else:
            print(f"Found {len(files)} unannotated files.")

        if dry_run:
            for f in files:
                print(f"  Would annotate: {f}")
            return 0, 0

        success = 0
        failed = 0

        # Rate-limited handler to avoid overwhelming the API
        rate_policy = RateLimitPolicy(max_per_second=2.0, burst=5)

        def make_rate_limited_interp(file_path: Path) -> LazyCoroResult[bool, str]:
            return rate_limit(
                self._make_annotate_interp(file_path),
                policy=rate_policy,
            )

        batch_result = await combinators.batch_all(
            items=files,
            handler=make_rate_limited_interp,
            concurrency=batch_size,
        )

        for result in batch_result.unwrap():
            match result:
                case Ok(True):
                    success += 1
                case Ok(False) | Ok(_):
                    failed += 1
                case Error(_):
                    failed += 1

        print(f"\nDone: {success} annotated, {failed} failed.")
        return success, failed


__all__ = ["CorpusAnnotator", "ExtractedMetadata", "fix_encoding"]
