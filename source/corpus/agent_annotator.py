"""
Agent-based corpus annotator with tools.

Uses funcai agent with specialized tools for document analysis:
- Extract document header
- Find dates in document
- Search in document
- Validate findings
"""

import asyncio
import re
from contextvars import ContextVar
from datetime import date
from pathlib import Path
from typing import Any, Literal

import combinators
import yaml
from combinators import lift as L
from combinators import rate_limit
from combinators.concurrency import RateLimitPolicy
from kungfu import Error, LazyCoroResult, Ok
from pydantic import BaseModel, Field, field_validator

from funcai.agents.agent import agent
from funcai.agents.tool import Tool, tool
from funcai.core import message
from funcai.core.dialogue import Dialogue
from funcai.core.provider import ABCAIProvider
from source.contracts.corpus import CorpusMetadata
from source.corpus.annotator import fix_encoding
from source.corpus.constants import CORPUS_EXTENSIONS
from source.corpus.metadata import (
    DOC_TYPE_PRIORITIES,
    generate_citation_key,
    generate_doc_id,
    parse_yaml_frontmatter,
)

# ==============================================================================
# Constants
# ==============================================================================

DATE_FORMAT = "YYYY-MM-DD"
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Russian date patterns
RUSSIAN_DATE_PATTERNS = [
    # "31 марта 2025 года" or "31 марта 2025"
    re.compile(
        r"(\d{1,2})\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+(\d{4})\s*(года|г\.?)?",
        re.IGNORECASE,
    ),
    # "2025-03-31" ISO format
    re.compile(r"(\d{4})-(\d{2})-(\d{2})"),
    # "7–8 ноября 2020 г."
    re.compile(
        r"(\d{1,2})[-–](\d{1,2})\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+(\d{4})",
        re.IGNORECASE,
    ),
]

MONTH_MAP = {
    "января": 1,
    "февраля": 2,
    "марта": 3,
    "апреля": 4,
    "мая": 5,
    "июня": 6,
    "июля": 7,
    "августа": 8,
    "сентября": 9,
    "октября": 10,
    "ноября": 11,
    "декабря": 12,
}

# Context var for current document
_current_document: ContextVar[str] = ContextVar("current_document", default="")
_current_filename: ContextVar[str] = ContextVar("current_filename", default="")

# ==============================================================================
# Tool results
# ==============================================================================


class DocumentHeader(BaseModel):
    """Result of extracting document header."""

    first_lines: str = Field(description="Первые строки документа")
    total_lines: int = Field(description="Общее количество строк в документе")
    total_chars: int = Field(description="Общее количество символов")


class DateMatch(BaseModel):
    """A date found in the document."""

    date_text: str = Field(description="Текст даты как в документе")
    date_iso: str | None = Field(description="Дата в формате YYYY-MM-DD или null")
    context: str = Field(description="Контекст где найдена дата (до 100 символов)")
    position: int = Field(description="Позиция в документе (символ)")


class DatesFound(BaseModel):
    """Result of finding dates in document."""

    dates: list[DateMatch] = Field(description="Найденные даты")
    count: int = Field(description="Количество найденных дат")


class SearchResult(BaseModel):
    """Result of searching in document."""

    found: bool = Field(description="Найден ли текст")
    matches: list[str] = Field(description="Найденные фрагменты с контекстом")
    count: int = Field(description="Количество совпадений")


class DocumentSection(BaseModel):
    """A section of the document."""

    content: str = Field(description="Содержимое секции")
    start_line: int = Field(description="Начальная строка")
    end_line: int = Field(description="Конечная строка")


# ==============================================================================
# Extraction schema
# ==============================================================================


class ExtractedMetadataAgent(BaseModel):
    """Final extraction result."""

    title: str = Field(
        description="Заголовок документа (из текста документа, не придуманный)"
    )
    doc_type: Literal[
        "charter",
        "regulations",
        "code",
        "decision",
        "precedent",
        "clarification",
    ] = Field(description="Тип документа")
    effective_date: str | None = Field(
        default=None,
        description=f"Дата в формате {DATE_FORMAT} или null",
    )
    tags: list[str] = Field(description="3-5 тегов для категоризации")
    reasoning: str = Field(description="Обоснование выбора типа документа")

    @field_validator("effective_date")
    @classmethod
    def validate_date_format(cls, v: str | None) -> str | None:
        if v is None or v == "null" or v == "":
            return None
        if not DATE_PATTERN.match(v):
            return None
        return v


# ==============================================================================
# Tools for document analysis
# ==============================================================================


def _parse_russian_date(text: str) -> str | None:
    """Try to parse Russian date to ISO format."""
    text = text.lower().strip()

    # Try "31 марта 2025"
    for pattern in RUSSIAN_DATE_PATTERNS:
        match = pattern.search(text)
        if match:
            groups = match.groups()

            # ISO format: 2025-03-31
            if len(groups) == 3 and groups[0].isdigit() and len(groups[0]) == 4:
                return f"{groups[0]}-{groups[1]}-{groups[2]}"

            # Russian format: 31 марта 2025
            if len(groups) >= 3:
                day = groups[0]
                month_name = groups[1] if not groups[1].isdigit() else groups[2]
                year = (
                    groups[-1]
                    if groups[-1].isdigit() and len(groups[-1]) == 4
                    else groups[2]
                )

                if month_name.lower() in MONTH_MAP:
                    month = MONTH_MAP[month_name.lower()]
                    try:
                        return f"{year}-{month:02d}-{int(day):02d}"
                    except (ValueError, TypeError):
                        pass

    return None


def make_annotator_tools() -> list[Tool]:
    """Create tools for document annotation."""

    @tool("Извлечь заголовок документа (первые N строк)")
    def get_document_header(num_lines: int = 30) -> DocumentHeader:
        """
        Получить первые строки документа где обычно находится заголовок.

        Args:
            num_lines: Количество строк (по умолчанию 30)

        Returns:
            Первые строки документа, общее количество строк и символов.
        """
        content = _current_document.get()
        lines = content.split("\n")

        header_lines = lines[:num_lines]
        header_text = "\n".join(header_lines)

        return DocumentHeader(
            first_lines=header_text,
            total_lines=len(lines),
            total_chars=len(content),
        )

    @tool("Найти все даты в документе")
    def find_dates_in_document() -> DatesFound:
        """
        Найти все даты в документе с контекстом.

        Returns:
            Список найденных дат с контекстом и ISO-форматом.
        """
        content = _current_document.get()
        dates: list[DateMatch] = []

        for pattern in RUSSIAN_DATE_PATTERNS:
            for match in pattern.finditer(content):
                date_text = match.group(0)
                pos = match.start()

                # Get context (50 chars before and after)
                start = max(0, pos - 50)
                end = min(len(content), pos + len(date_text) + 50)
                context = content[start:end].replace("\n", " ").strip()

                # Try to parse to ISO
                date_iso = _parse_russian_date(date_text)

                dates.append(
                    DateMatch(
                        date_text=date_text,
                        date_iso=date_iso,
                        context=context,
                        position=pos,
                    )
                )

        # Deduplicate by date_text
        seen: set[str] = set()
        unique_dates: list[DateMatch] = []
        for d in dates:
            if d.date_text not in seen:
                seen.add(d.date_text)
                unique_dates.append(d)

        return DatesFound(dates=unique_dates, count=len(unique_dates))

    @tool("Поиск текста в документе")
    def search_in_document(query: str, case_sensitive: bool = False) -> SearchResult:
        """
        Найти текст в документе и вернуть контекст.

        Args:
            query: Текст для поиска
            case_sensitive: Учитывать регистр (по умолчанию нет)

        Returns:
            Найденные фрагменты с контекстом.
        """
        content = _current_document.get()

        if not case_sensitive:
            search_content = content.lower()
            search_query = query.lower()
        else:
            search_content = content
            search_query = query

        matches: list[str] = []
        start = 0

        while True:
            pos = search_content.find(search_query, start)
            if pos == -1:
                break

            # Get context (80 chars before and after)
            ctx_start = max(0, pos - 80)
            ctx_end = min(len(content), pos + len(query) + 80)
            context = content[ctx_start:ctx_end].replace("\n", " ").strip()
            matches.append(f"...{context}...")

            start = pos + 1

            # Limit to 5 matches
            if len(matches) >= 5:
                break

        return SearchResult(
            found=len(matches) > 0,
            matches=matches,
            count=len(matches),
        )

    @tool("Получить секцию документа по номерам строк")
    def get_document_section(start_line: int, end_line: int) -> DocumentSection:
        """
        Получить часть документа по номерам строк.

        Args:
            start_line: Начальная строка (с 1)
            end_line: Конечная строка

        Returns:
            Содержимое указанной секции.
        """
        content = _current_document.get()
        lines = content.split("\n")

        # Adjust for 1-based indexing
        start = max(0, start_line - 1)
        end = min(len(lines), end_line)

        section_lines = lines[start:end]
        section_text = "\n".join(section_lines)

        return DocumentSection(
            content=section_text,
            start_line=start + 1,
            end_line=end,
        )

    @tool("Получить имя файла документа")
    def get_filename() -> str:
        """
        Получить имя файла текущего документа.

        Returns:
            Имя файла (например: Libertarian_Party_Charter.md)
        """
        return _current_filename.get()

    return [
        get_document_header,
        find_dates_in_document,
        search_in_document,
        get_document_section,
        get_filename,
    ]


# ==============================================================================
# Prompts
# ==============================================================================

SYSTEM_PROMPT = """\
Ты — эксперт по классификации нормативных документов.

Твоя задача: определить тип документа и извлечь метаданные.

## Типы документов

- **charter**: Устав организации/партии — основной учредительный документ
- **regulations**: Регламент, положение, порядок, политика — правила и процедуры
- **code**: Кодекс (этический, дисциплинарный) — свод норм поведения
- **decision**: Решение, постановление, протокол заседания — решение органа
- **precedent**: Прецедентное решение по конкретному делу
- **clarification**: Разъяснение, толкование норм

## Инструкции

1. Используй инструменты для анализа документа
2. Начни с get_document_header чтобы увидеть начало документа
3. Используй find_dates_in_document чтобы найти даты
4. Если нужно — используй search_in_document для поиска
5. После анализа — дай финальный ответ

## Важно

- Заголовок ДОЛЖЕН быть взят ИЗ ТЕКСТА документа
- Если документ начинается с "Устав..." — это charter
- Если это протокол заседания — это decision
- Дата должна быть в формате YYYY-MM-DD"""


USER_PROMPT = """\
Проанализируй документ и извлеки метаданные.

Используй доступные инструменты:
1. get_document_header() - посмотреть начало документа
2. find_dates_in_document() - найти все даты
3. search_in_document(query) - найти текст
4. get_document_section(start, end) - получить часть документа
5. get_filename() - узнать имя файла

После анализа дай финальный ответ с полями:
- title: заголовок из документа
- doc_type: тип документа
- effective_date: дата в формате YYYY-MM-DD или null
- tags: 3-5 тегов
- reasoning: почему выбран этот тип"""


# ==============================================================================
# Agent Annotator
# ==============================================================================


class AgentAnnotator:
    """
    Annotator with tools for document analysis.

    Uses funcai agent with specialized tools to:
    - Extract document header
    - Find dates
    - Search in document
    - Make informed decisions about metadata
    """

    def __init__(
        self,
        provider: ABCAIProvider[Any],
        max_steps: int = 10,
        debug: bool = False,
    ):
        self.provider = provider
        self.max_steps = max_steps
        self.debug = debug
        self.tools = make_annotator_tools()

    def _log(self, msg: str) -> None:
        if self.debug:
            print(f"    [agent] {msg}")

    async def extract_metadata(
        self,
        content: str,
        file_path: Path,
    ) -> CorpusMetadata | None:
        """
        Extract metadata using agent with tools.
        """
        _, body = parse_yaml_frontmatter(content)
        content_text = body.strip()

        # Set context vars for tools
        _current_document.set(content_text)
        _current_filename.set(file_path.name)

        self._log(f"Processing: {file_path.name} ({len(content_text)} chars)")

        dialogue = Dialogue(
            [
                message.system(text=SYSTEM_PROMPT),
                message.user(text=USER_PROMPT),
            ]
        )

        result = await agent(
            dialogue,
            self.provider,
            self.tools,
            max_steps=self.max_steps,
            schema=ExtractedMetadataAgent,
        )

        match result:
            case Ok(response):
                extracted = response.parsed
                self._log(f"Result: {extracted.title} ({extracted.doc_type})")
                self._log(f"Reasoning: {extracted.reasoning}")
            case Error(e):
                self._log(f"Agent failed: {e}")
                return None

        # Parse date
        effective_date: date | None = None
        if extracted.effective_date and DATE_PATTERN.match(extracted.effective_date):
            try:
                parts = extracted.effective_date.split("-")
                effective_date = date(int(parts[0]), int(parts[1]), int(parts[2]))
            except (ValueError, IndexError):
                pass

        # Build final metadata
        doc_id = generate_doc_id(file_path)
        citation_key = generate_citation_key(file_path, extracted.doc_type)
        priority = DOC_TYPE_PRIORITIES.get(extracted.doc_type, 0)

        return CorpusMetadata(
            doc_id=doc_id,
            title=extracted.title,
            doc_type=extracted.doc_type,
            effective_date=effective_date,
            citation_key=citation_key,
            priority=priority,
            tags=extracted.tags,
        )

    def find_unannotated(self, corpus_path: Path) -> list[Path]:
        """Find files without YAML frontmatter."""
        unannotated: list[Path] = []

        for ext in CORPUS_EXTENSIONS:
            for file_path in corpus_path.rglob(ext):
                content = file_path.read_text(encoding="utf-8")
                metadata, _ = parse_yaml_frontmatter(content)
                if metadata is None:
                    unannotated.append(file_path)

        return sorted(unannotated)

    def find_all_files(self, corpus_path: Path) -> list[Path]:
        """Find all corpus files."""
        files: list[Path] = []

        for ext in CORPUS_EXTENSIONS:
            for file_path in corpus_path.rglob(ext):
                files.append(file_path)

        return sorted(files)

    async def find_unannotated_async(
        self, corpus_path: Path, concurrency: int = 20
    ) -> list[Path]:
        """
        Find files without YAML frontmatter using parallel file reading.

        Args:
            corpus_path: Path to corpus directory
            concurrency: Number of files to read in parallel

        Returns:
            Sorted list of paths to unannotated files
        """
        all_files: list[Path] = []
        for ext in CORPUS_EXTENSIONS:
            all_files.extend(corpus_path.rglob(ext))

        if not all_files:
            return []

        async def check_file(path: Path) -> bool:
            content = await asyncio.to_thread(path.read_text, encoding="utf-8")
            metadata, _ = parse_yaml_frontmatter(content)
            return metadata is None

        def make_check_interp(path: Path) -> LazyCoroResult[bool, str]:
            return L.catching_async(lambda: check_file(path), on_error=str)

        batch_result = await combinators.batch_all(
            items=all_files,
            handler=make_check_interp,
            concurrency=concurrency,
        )

        unannotated: list[Path] = []
        for path, result in zip(all_files, batch_result.unwrap()):
            match result:
                case Ok():
                    unannotated.append(path)
                case Error(_):
                    pass

        return sorted(unannotated)

    def write_frontmatter(self, file_path: Path, metadata: CorpusMetadata) -> None:
        """Write YAML frontmatter to file."""
        content = file_path.read_text(encoding="utf-8")
        _, body = parse_yaml_frontmatter(content)
        content = body.strip()

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

        yaml_str = yaml.dump(
            fm_dict,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )

        new_content = f"---\n{yaml_str}---\n\n{content}"
        file_path.write_text(new_content, encoding="utf-8")

    async def annotate_file(self, file_path: Path) -> bool:
        """Annotate a single file."""
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
            print(f"  ✓ Done: {file_path.name} → {metadata.doc_type}")
            return True

        except Exception as e:
            print(f"  ✗ Error: {file_path.name}: {e}")
            return False

    def _make_annotate_interp(self, file_path: Path) -> LazyCoroResult[bool, str]:
        """Wrap annotate_file for batch processing."""
        return L.catching_async(
            lambda: self.annotate_file(file_path),
            on_error=str,
        )

    async def annotate_corpus(
        self,
        corpus_path: Path,
        batch_size: int = 1,
        dry_run: bool = False,
        force: bool = False,
    ) -> tuple[int, int]:
        """
        Annotate corpus files using agent with tools.

        Uses parallel file scanning for faster discovery.
        batch_size=1 recommended because agent makes multiple tool calls.
        """
        if force:
            files = self.find_all_files(corpus_path)
        else:
            files = await self.find_unannotated_async(corpus_path)

        if not files:
            print("No files to annotate.")
            return 0, 0

        mode = "force re-annotation" if force else "unannotated"
        print(f"Found {len(files)} files ({mode}).")
        print("Using agent mode with tools (header, dates, search)")

        if dry_run:
            for f in files:
                print(f"  Would annotate: {f}")
            return 0, 0

        success = 0
        failed = 0

        # Rate-limited handler - agent mode is more expensive, so lower rate
        rate_policy = RateLimitPolicy(max_per_second=1.0, burst=2)

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
                case _:
                    failed += 1

        print(f"\nDone: {success} annotated, {failed} failed.")
        return success, failed


__all__ = ["AgentAnnotator", "ExtractedMetadataAgent", "DATE_FORMAT"]
