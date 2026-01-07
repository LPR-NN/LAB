"""Document chunking for vector search."""

import re
from dataclasses import dataclass
from typing import Literal

from source.contracts.corpus import CorpusDocument


def clean_text_for_embedding(text: str) -> str:
    """
    Clean text from noise before embedding.

    Removes:
    - Pandoc/Quarto div markers (:::, ::::::, etc)
    - Span markers with classes [{text}]{.class}
    - HTML-like ID attributes {#id .class}
    - Repeated dots from PDF table of contents
    - Excessive whitespace

    Args:
        text: Raw text with potential markup noise

    Returns:
        Cleaned text suitable for embedding
    """
    # Remove Pandoc div markers (lines with only colons and optional attributes)
    # Matches lines like ":::::: {#content .content}" or ":::"
    text = re.sub(r"^:+\s*(?:\{[^}]*\})?\s*$", "", text, flags=re.MULTILINE)

    # Remove inline ID/class attributes {#id .class}
    # But preserve header content like "## Header {#id}"
    text = re.sub(r"\s*\{#[^}]+\}", "", text)

    # Remove span markers [{text}]{.class} -> text
    # Matches patterns like "[(subtitle text)]{.subtitle}"
    text = re.sub(r"\[([^\]]*)\]\{[^}]+\}", r"\1", text)

    # Remove repeated dots from PDF table of contents
    # Matches sequences of dots with spaces ". . . . . ."
    text = re.sub(r"(?:\.\s*){3,}", " ", text)

    # Remove standalone page numbers (lines with just numbers)
    text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)

    # Collapse multiple newlines to maximum two
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse multiple spaces to single space
    text = re.sub(r"[ \t]+", " ", text)

    # Strip leading/trailing whitespace from lines
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Remove empty lines at start/end
    text = text.strip()

    return text


@dataclass
class Chunk:
    """A chunk of document content with metadata."""

    chunk_id: str
    doc_id: str
    citation_key: str
    doc_type: str
    priority: int
    title: str
    section: str | None
    content: str
    start_char: int
    end_char: int

    def to_metadata(self) -> dict[str, str | int | None]:
        """Convert to metadata dict for vector DB."""
        return {
            "doc_id": self.doc_id,
            "citation_key": self.citation_key,
            "doc_type": self.doc_type,
            "priority": self.priority,
            "title": self.title,
            "section": self.section,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }


# Regex patterns for section detection
SECTION_PATTERNS = [
    # Numbered sections: "1.2.3" or "1.2.3."
    r"^(\d+(?:\.\d+)*\.?)\s+",
    # Article format: "Статья 5" or "Статья 5."
    r"^(Статья\s+\d+\.?)\s+",
    # Section format: "Раздел 1" or "Глава 2"
    r"^((?:Раздел|Глава)\s+\d+\.?)\s+",
    # Пункт format: "п. 1.2" or "пункт 3"
    r"^((?:п\.|пункт)\s*\d+(?:\.\d+)*\.?)\s+",
]


@dataclass
class DocumentChunker:
    """
    Splits documents into chunks for vector indexing.

    Supports multiple strategies:
    - 'section': Split by document sections (headers, numbered items)
    - 'paragraph': Split by paragraphs (double newlines)
    - 'fixed': Fixed-size chunks with overlap

    For legal documents, 'section' is recommended as it preserves
    logical boundaries.
    """

    strategy: Literal["section", "paragraph", "fixed"] = "section"
    max_chunk_size: int = 1500
    min_chunk_size: int = 100
    overlap: int = 100
    include_title_in_chunk: bool = True

    def chunk_document(self, document: CorpusDocument) -> list[Chunk]:
        """
        Split a document into chunks.

        Args:
            document: Document to chunk

        Returns:
            List of Chunk objects
        """
        if self.strategy == "section":
            return self._chunk_by_section(document)
        elif self.strategy == "paragraph":
            return self._chunk_by_paragraph(document)
        else:
            return self._chunk_fixed(document)

    def _chunk_by_section(self, document: CorpusDocument) -> list[Chunk]:
        """Split document by sections."""
        content = document.content
        chunks: list[Chunk] = []

        # Find all section boundaries
        boundaries = self._find_section_boundaries(content)

        if not boundaries:
            # No sections found, fall back to paragraph chunking
            return self._chunk_by_paragraph(document)

        # Add document end as final boundary
        boundaries.append((len(content), None))

        # Accumulator for merging small sections
        accumulated_content = ""
        accumulated_start = 0
        first_section_name: str | None = None

        for i, (start, section_name) in enumerate(boundaries[:-1]):
            end = boundaries[i + 1][0]
            section_content = content[start:end].strip()

            # Accumulate small sections instead of skipping them
            if len(section_content) < self.min_chunk_size:
                if not accumulated_content:
                    accumulated_start = start
                    first_section_name = section_name
                accumulated_content += (
                    "\n\n" + section_content if accumulated_content else section_content
                )
                continue

            # If we have accumulated content, merge with current section
            if accumulated_content:
                section_content = accumulated_content + "\n\n" + section_content
                start = accumulated_start
                section_name = first_section_name
                accumulated_content = ""
                first_section_name = None

            # If section is too large, split it further
            if len(section_content) > self.max_chunk_size:
                sub_chunks = self._split_large_section(
                    section_content,
                    document,
                    section_name,
                    start,
                )
                chunks.extend(sub_chunks)
            else:
                chunk_content = clean_text_for_embedding(section_content)
                if self.include_title_in_chunk:
                    chunk_content = f"{document.metadata.title}\n\n{chunk_content}"

                chunk_id = f"{document.doc_id}:{len(chunks)}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        doc_id=document.doc_id,
                        citation_key=document.citation_key,
                        doc_type=document.doc_type,
                        priority=document.priority,
                        title=document.metadata.title,
                        section=section_name,
                        content=chunk_content,
                        start_char=start,
                        end_char=end,
                    )
                )

        # Don't forget any remaining accumulated content
        if accumulated_content and len(accumulated_content) >= self.min_chunk_size:
            chunk_content = clean_text_for_embedding(accumulated_content)
            if self.include_title_in_chunk:
                chunk_content = f"{document.metadata.title}\n\n{chunk_content}"

            chunk_id = f"{document.doc_id}:{len(chunks)}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=document.doc_id,
                    citation_key=document.citation_key,
                    doc_type=document.doc_type,
                    priority=document.priority,
                    title=document.metadata.title,
                    section=first_section_name,
                    content=chunk_content,
                    start_char=accumulated_start,
                    end_char=len(content),
                )
            )

        # If no chunks were created, create one from entire document
        if not chunks:
            chunks = self._chunk_fixed(document)

        return chunks

    def _find_section_boundaries(self, content: str) -> list[tuple[int, str | None]]:
        """Find section boundaries in content."""
        boundaries: list[tuple[int, str | None]] = []

        lines = content.split("\n")
        current_pos = 0

        for line in lines:
            stripped = line.strip()

            for pattern in SECTION_PATTERNS:
                match = re.match(pattern, stripped)
                if match:
                    section_name = match.group(1).strip()
                    boundaries.append((current_pos, section_name))
                    break

            # Also detect markdown headers
            if stripped.startswith("#"):
                header_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
                if header_match:
                    section_name = header_match.group(2).strip()
                    boundaries.append((current_pos, section_name))

            current_pos += len(line) + 1  # +1 for newline

        return boundaries

    def _split_large_section(
        self,
        content: str,
        document: CorpusDocument,
        section_name: str | None,
        base_offset: int,
    ) -> list[Chunk]:
        """Split a large section into smaller chunks."""
        chunks: list[Chunk] = []
        paragraphs = re.split(r"\n\n+", content)

        current_chunk = ""
        current_start = 0
        chunk_idx = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Would adding this paragraph exceed max size?
            if (
                current_chunk
                and len(current_chunk) + len(para) + 2 > self.max_chunk_size
            ):
                # Save current chunk
                chunk_content = clean_text_for_embedding(current_chunk)
                if self.include_title_in_chunk:
                    chunk_content = f"{document.metadata.title}\n\n{chunk_content}"

                chunk_id = f"{document.doc_id}:{section_name or 'sec'}:{chunk_idx}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        doc_id=document.doc_id,
                        citation_key=document.citation_key,
                        doc_type=document.doc_type,
                        priority=document.priority,
                        title=document.metadata.title,
                        section=section_name,
                        content=chunk_content,
                        start_char=base_offset + current_start,
                        end_char=base_offset + current_start + len(current_chunk),
                    )
                )
                chunk_idx += 1

                # Start new chunk with overlap
                if self.overlap > 0 and len(current_chunk) > self.overlap:
                    overlap_text = current_chunk[-self.overlap :]
                    current_chunk = overlap_text + "\n\n" + para
                    current_start = (
                        current_start
                        + len(current_chunk)
                        - len(overlap_text)
                        - len(para)
                        - 2
                    )
                else:
                    current_chunk = para
                    current_start = content.find(para, current_start)
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                    current_start = content.find(para)

        # Don't forget the last chunk
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunk_content = clean_text_for_embedding(current_chunk)
            if self.include_title_in_chunk:
                chunk_content = f"{document.metadata.title}\n\n{chunk_content}"

            chunk_id = f"{document.doc_id}:{section_name or 'sec'}:{chunk_idx}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=document.doc_id,
                    citation_key=document.citation_key,
                    doc_type=document.doc_type,
                    priority=document.priority,
                    title=document.metadata.title,
                    section=section_name,
                    content=chunk_content,
                    start_char=base_offset + current_start,
                    end_char=base_offset + current_start + len(current_chunk),
                )
            )

        return chunks

    def _chunk_by_paragraph(self, document: CorpusDocument) -> list[Chunk]:
        """Split document by paragraphs, merging small ones."""
        content = document.content
        chunks: list[Chunk] = []

        paragraphs = re.split(r"\n\n+", content)
        current_chunk = ""
        current_start = 0
        chunk_idx = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if (
                current_chunk
                and len(current_chunk) + len(para) + 2 > self.max_chunk_size
            ):
                # Save current chunk
                chunk_content = clean_text_for_embedding(current_chunk)
                if self.include_title_in_chunk:
                    chunk_content = f"{document.metadata.title}\n\n{chunk_content}"

                chunk_id = f"{document.doc_id}:{chunk_idx}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        doc_id=document.doc_id,
                        citation_key=document.citation_key,
                        doc_type=document.doc_type,
                        priority=document.priority,
                        title=document.metadata.title,
                        section=None,
                        content=chunk_content,
                        start_char=current_start,
                        end_char=current_start + len(current_chunk),
                    )
                )
                chunk_idx += 1
                current_chunk = para
                current_start = content.find(para, current_start + 1)
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                    current_start = content.find(para)

        # Last chunk
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunk_content = clean_text_for_embedding(current_chunk)
            if self.include_title_in_chunk:
                chunk_content = f"{document.metadata.title}\n\n{chunk_content}"

            chunk_id = f"{document.doc_id}:{chunk_idx}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=document.doc_id,
                    citation_key=document.citation_key,
                    doc_type=document.doc_type,
                    priority=document.priority,
                    title=document.metadata.title,
                    section=None,
                    content=chunk_content,
                    start_char=current_start,
                    end_char=current_start + len(current_chunk),
                )
            )

        return chunks if chunks else self._chunk_fixed(document)

    def _chunk_fixed(self, document: CorpusDocument) -> list[Chunk]:
        """Split document into fixed-size chunks with overlap."""
        content = document.content
        chunks: list[Chunk] = []

        # If document is small enough, return as single chunk
        if len(content) <= self.max_chunk_size:
            chunk_content = clean_text_for_embedding(content)
            if self.include_title_in_chunk:
                chunk_content = f"{document.metadata.title}\n\n{chunk_content}"

            return [
                Chunk(
                    chunk_id=f"{document.doc_id}:0",
                    doc_id=document.doc_id,
                    citation_key=document.citation_key,
                    doc_type=document.doc_type,
                    priority=document.priority,
                    title=document.metadata.title,
                    section=None,
                    content=chunk_content,
                    start_char=0,
                    end_char=len(content),
                )
            ]

        # Split with overlap
        step = self.max_chunk_size - self.overlap
        chunk_idx = 0

        for start in range(0, len(content), step):
            end = min(start + self.max_chunk_size, len(content))
            chunk_text = content[start:end].strip()

            if len(chunk_text) < self.min_chunk_size:
                continue

            chunk_content = clean_text_for_embedding(chunk_text)
            if self.include_title_in_chunk:
                chunk_content = f"{document.metadata.title}\n\n{chunk_content}"

            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}:{chunk_idx}",
                    doc_id=document.doc_id,
                    citation_key=document.citation_key,
                    doc_type=document.doc_type,
                    priority=document.priority,
                    title=document.metadata.title,
                    section=None,
                    content=chunk_content,
                    start_char=start,
                    end_char=end,
                )
            )
            chunk_idx += 1

        return chunks


__all__ = ["DocumentChunker", "Chunk", "clean_text_for_embedding"]
