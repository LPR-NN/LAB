"""Document viewer for markdown/text files."""

import html
import re
from pathlib import Path

VIEWER_CSS = """
:root {
    --bg-primary: #fdfcfa;
    --bg-secondary: #f5f3ef;
    --bg-accent: #e8e4dc;
    --text-primary: #1a1a1a;
    --text-secondary: #4a4a4a;
    --text-muted: #6b6b6b;
    --accent-color: #8b4513;
    --accent-light: #cd853f;
    --border-color: #d4d0c8;
    --code-bg: #f4f1ed;
    --highlight-bg: #fff3cd;
}

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: "Charter", "Georgia", serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    line-height: 1.8;
    padding: 0;
    margin: 0;
}

.container {
    max-width: 850px;
    margin: 0 auto;
    padding: 2.5rem 2rem;
}

.back-link {
    display: inline-block;
    margin-bottom: 1.5rem;
    color: var(--accent-color);
    text-decoration: none;
    font-size: 0.9rem;
}

.back-link:hover {
    text-decoration: underline;
}

.back-link::before {
    content: "← ";
}

/* Document header */
.doc-header {
    padding-bottom: 1.5rem;
    border-bottom: 2px solid var(--accent-color);
    margin-bottom: 2rem;
}

.doc-title {
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.75rem;
    line-height: 1.3;
}

.doc-meta {
    font-size: 0.85rem;
    color: var(--text-muted);
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    align-items: center;
}

.doc-type {
    background: var(--accent-color);
    color: white;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    text-transform: uppercase;
    font-size: 0.7rem;
    letter-spacing: 0.05em;
    font-weight: 600;
}

.doc-path {
    font-family: "JetBrains Mono", "Fira Code", monospace;
    font-size: 0.75rem;
    color: var(--text-muted);
}

/* Document content */
.doc-content {
    font-family: "Charter", "Georgia", serif;
    font-size: 1rem;
    line-height: 1.85;
}

.doc-content.preformatted {
    white-space: pre-wrap;
    font-family: "JetBrains Mono", "Fira Code", monospace;
    font-size: 0.9rem;
    line-height: 1.6;
    background: var(--bg-secondary);
    padding: 1.5rem;
    border-radius: 6px;
    border: 1px solid var(--border-color);
}

/* Headers */
.doc-content h1, .doc-content h2, .doc-content h3, .doc-content h4 {
    color: var(--text-primary);
    font-weight: 600;
    margin-top: 2rem;
    margin-bottom: 1rem;
    scroll-margin-top: 2rem;
}

.doc-content h1 {
    font-size: 1.4rem;
    border-bottom: 1px solid var(--border-color);
    padding-bottom: 0.5rem;
}

.doc-content h2 {
    font-size: 1.2rem;
}

.doc-content h3 {
    font-size: 1.1rem;
}

.doc-content h4 {
    font-size: 1rem;
}

/* Paragraphs and sections */
.doc-content p {
    margin-bottom: 1rem;
}

.doc-content .section-anchor {
    scroll-margin-top: 2rem;
}

/* Numbered sections (п. 2.3.2) */
.doc-content .numbered-section {
    padding: 0.75rem 1rem;
    margin: 0.75rem 0;
    background: var(--bg-secondary);
    border-left: 3px solid var(--accent-light);
    border-radius: 0 4px 4px 0;
    scroll-margin-top: 2rem;
}

.doc-content .section-number {
    font-weight: 600;
    color: var(--accent-color);
    font-family: "JetBrains Mono", "Fira Code", monospace;
    font-size: 0.9em;
}

/* Lists */
.doc-content ul, .doc-content ol {
    margin-left: 1.5rem;
    margin-bottom: 1rem;
}

.doc-content li {
    margin-bottom: 0.5rem;
}

/* Blockquotes */
.doc-content blockquote {
    border-left: 3px solid var(--accent-color);
    padding: 0.75rem 1rem;
    margin: 1rem 0;
    background: var(--bg-secondary);
    color: var(--text-secondary);
    font-style: italic;
    border-radius: 0 4px 4px 0;
}

/* Code */
.doc-content code {
    background: var(--code-bg);
    padding: 0.15rem 0.4rem;
    border-radius: 3px;
    font-family: "JetBrains Mono", "Fira Code", monospace;
    font-size: 0.85em;
}

.doc-content pre {
    background: var(--code-bg);
    padding: 1rem 1.25rem;
    border-radius: 6px;
    overflow-x: auto;
    margin: 1rem 0;
    border: 1px solid var(--border-color);
}

.doc-content pre code {
    padding: 0;
    background: none;
}

/* Links */
.doc-content a {
    color: var(--accent-color);
    text-decoration: none;
}

.doc-content a:hover {
    text-decoration: underline;
}

/* Strong/emphasis */
.doc-content strong {
    font-weight: 600;
}

.doc-content em {
    font-style: italic;
}

/* Target highlight animation */
:target {
    animation: highlight-fade 2s ease-out;
}

@keyframes highlight-fade {
    0% {
        background-color: var(--highlight-bg);
        box-shadow: 0 0 0 4px var(--highlight-bg);
    }
    100% {
        background-color: transparent;
        box-shadow: none;
    }
}

/* Highlighted text from search/target */
.highlighted {
    background-color: var(--highlight-bg);
    padding: 0.1rem 0.2rem;
    border-radius: 2px;
}

/* YAML frontmatter (hidden by default, can show with class) */
.frontmatter {
    display: none;
}

.show-frontmatter .frontmatter {
    display: block;
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    padding: 1rem;
    margin-bottom: 2rem;
    font-family: "JetBrains Mono", "Fira Code", monospace;
    font-size: 0.8rem;
    color: var(--text-muted);
}

/* Table of contents */
.toc {
    background: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    padding: 1rem 1.5rem;
    margin-bottom: 2rem;
}

.toc-title {
    font-weight: 600;
    color: var(--accent-color);
    margin-bottom: 0.5rem;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.toc ul {
    list-style: none;
    margin: 0;
    padding: 0;
}

.toc li {
    margin: 0.25rem 0;
}

.toc a {
    color: var(--text-secondary);
    text-decoration: none;
    font-size: 0.9rem;
}

.toc a:hover {
    color: var(--accent-color);
}

/* Print styles */
@media print {
    .back-link, .toc {
        display: none;
    }
    .container {
        max-width: 100%;
        padding: 1cm;
    }
}

/* Responsive */
@media (max-width: 600px) {
    .container {
        padding: 1.5rem 1rem;
    }
    .doc-title {
        font-size: 1.3rem;
    }
}
"""


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return html.escape(text)


def _make_anchor_id(text: str) -> str:
    """Create a URL-safe anchor ID from text."""
    # Replace common patterns
    text = text.lower()
    text = re.sub(r"[пП]\.\s*", "p-", text)  # п. -> p-
    text = re.sub(r"§\s*", "s-", text)  # § -> s-
    text = re.sub(r"[\s.,:;]+", "-", text)  # spaces and punctuation -> dash
    text = re.sub(r"[^\w-]", "", text)  # remove non-word chars
    text = re.sub(r"-+", "-", text)  # collapse multiple dashes
    text = text.strip("-")
    return text or "section"


def _process_numbered_sections(html_content: str) -> str:
    """
    Add anchors to numbered sections like п. 2.3.2.2 or 2.3.2.

    Wraps sections in divs with IDs for linking.
    """
    # Pattern for numbered sections at start of paragraph or line
    # Matches: "2.3.2.2" or "п. 2.3.2.2" or "п.2.3.2"
    pattern = r"(<p>|^|\n)((?:[пП]\.\s*)?(\d+(?:\.\d+)+))"

    def replace_section(match: re.Match[str]) -> str:
        prefix = match.group(1)
        full_match = match.group(2)
        number = match.group(3)
        anchor_id = f"p-{number.replace('.', '-')}"
        return f'{prefix}<span id="{anchor_id}" class="section-anchor numbered-section"><span class="section-number">{_escape_html(full_match)}</span>'

    # Process and close spans properly
    result = re.sub(pattern, replace_section, html_content)

    # Close unclosed spans at paragraph end
    result = re.sub(
        r'(<span class="section-anchor numbered-section">.*?)(</p>)',
        r"\1</span>\2",
        result,
    )

    return result


def _rich_markdown_to_html(content: str) -> str:
    """
    Convert markdown to HTML with section anchors and rich formatting.
    """
    lines = content.split("\n")
    html_parts: list[str] = []
    in_list = False
    list_type: str | None = None
    in_blockquote = False
    paragraph_buffer: list[str] = []
    pending_section_number: str | None = None  # For standalone section numbers
    toc_items: list[tuple[str, str, int]] = []  # (id, title, level)

    def flush_paragraph() -> None:
        nonlocal paragraph_buffer, pending_section_number
        if paragraph_buffer:
            text = " ".join(paragraph_buffer)
            text = _process_inline(text)

            # If we have a pending section number, wrap this paragraph with it
            if pending_section_number:
                anchor_id = f"p-{pending_section_number.replace('.', '-')}"
                html_parts.append(
                    f'<div id="{anchor_id}" class="numbered-section">'
                    f'<span class="section-number">{pending_section_number}</span> {text}</div>'
                )
                pending_section_number = None
            else:
                # Check for numbered section at start of text (inline format)
                section_match = re.match(
                    r"^((?:[пП]\.\s*)?(\d+(?:\.\d+)+))\s*(.*)", text
                )
                if section_match:
                    full_num = section_match.group(1)
                    number = section_match.group(2)
                    rest = section_match.group(3)
                    anchor_id = f"p-{number.replace('.', '-')}"
                    html_parts.append(
                        f'<div id="{anchor_id}" class="numbered-section">'
                        f'<span class="section-number">{full_num}</span> {rest}</div>'
                    )
                else:
                    html_parts.append(f"<p>{text}</p>")
            paragraph_buffer = []

    def _process_inline(text: str) -> str:
        """Process inline markdown."""
        # Escape HTML first
        text = _escape_html(text)

        # Links [text](url)
        text = re.sub(
            r"\[([^\]]+)\]\(([^)]+)\)",
            r'<a href="\2" target="_blank" rel="noopener">\1</a>',
            text,
        )

        # Bold **text**
        text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)

        # Italic *text*
        text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", text)

        # Inline code `code`
        text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)

        return text

    for line in lines:
        stripped = line.strip()

        # Empty line - close any open blocks
        if not stripped:
            flush_paragraph()
            if in_list:
                tag = "ul" if list_type == "ul" else "ol"
                html_parts.append(f"</{tag}>")
                in_list = False
                list_type = None
            if in_blockquote:
                html_parts.append("</blockquote>")
                in_blockquote = False
            continue

        # Check for standalone section number (like "3.4.1.2" on its own line)
        standalone_num_match = re.match(r"^(\d+(?:\.\d+)+)$", stripped)
        if standalone_num_match:
            flush_paragraph()
            pending_section_number = standalone_num_match.group(1)
            continue

        # Headers with anchors
        if stripped.startswith("####"):
            flush_paragraph()
            header_text = stripped[4:].strip()
            anchor_id = _make_anchor_id(header_text)
            text = _process_inline(header_text)
            html_parts.append(f'<h4 id="{anchor_id}">{text}</h4>')
            toc_items.append((anchor_id, header_text, 4))
            continue
        if stripped.startswith("###"):
            flush_paragraph()
            header_text = stripped[3:].strip()
            anchor_id = _make_anchor_id(header_text)
            text = _process_inline(header_text)
            html_parts.append(f'<h3 id="{anchor_id}">{text}</h3>')
            toc_items.append((anchor_id, header_text, 3))
            continue
        if stripped.startswith("##"):
            flush_paragraph()
            header_text = stripped[2:].strip()
            anchor_id = _make_anchor_id(header_text)
            text = _process_inline(header_text)
            html_parts.append(f'<h2 id="{anchor_id}">{text}</h2>')
            toc_items.append((anchor_id, header_text, 2))
            continue
        if stripped.startswith("#"):
            flush_paragraph()
            header_text = stripped[1:].strip()
            anchor_id = _make_anchor_id(header_text)
            text = _process_inline(header_text)
            html_parts.append(f'<h1 id="{anchor_id}">{text}</h1>')
            toc_items.append((anchor_id, header_text, 1))
            continue

        # Blockquote
        if stripped.startswith(">"):
            flush_paragraph()
            text = _process_inline(stripped[1:].strip())
            if not in_blockquote:
                html_parts.append("<blockquote>")
                in_blockquote = True
            html_parts.append(f"<p>{text}</p>")
            continue

        # Unordered list
        if stripped.startswith("- ") or stripped.startswith("* "):
            flush_paragraph()
            text = _process_inline(stripped[2:])
            if not in_list or list_type != "ul":
                if in_list:
                    tag = "ul" if list_type == "ul" else "ol"
                    html_parts.append(f"</{tag}>")
                html_parts.append("<ul>")
                in_list = True
                list_type = "ul"
            html_parts.append(f"<li>{text}</li>")
            continue

        # Ordered list
        if re.match(r"^\d+\.?\s+", stripped):
            flush_paragraph()
            text = re.sub(r"^\d+\.?\s+", "", stripped)
            text = _process_inline(text)
            if not in_list or list_type != "ol":
                if in_list:
                    tag = "ul" if list_type == "ul" else "ol"
                    html_parts.append(f"</{tag}>")
                html_parts.append("<ol>")
                in_list = True
                list_type = "ol"
            html_parts.append(f"<li>{text}</li>")
            continue

        # Close any open list before paragraph
        if in_list:
            tag = "ul" if list_type == "ul" else "ol"
            html_parts.append(f"</{tag}>")
            in_list = False
            list_type = None

        if in_blockquote:
            html_parts.append("</blockquote>")
            in_blockquote = False

        # Regular paragraph line
        paragraph_buffer.append(stripped)

    # Flush remaining content
    flush_paragraph()
    if in_list:
        tag = "ul" if list_type == "ul" else "ol"
        html_parts.append(f"</{tag}>")
    if in_blockquote:
        html_parts.append("</blockquote>")

    return "\n".join(html_parts)


def _extract_title_from_frontmatter(content: str) -> str | None:
    """Extract title from YAML frontmatter."""
    if not content.startswith("---"):
        return None

    parts = content.split("---", 2)
    if len(parts) < 3:
        return None

    frontmatter = parts[1]
    match = re.search(r"^title:\s*['\"]?(.+?)['\"]?\s*$", frontmatter, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None


def _strip_frontmatter(content: str) -> str:
    """Remove YAML frontmatter from content."""
    if not content.startswith("---"):
        return content

    parts = content.split("---", 2)
    if len(parts) < 3:
        return content

    return parts[2].strip()


def _get_doc_type_label(file_path: str) -> str:
    """Get document type label based on path."""
    path = file_path.lower()

    if "assets/" in path:
        return "Вложение"
    if "main-docs/" in path:
        if "charter" in path:
            return "Устав"
        if "principles" in path:
            return "Декларация"
        if "membership" in path:
            return "Положение"
        return "Документ"
    if "/ek/" in path:
        return "Решение ЭК"
    if "/fk/" in path:
        return "Решение ФК"
    if "posts/" in path:
        return "Решение"

    return "Документ"


def render_document_viewer(
    content: str,
    file_path: str,
    title: str | None = None,
    back_url: str = "/",
) -> str:
    """
    Render a document viewer HTML page.

    Args:
        content: Document content
        file_path: Original file path (for metadata)
        title: Custom title (optional, extracted from frontmatter or path)
        back_url: URL for back link

    Returns:
        HTML page string
    """
    # Try to extract title from frontmatter
    if not title:
        title = _extract_title_from_frontmatter(content)

    # Fallback to filename
    if not title:
        path = Path(file_path)
        title = path.stem.replace("_", " ").replace("-", " ")

    # Determine if markdown and render accordingly
    is_markdown = file_path.endswith((".md", ".markdown"))

    if is_markdown:
        # Strip frontmatter before rendering
        body_content = _strip_frontmatter(content)
        rendered_content = _rich_markdown_to_html(body_content)
        content_class = "doc-content markdown"
    else:
        rendered_content = _escape_html(content)
        content_class = "doc-content preformatted"

    doc_type = _get_doc_type_label(file_path)

    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{_escape_html(title)} — Просмотр документа</title>
    <style>
{VIEWER_CSS}
    </style>
</head>
<body>
    <div class="container">
        <a href="{_escape_html(back_url)}" class="back-link">Назад</a>

        <header class="doc-header">
            <h1 class="doc-title">{_escape_html(title)}</h1>
            <div class="doc-meta">
                <span class="doc-type">{_escape_html(doc_type)}</span>
                <span class="doc-path">{_escape_html(file_path)}</span>
            </div>
        </header>

        <div class="{content_class}">
            {rendered_content}
        </div>
    </div>
    <script>
    // Highlight target element on page load
    if (window.location.hash) {{
        const target = document.querySelector(window.location.hash);
        if (target) {{
            target.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
        }}
    }}
    </script>
</body>
</html>"""


__all__ = ["render_document_viewer"]
