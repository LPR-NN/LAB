"""HTML exporter for committee decisions."""

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # For future type-only imports

from source.contracts.audit import DecisionPackage
from source.contracts.decision import Decision
from source.contracts.request import Request


@dataclass
class SiteConfig:
    """Site configuration from cases.json."""

    site_title: str = "Решения ИИ арбитра"
    org_name: str = "LLM инициатива"
    cases: dict[str, dict[str, str]] = field(default_factory=dict)
    model_ranking: list[str] = field(default_factory=list)
    disabled_models: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, config_path: Path) -> "SiteConfig":
        """Load config from JSON file."""
        if not config_path.exists():
            return cls()
        data = json.loads(config_path.read_text(encoding="utf-8"))
        return cls(
            site_title=data.get("site_title", cls.site_title),
            org_name=data.get("org_name", cls.org_name),
            cases=data.get("cases", {}),
            model_ranking=data.get("model_ranking", []),
            disabled_models=data.get("disabled_models", []),
        )

    def get_case_title(self, case_slug: str, default: str) -> str:
        """Get custom title for a case or return default."""
        case_data = self.cases.get(case_slug, {})
        return case_data.get("title", default)

    def get_model_rank(self, model_id: str) -> int:
        """
        Get rank for a model (1 = best, higher = worse).

        Models in model_ranking list get their position (1-indexed).
        Models not in the list get a high rank (sorted to the end).
        """
        try:
            return self.model_ranking.index(model_id) + 1
        except ValueError:
            return 999

    def is_model_disabled(self, model_id: str) -> bool:
        """Check if a model is disabled."""
        return model_id in self.disabled_models


@dataclass
class DecisionContext:
    """Context for rendering a decision."""

    package: DecisionPackage
    decision: Decision
    request: Request


@dataclass
class CaseGroup:
    """A group of decisions for the same case from different models."""

    case_id: str
    case_slug: str
    title: str
    packages: list[DecisionContext] = field(default_factory=list)


CSS_STYLES = """
:root {
    --bg-primary: #fdfcfa;
    --bg-secondary: #f5f3ef;
    --bg-accent: #e8e4dc;
    --text-primary: #1a1a1a;
    --text-secondary: #4a4a4a;
    --text-muted: #6b6b6b;
    --border-color: #d4d0c8;
    --accent-color: #8b4513;
    --accent-light: #cd853f;
    --heading-color: #2c2c2c;
    --link-color: #5c4a3d;
    --success: #2d5a3d;
    --warning: #8b6914;
    --info: #3d5a7a;
}

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

html {
    font-size: 16px;
    scroll-behavior: smooth;
}

body {
    font-family: "Charter", "Georgia", "Cambria", "Times New Roman", serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    line-height: 1.7;
    padding: 0;
    margin: 0;
}

.container {
    max-width: 850px;
    margin: 0 auto;
    padding: 3rem 2rem;
}

/* Header */
.decision-header {
    text-align: center;
    padding-bottom: 2.5rem;
    border-bottom: 2px solid var(--accent-color);
    margin-bottom: 2.5rem;
}

.org-name {
    font-size: 0.85rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.5rem;
    font-weight: 500;
}

.decision-title {
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--heading-color);
    margin: 1rem 0;
    line-height: 1.3;
}

.decision-meta {
    font-size: 0.9rem;
    color: var(--text-secondary);
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 1.5rem;
    margin-top: 1rem;
}

.meta-item {
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

.meta-label {
    font-weight: 500;
    color: var(--text-muted);
}

/* Verdict banner */
.verdict-banner {
    background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-accent) 100%);
    border: 1px solid var(--border-color);
    border-left: 4px solid var(--accent-color);
    padding: 1.5rem 2rem;
    margin-bottom: 2.5rem;
    border-radius: 0 4px 4px 0;
}

.verdict-type {
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--accent-color);
    font-weight: 600;
    margin-bottom: 0.5rem;
}

.verdict-summary {
    font-size: 1.1rem;
    color: var(--heading-color);
    font-weight: 500;
    line-height: 1.5;
}

/* Sections */
.section {
    margin-bottom: 2.5rem;
}

.section-title {
    font-size: 1rem;
    font-weight: 600;
    color: var(--heading-color);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--border-color);
    margin-bottom: 1.25rem;
}

.section-number {
    color: var(--accent-color);
    margin-right: 0.5rem;
}

/* Parties */
.parties-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
}

.party-card {
    background: var(--bg-secondary);
    padding: 1rem 1.25rem;
    border-radius: 4px;
    border: 1px solid var(--border-color);
}

.party-role {
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--accent-color);
    font-weight: 600;
    margin-bottom: 0.25rem;
}

.party-name {
    font-weight: 500;
    color: var(--heading-color);
    margin-bottom: 0.25rem;
}

.party-desc {
    font-size: 0.85rem;
    color: var(--text-muted);
}

/* Facts list */
.facts-list {
    list-style: none;
    counter-reset: fact-counter;
}

.fact-item {
    counter-increment: fact-counter;
    padding: 1rem 1.25rem;
    background: var(--bg-secondary);
    border-radius: 4px;
    margin-bottom: 0.75rem;
    border: 1px solid var(--border-color);
    position: relative;
    padding-left: 3.5rem;
}

.fact-item::before {
    content: counter(fact-counter);
    position: absolute;
    left: 1.25rem;
    top: 1rem;
    width: 1.5rem;
    height: 1.5rem;
    background: var(--accent-color);
    color: white;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 600;
}

.fact-text {
    color: var(--text-primary);
    line-height: 1.6;
}

.fact-refs {
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-top: 0.5rem;
    font-style: italic;
}

.fact-refs a.evidence-link {
    color: var(--accent-color);
    text-decoration: none;
}

.fact-refs a.evidence-link:hover {
    text-decoration: underline;
}

/* Corpus document links */
a.corpus-link {
    color: var(--accent-color);
    text-decoration: none;
    font-family: "JetBrains Mono", "Fira Code", monospace;
}

a.corpus-link:hover {
    text-decoration: underline;
}

.confidence-badge {
    position: absolute;
    right: 1rem;
    top: 0.75rem;
    font-size: 0.65rem;
    font-family: "JetBrains Mono", "Fira Code", monospace;
    padding: 0.1rem 0.4rem;
    background: transparent;
    border: 1px solid var(--border-color);
    border-radius: 3px;
    color: var(--text-muted);
    opacity: 0.6;
}

/* Norms */
.norms-list {
    list-style: none;
}

.norm-item {
    padding: 1rem 1.25rem;
    border-left: 3px solid var(--accent-light);
    background: var(--bg-secondary);
    margin-bottom: 0.75rem;
}

.norm-citation {
    font-weight: 600;
    color: var(--accent-color);
    font-family: "JetBrains Mono", "Fira Code", "Monaco", monospace;
    font-size: 0.85rem;
}

.norm-section {
    color: var(--text-secondary);
    font-size: 0.85rem;
    margin-left: 0.5rem;
}

.norm-relevance {
    margin-top: 0.5rem;
    color: var(--text-primary);
    font-size: 0.95rem;
}

/* Reasoning chain */
.reasoning-chain {
    position: relative;
}

.reasoning-step {
    position: relative;
    padding: 1.25rem 1.5rem;
    background: var(--bg-secondary);
    border-radius: 4px;
    margin-bottom: 1rem;
    border: 1px solid var(--border-color);
}

.reasoning-step::before {
    content: "";
    position: absolute;
    left: 2rem;
    top: -1rem;
    height: 1rem;
    width: 2px;
    background: var(--accent-light);
}

.reasoning-step:first-child::before {
    display: none;
}

.reasoning-label {
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 600;
    margin-bottom: 0.25rem;
}

.reasoning-fact .reasoning-label {
    color: var(--info);
}

.reasoning-norm .reasoning-label {
    color: var(--accent-color);
}

.reasoning-conclusion .reasoning-label {
    color: var(--success);
}

.reasoning-content {
    color: var(--text-primary);
    line-height: 1.6;
}

.reasoning-fact {
    border-left: 3px solid var(--info);
}

.reasoning-norm {
    border-left: 3px solid var(--accent-color);
}

.reasoning-conclusion {
    border-left: 3px solid var(--success);
    background: linear-gradient(135deg, #f0f7f2 0%, var(--bg-secondary) 100%);
}

/* Citations */
.citations-list {
    list-style: none;
}

.citation-item {
    padding: 1rem 1.25rem;
    background: var(--bg-secondary);
    border-radius: 4px;
    margin-bottom: 0.75rem;
    border: 1px solid var(--border-color);
}

.citation-ref {
    font-weight: 600;
    color: var(--accent-color);
    font-family: "JetBrains Mono", "Fira Code", "Monaco", monospace;
    font-size: 0.85rem;
    margin-bottom: 0.5rem;
}

.citation-text {
    font-style: italic;
    color: var(--text-secondary);
    padding-left: 1rem;
    border-left: 2px solid var(--accent-light);
    line-height: 1.6;
}

/* Notes (uncertainty, minority view) */
.note-box {
    padding: 1.25rem 1.5rem;
    border-radius: 4px;
    margin-bottom: 1.5rem;
    border: 1px solid var(--border-color);
}

.note-box.uncertainty {
    background: linear-gradient(135deg, #fef9e7 0%, #fdf6d3 100%);
    border-color: #e6d396;
}

.note-box.minority {
    background: linear-gradient(135deg, #f0f4f8 0%, #e8eef4 100%);
    border-color: #bccad8;
}

.note-title {
    font-size: 0.8rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 600;
    margin-bottom: 0.5rem;
}

.uncertainty .note-title {
    color: var(--warning);
}

.minority .note-title {
    color: var(--info);
}

.note-content {
    color: var(--text-primary);
    line-height: 1.6;
}

/* Next steps */
.next-steps-list {
    list-style: none;
    counter-reset: step-counter;
}

.next-step-item {
    counter-increment: step-counter;
    padding: 0.75rem 1rem;
    padding-left: 2.5rem;
    position: relative;
    background: var(--bg-secondary);
    border-radius: 4px;
    margin-bottom: 0.5rem;
    border: 1px solid var(--border-color);
}

.next-step-item::before {
    content: counter(step-counter) ".";
    position: absolute;
    left: 1rem;
    color: var(--accent-color);
    font-weight: 600;
}

/* Footer */
.decision-footer {
    margin-top: 3rem;
    padding-top: 2rem;
    border-top: 2px solid var(--accent-color);
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
}

.footer-main {
    text-align: left;
}

.footer-date {
    font-size: 0.9rem;
    color: var(--text-secondary);
}

.footer-signature {
    font-size: 0.85rem;
    color: var(--text-muted);
    font-style: italic;
    margin-top: 0.5rem;
}

.tech-widget {
    text-align: right;
    font-family: "JetBrains Mono", "Fira Code", monospace;
    font-size: 0.7rem;
    color: var(--text-muted);
    opacity: 0.7;
    line-height: 1.5;
}

.tech-widget:hover {
    opacity: 1;
}

.tech-widget .tech-label {
    color: var(--text-muted);
    opacity: 0.6;
}

.tech-widget .tech-value {
    color: var(--text-secondary);
}

/* Attachments section */
.attachments-list {
    list-style: none;
}

.attachment-item {
    padding: 1rem 1.25rem;
    background: var(--bg-secondary);
    border-radius: 4px;
    margin-bottom: 0.75rem;
    border: 1px solid var(--border-color);
    display: flex;
    align-items: flex-start;
    gap: 1rem;
}

.attachment-icon {
    flex-shrink: 0;
    width: 2rem;
    height: 2rem;
    background: var(--accent-color);
    color: white;
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8rem;
    font-weight: 600;
}

.attachment-info {
    flex-grow: 1;
    min-width: 0;
}

.attachment-name {
    font-weight: 600;
    color: var(--accent-color);
    text-decoration: none;
    display: block;
    margin-bottom: 0.25rem;
}

.attachment-name:hover {
    text-decoration: underline;
}

.attachment-type {
    font-size: 0.7rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--text-muted);
    background: var(--bg-primary);
    padding: 0.1rem 0.4rem;
    border-radius: 3px;
    border: 1px solid var(--border-color);
    margin-right: 0.5rem;
}

.attachment-note {
    font-size: 0.85rem;
    color: var(--text-secondary);
    margin-top: 0.5rem;
    line-height: 1.5;
}

/* Print styles */
@media print {
    body {
        font-size: 11pt;
        background: white;
    }

    .container {
        max-width: 100%;
        padding: 1cm;
    }

    .decision-header {
        page-break-after: avoid;
    }

    .section {
        page-break-inside: avoid;
    }

    .reasoning-step,
    .fact-item,
    .norm-item,
    .citation-item {
        page-break-inside: avoid;
    }
}

/* Responsive */
@media (max-width: 600px) {
    html {
        font-size: 15px;
    }

    .container {
        padding: 1.5rem 1rem;
    }

    .decision-meta {
        flex-direction: column;
        gap: 0.5rem;
    }

    .parties-grid {
        grid-template-columns: 1fr;
    }
}
"""


VERDICT_LABELS = {
    "decision": "Решение",
    "sanction": "Санкция",
    "clarification": "Разъяснение",
    "refusal": "Отказ",
    "needs_more_info": "Требуется информация",
}

ROLE_LABELS = {
    "complainant": "Заявитель",
    "respondent": "Ответчик",
    "affected_party": "Затронутая сторона",
    "witness": "Свидетель",
}

ATTACHMENT_TYPE_LABELS = {
    "statement": "Заявление",
    "evidence": "Доказательство",
    "transcript": "Протокол",
    "document": "Документ",
    "correspondence": "Переписка",
    "other": "Прочее",
}


def escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _make_section_anchor(section: str) -> str:
    """Convert section reference like 'п. 3.4.1.2' to anchor '#p-3-4-1-2'."""
    # Extract the number pattern
    match = re.search(r"(\d+(?:\.\d+)+)", section)
    if match:
        number = match.group(1)
        return f"#p-{number.replace('.', '-')}"
    return ""


def _linkify_corpus_citation(text: str) -> str:
    """
    Convert corpus citations like [UST-B1CAE0] or [MAIN-DOCS-...] to clickable links.

    Handles patterns:
    - [UST-B1CAE0]
    - [UST-B1CAE0, п. 3.4.1.2] -> links to #p-3-4-1-2
    - [MAIN-DOCS-LIBERTARIAN-PARTY-CHARTER]
    - [DEC-10]
    - [COD-E718C4]
    - [EK-ЭК]

    Result: [<a href="...#anchor">KEY</a>] or [<a href="...#anchor">KEY</a>, section]
    """
    # Pattern: [KEY followed by optional section info and closing ]
    # KEY can include A-Z, 0-9, -, _, and Cyrillic А-ЯЁ
    pattern = r"\[([A-ZА-ЯЁ][A-ZА-ЯЁ0-9_-]+(?:-[A-ZА-ЯЁ0-9_-]+)*)(?:,\s*([^\]]+))?\]"

    def replace_citation(match: re.Match[str]) -> str:
        citation_key = match.group(1)
        section_info = match.group(2)  # May be None

        # Build URL with optional anchor
        url = f"/corpus/by-key/{citation_key}"
        if section_info:
            anchor = _make_section_anchor(section_info)
            url += anchor
            # Return with section info visible
            return (
                f'[<a href="{url}" class="corpus-link" target="_blank">'
                f"{escape_html(citation_key)}</a>, {escape_html(section_info)}]"
            )
        else:
            return f'[<a href="{url}" class="corpus-link" target="_blank">{escape_html(citation_key)}</a>]'

    return re.sub(pattern, replace_citation, text)


def _linkify_evidence_ref(ref: str) -> str:
    """
    Convert evidence reference to a clickable link if it's a file path.

    Converts paths like "assets/generalov.md" to links.
    Handles references with additional context like "assets/generalov.md (п.5–6)".
    """
    ref = ref.strip()

    # Extract just the file path (before any parenthetical or whitespace suffix)
    # Pattern: path possibly followed by space and parenthetical
    path_match = re.match(r"^((?:assets|data)/[^\s(]+)", ref)

    if not path_match:
        return escape_html(ref)

    file_path = path_match.group(1)
    suffix = ref[len(file_path) :]  # Everything after the path

    # Build URL based on path prefix
    if file_path.startswith("assets/"):
        url = f"/attachments/{file_path[7:]}"  # Remove "assets/" prefix
    elif file_path.startswith("data/"):
        url = f"/corpus/{file_path[5:]}"  # Remove "data/" prefix
    else:
        return escape_html(ref)

    # Build the link with optional suffix
    link = f'<a href="{url}" class="evidence-link" target="_blank">{escape_html(file_path)}</a>'
    if suffix:
        link += escape_html(suffix)

    return link


class HTMLExporter:
    """Exports decisions to HTML format."""

    def __init__(self, org_name: str = "LLM инициатива"):
        self.org_name = org_name

    def export_from_package(self, package: DecisionPackage) -> str:
        """Export a decision package to HTML."""
        decision = Decision.model_validate_json(package.decision_json)
        request = Request.model_validate_json(package.request_json)
        ctx = DecisionContext(package=package, decision=decision, request=request)
        return self._render(ctx)

    def export_from_files(self, package_path: Path) -> str:
        """Export from a JSON package file."""
        content = package_path.read_text(encoding="utf-8")
        package = DecisionPackage.model_validate_json(content)
        return self.export_from_package(package)

    def _render(self, ctx: DecisionContext) -> str:
        """Render the full HTML document."""
        return f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{escape_html(VERDICT_LABELS.get(ctx.decision.verdict, ctx.decision.verdict))} — {escape_html(self.org_name)}</title>
    <style>
{CSS_STYLES}
    </style>
</head>
<body>
    <div class="container">
        {self.render_header(ctx)}
        {self.render_verdict(ctx)}
        {self.render_parties(ctx)}
        {self.render_facts(ctx)}
        {self.render_norms(ctx)}
        {self.render_reasoning(ctx)}
        {self.render_citations(ctx)}
        {self.render_notes(ctx)}
        {self.render_next_steps(ctx)}
        {self.render_footer(ctx)}
    </div>
</body>
</html>"""

    def render_header(self, ctx: DecisionContext) -> str:
        """Render the document header."""
        timestamp = ctx.package.timestamp
        date_str = timestamp.strftime("%d.%m.%Y")

        return f"""
        <header class="decision-header">
            <div class="org-name">{escape_html(self.org_name)}</div>
            <h1 class="decision-title">{escape_html(VERDICT_LABELS.get(ctx.decision.verdict, ctx.decision.verdict))}</h1>
            <div class="decision-meta">
                <div class="meta-item">
                    <span class="meta-label">Дата:</span>
                    <span>{date_str}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Модель:</span>
                    <span>{escape_html(ctx.package.model_id)}</span>
                </div>
            </div>
        </header>"""

    def render_verdict(self, ctx: DecisionContext) -> str:
        """Render the verdict banner."""
        return f"""
        <div class="verdict-banner">
            <div class="verdict-type">{escape_html(VERDICT_LABELS.get(ctx.decision.verdict, ctx.decision.verdict))}</div>
            <div class="verdict-summary">{escape_html(ctx.decision.verdict_summary)}</div>
        </div>"""

    def render_attachments(self, ctx: DecisionContext) -> str:
        """Render the attachments (case materials) section."""
        if not ctx.request.attachments:
            return ""

        attachments_html: list[str] = []
        for attachment in ctx.request.attachments:
            file_path = attachment.file_path
            file_name = Path(file_path).name

            # Build URL for viewing
            if file_path.startswith("assets/"):
                url = f"/attachments/{file_path[7:]}"
            elif file_path.startswith("data/"):
                url = f"/corpus/{file_path[5:]}"
            else:
                url = f"/attachments/{file_path}"

            # Get type label
            type_label = ATTACHMENT_TYPE_LABELS.get(
                attachment.attachment_type, attachment.attachment_type
            )

            # Icon based on extension
            ext = Path(file_path).suffix.lower()
            icon_map = {".md": "MD", ".txt": "TXT", ".pdf": "PDF", ".json": "JS"}
            icon = icon_map.get(ext, "DOC")

            note_html = ""
            if attachment.relevance_note:
                note_html = f'<div class="attachment-note">{escape_html(attachment.relevance_note)}</div>'

            attachments_html.append(f"""
                <li class="attachment-item">
                    <div class="attachment-icon">{icon}</div>
                    <div class="attachment-info">
                        <a href="{url}" class="attachment-name" target="_blank">{escape_html(file_name)}</a>
                        <span class="attachment-type">{escape_html(type_label)}</span>
                        {note_html}
                    </div>
                </li>""")

        return f"""
        <section class="section">
            <h2 class="section-title"><span class="section-number">0.</span> Материалы дела</h2>
            <ul class="attachments-list">
                {"".join(attachments_html)}
            </ul>
        </section>"""

    def render_parties(self, ctx: DecisionContext) -> str:
        """Render the parties section."""
        if not ctx.request.parties:
            return ""

        parties_html: list[str] = []
        for party in ctx.request.parties:
            role_label = ROLE_LABELS.get(party.role, party.role)
            desc = (
                f'<div class="party-desc">{escape_html(party.description)}</div>'
                if party.description
                else ""
            )
            parties_html.append(f"""
                <div class="party-card">
                    <div class="party-role">{escape_html(role_label)}</div>
                    <div class="party-name">{escape_html(party.name)}</div>
                    {desc}
                </div>""")

        return f"""
        <section class="section">
            <h2 class="section-title"><span class="section-number">I.</span> Стороны</h2>
            <div class="parties-grid">
                {"".join(parties_html)}
            </div>
        </section>"""

    def render_facts(self, ctx: DecisionContext) -> str:
        """Render findings of fact."""
        if not ctx.decision.findings_of_fact:
            return ""

        facts_html: list[str] = []
        for fact in ctx.decision.findings_of_fact:
            if fact.evidence_refs:
                linked_refs = ", ".join(
                    _linkify_evidence_ref(r) for r in fact.evidence_refs
                )
                refs_html = f'<div class="fact-refs">Источники: {linked_refs}</div>'
            else:
                refs_html = ""
            confidence = int(fact.confidence * 100)
            badge = f'<span class="confidence-badge">{confidence}%</span>'

            facts_html.append(f"""
                <li class="fact-item">
                    {badge}
                    <div class="fact-text">{escape_html(fact.fact)}</div>
                    {refs_html}
                </li>""")

        return f"""
        <section class="section">
            <h2 class="section-title"><span class="section-number">II.</span> Установленные факты</h2>
            <ol class="facts-list">
                {"".join(facts_html)}
            </ol>
        </section>"""

    def render_norms(self, ctx: DecisionContext) -> str:
        """Render applicable norms."""
        if not ctx.decision.applicable_norms:
            return ""

        norms_html: list[str] = []
        for norm in ctx.decision.applicable_norms:
            # Make citation_key a link with section anchor
            citation_with_section = f"[{norm.citation_key}, {norm.section}]"
            citation_link = _linkify_corpus_citation(citation_with_section)
            norms_html.append(f"""
                <li class="norm-item">
                    <div>
                        <span class="norm-citation">{citation_link}</span>
                    </div>
                    <div class="norm-relevance">{escape_html(norm.relevance)}</div>
                </li>""")

        return f"""
        <section class="section">
            <h2 class="section-title"><span class="section-number">III.</span> Применимые нормы</h2>
            <ul class="norms-list">
                {"".join(norms_html)}
            </ul>
        </section>"""

    def render_reasoning(self, ctx: DecisionContext) -> str:
        """Render reasoning chain."""
        if not ctx.decision.reasoning:
            return ""

        steps_html: list[str] = []
        for step in ctx.decision.reasoning:
            # Process norm_or_precedent to make citations clickable
            norm_html = _linkify_corpus_citation(escape_html(step.norm_or_precedent))
            steps_html.append(f"""
                <div class="reasoning-step reasoning-fact">
                    <div class="reasoning-label">Факт</div>
                    <div class="reasoning-content">{escape_html(step.fact)}</div>
                </div>
                <div class="reasoning-step reasoning-norm">
                    <div class="reasoning-label">Норма</div>
                    <div class="reasoning-content">{norm_html}</div>
                </div>
                <div class="reasoning-step reasoning-conclusion">
                    <div class="reasoning-label">Вывод</div>
                    <div class="reasoning-content">{escape_html(step.conclusion)}</div>
                </div>""")

        return f"""
        <section class="section">
            <h2 class="section-title"><span class="section-number">IV.</span> Обоснование</h2>
            <div class="reasoning-chain">
                {"".join(steps_html)}
            </div>
        </section>"""

    def render_citations(self, ctx: DecisionContext) -> str:
        """Render citations."""
        if not ctx.decision.citations:
            return ""

        citations_html: list[str] = []
        for citation in ctx.decision.citations:
            # Make doc_id a clickable link with section anchor
            citation_with_section = f"[{citation.doc_id}, {citation.section}]"
            doc_link = _linkify_corpus_citation(citation_with_section)
            citations_html.append(f"""
                <li class="citation-item">
                    <div class="citation-ref">{doc_link}</div>
                    <blockquote class="citation-text">«{escape_html(citation.quoted_text)}»</blockquote>
                </li>""")

        return f"""
        <section class="section">
            <h2 class="section-title"><span class="section-number">V.</span> Цитаты</h2>
            <ul class="citations-list">
                {"".join(citations_html)}
            </ul>
        </section>"""

    def render_notes(self, ctx: DecisionContext) -> str:
        """Render uncertainty and minority view."""
        notes_html: list[str] = []

        if ctx.decision.uncertainty:
            notes_html.append(f"""
            <div class="note-box uncertainty">
                <div class="note-title">Неопределённость</div>
                <div class="note-content">{escape_html(ctx.decision.uncertainty)}</div>
            </div>""")

        if ctx.decision.minority_view:
            notes_html.append(f"""
            <div class="note-box minority">
                <div class="note-title">Альтернативная позиция</div>
                <div class="note-content">{escape_html(ctx.decision.minority_view)}</div>
            </div>""")

        if not notes_html:
            return ""

        return f"""
        <section class="section">
            <h2 class="section-title"><span class="section-number">VI.</span> Примечания</h2>
            {"".join(notes_html)}
        </section>"""

    def render_next_steps(self, ctx: DecisionContext) -> str:
        """Render recommended next steps."""
        if not ctx.decision.recommended_next_steps:
            return ""

        steps_html: list[str] = []
        for step in ctx.decision.recommended_next_steps:
            steps_html.append(f'<li class="next-step-item">{escape_html(step)}</li>')

        return f"""
        <section class="section">
            <h2 class="section-title"><span class="section-number">VII.</span> Рекомендуемые действия</h2>
            <ol class="next-steps-list">
                {"".join(steps_html)}
            </ol>
        </section>"""

    def render_footer(self, ctx: DecisionContext) -> str:
        """Render the document footer."""
        timestamp = ctx.package.timestamp
        date_str = timestamp.strftime("%d.%m.%Y")
        short_id = ctx.package.package_id[:8]

        return f"""
        <footer class="decision-footer">
            <div class="footer-main">
                <div class="footer-date">{date_str}</div>
                <div class="footer-signature">ИИ арбитр · {escape_html(self.org_name)}</div>
            </div>
            <div class="tech-widget">
                <div><span class="tech-label">model:</span> <span class="tech-value">{escape_html(ctx.package.model_id)}</span></div>
                <div><span class="tech-label">agent:</span> <span class="tech-value">{escape_html(ctx.package.agent_version)}</span></div>
                <div><span class="tech-label">id:</span> <span class="tech-value">{escape_html(short_id)}</span></div>
            </div>
        </footer>"""


def export_decision_to_html(
    package_path: Path,
    output_path: Path | None = None,
    org_name: str = "LLM инициатива",
) -> Path:
    """
    Export a decision package to HTML.

    Args:
        package_path: Path to the JSON package file
        output_path: Optional output path (defaults to same dir with .html extension)
        org_name: Organization name for the header

    Returns:
        Path to the generated HTML file
    """
    exporter = HTMLExporter(org_name=org_name)
    html = exporter.export_from_files(package_path)

    if output_path is None:
        output_path = package_path.with_suffix(".html")

    output_path.write_text(html, encoding="utf-8")
    return output_path


TABS_CSS = """
/* Model selector tabs */
.model-selector {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-bottom: 2rem;
    padding: 1rem;
    background: var(--bg-secondary);
    border-radius: 8px;
    border: 1px solid var(--border-color);
}

.model-tab {
    padding: 0.6rem 1.2rem;
    border: 1px solid var(--border-color);
    background: var(--bg-primary);
    border-radius: 4px;
    cursor: pointer;
    font-family: inherit;
    font-size: 0.9rem;
    color: var(--text-secondary);
    transition: all 0.2s;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.model-tab:hover {
    background: var(--bg-accent);
    border-color: var(--accent-light);
}

.model-tab.active {
    background: var(--accent-color);
    color: white;
    border-color: var(--accent-color);
}

.model-tab .model-rank {
    font-size: 0.7rem;
    font-weight: 700;
    padding: 0.15rem 0.4rem;
    border-radius: 4px;
    background: rgba(0, 0, 0, 0.08);
    color: var(--text-muted);
    font-family: "JetBrains Mono", monospace;
}

.model-tab.active .model-rank {
    background: rgba(255, 255, 255, 0.2);
    color: rgba(255, 255, 255, 0.9);
}

/* Gold, silver, bronze for top 3 */
.model-tab:nth-child(1) .model-rank {
    background: linear-gradient(135deg, #ffd700, #ffb800);
    color: #5c4a00;
}
.model-tab:nth-child(2) .model-rank {
    background: linear-gradient(135deg, #c0c0c0, #a8a8a8);
    color: #404040;
}
.model-tab:nth-child(3) .model-rank {
    background: linear-gradient(135deg, #cd7f32, #b8722e);
    color: #3d2510;
}

.model-tab.active:nth-child(1) .model-rank,
.model-tab.active:nth-child(2) .model-rank,
.model-tab.active:nth-child(3) .model-rank {
    opacity: 0.9;
}

.model-tab .model-name {
    font-weight: 600;
}

.model-tab .model-date {
    font-size: 0.8rem;
    opacity: 0.8;
    margin-left: 0.25rem;
}

/* Disabled models */
.model-tab.disabled {
    opacity: 0.5;
    cursor: not-allowed;
    background: var(--bg-secondary);
    border-style: dashed;
}

.model-tab.disabled:hover {
    background: var(--bg-secondary);
    border-color: var(--border-color);
    transform: none;
}

.model-tab.disabled .model-rank {
    background: #888;
    color: #ccc;
}

.model-tab.disabled .model-name {
    opacity: 0.7;
}

.disabled-notice {
    text-align: center;
    padding: 4rem 2rem;
    color: var(--text-muted);
}

.disabled-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
    opacity: 0.5;
}

.disabled-text {
    font-size: 1rem;
    font-style: italic;
}

/* Decision panels */
.decision-panel {
    display: none;
}

.decision-panel.active {
    display: block;
}

/* Back link */
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

@media print {
    .model-selector {
        display: none;
    }
    .back-link {
        display: none;
    }
    .decision-panel {
        display: block !important;
        page-break-before: always;
    }
    .decision-panel:first-of-type {
        page-break-before: auto;
    }
}
"""


TABS_JS = """
function switchModel(modelId) {
    // Update tabs
    document.querySelectorAll('.model-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.model === modelId);
    });
    // Update panels
    document.querySelectorAll('.decision-panel').forEach(panel => {
        panel.classList.toggle('active', panel.id === 'panel-' + modelId);
    });
    // Update URL hash
    history.replaceState(null, '', '#' + modelId);
}

// Initialize from URL hash
document.addEventListener('DOMContentLoaded', function() {
    const hash = window.location.hash.slice(1);
    if (hash) {
        const tab = document.querySelector(`.model-tab[data-model="${hash}"]`);
        if (tab) switchModel(hash);
    }
});
"""


def _slugify(text: str) -> str:
    """Convert text to URL-friendly slug."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s]+", "-", text)
    return text.strip("-")[:50]


def _get_case_id(ctx: DecisionContext) -> str:
    """Extract case ID from a decision context."""
    if ctx.request.case_id:
        return ctx.request.case_id

    if ctx.request.attachments:
        first_attachment = ctx.request.attachments[0].file_path
        return hashlib.md5(first_attachment.encode()).hexdigest()[:8]

    return hashlib.md5(ctx.request.query.encode()).hexdigest()[:8]


def _get_model_slug(model_id: str) -> str:
    """Convert model ID to a slug."""
    slug = model_id.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")


@dataclass
class CaseOverrides:
    """Manual overrides for case metadata preserved from existing HTML."""

    title: str | None = None
    org_name: str | None = None


def parse_overrides_from_html(html_path: Path) -> CaseOverrides:
    """Parse custom title and org_name from existing HTML file."""
    if not html_path.exists():
        return CaseOverrides()

    content = html_path.read_text(encoding="utf-8")

    title_match = re.search(r'<h1\s+class="decision-title">([^<]+)</h1>', content)
    org_match = re.search(r'<div\s+class="org-name">([^<]+)</div>', content)

    return CaseOverrides(
        title=title_match.group(1) if title_match else None,
        org_name=org_match.group(1) if org_match else None,
    )


class CaseExporter:
    """Exports decisions grouped by case with model selection."""

    def __init__(self, org_name: str = "LLM инициатива"):
        self.org_name = org_name
        self.html_exporter = HTMLExporter(org_name=org_name)

    def load_packages(self, audit_dir: Path) -> list[DecisionContext]:
        """Load all packages from audit directory."""
        contexts: list[DecisionContext] = []
        for json_file in audit_dir.glob("*.json"):
            try:
                content = json_file.read_text(encoding="utf-8")
                package = DecisionPackage.model_validate_json(content)
                decision = Decision.model_validate_json(package.decision_json)
                request = Request.model_validate_json(package.request_json)
                contexts.append(
                    DecisionContext(package=package, decision=decision, request=request)
                )
            except Exception:
                continue
        return contexts

    def group_by_case(self, contexts: list[DecisionContext]) -> list[CaseGroup]:
        """Group decision contexts by case."""
        groups: dict[str, CaseGroup] = {}

        for ctx in contexts:
            case_id = _get_case_id(ctx)

            if case_id not in groups:
                if ctx.request.case_id:
                    title = ctx.request.case_id
                    slug = _slugify(ctx.request.case_id)
                elif ctx.request.attachments:
                    attachment_name = Path(ctx.request.attachments[0].file_path).stem
                    title = attachment_name.replace("-", " ").replace("_", " ").title()
                    slug = _slugify(attachment_name)
                else:
                    title = ctx.request.query[:60] + "..."
                    slug = case_id

                groups[case_id] = CaseGroup(
                    case_id=case_id, case_slug=slug, title=title
                )

            groups[case_id].packages.append(ctx)

        for group in groups.values():
            group.packages.sort(key=lambda c: c.package.timestamp, reverse=True)

        return sorted(groups.values(), key=lambda g: g.title)

    def export_case(
        self,
        group: CaseGroup,
        back_url: str = "../index.html",
        overrides: CaseOverrides | None = None,
        config: SiteConfig | None = None,
    ) -> str:
        """Export a single case with all model versions.

        Args:
            group: Case group to export
            back_url: URL for the "back to all" link
            overrides: Manual overrides for title/org_name (preserved from existing HTML)
            config: Site config with model_ranking for sorting
        """
        panels_html: list[str] = []
        tabs_html: list[str] = []

        display_title = (
            overrides.title if overrides and overrides.title else group.title
        )
        display_org = (
            overrides.org_name if overrides and overrides.org_name else self.org_name
        )

        # Sort packages by model rank (best first)
        sorted_packages = group.packages
        if config and config.model_ranking:
            sorted_packages = sorted(
                group.packages,
                key=lambda ctx: config.get_model_rank(ctx.package.model_id),
            )

        # Find first enabled model for default active state
        first_enabled_idx = 0
        for idx, ctx in enumerate(sorted_packages):
            if not (config and config.is_model_disabled(ctx.package.model_id)):
                first_enabled_idx = idx
                break

        for i, ctx in enumerate(sorted_packages):
            model_slug = _get_model_slug(ctx.package.model_id)
            model_name = ctx.package.model_id
            date_str = ctx.package.timestamp.strftime("%d.%m")
            is_disabled = config and config.is_model_disabled(model_name)
            is_active = (i == first_enabled_idx) and not is_disabled
            rank = i + 1  # Position in sorted list

            # Rank badge for top models
            rank_badge = ""
            if len(sorted_packages) > 1:
                rank_badge = f'<span class="model-rank">#{rank}</span>'

            # Button classes and attributes
            tab_classes = ["model-tab"]
            if is_active:
                tab_classes.append("active")
            if is_disabled:
                tab_classes.append("disabled")

            disabled_attr = (
                ' disabled title="Решение недоступно"' if is_disabled else ""
            )
            onclick = "" if is_disabled else f"onclick=\"switchModel('{model_slug}')\""

            tabs_html.append(
                f'<button class="{" ".join(tab_classes)}" '
                f'data-model="{model_slug}" {onclick}{disabled_attr}>'
                f"{rank_badge}"
                f'<span class="model-name">{escape_html(model_name)}</span>'
                f'<span class="model-date">{date_str}</span>'
                f"</button>"
            )

            # Generate panel content (placeholder for disabled)
            if is_disabled:
                panel_content = """
                <div class="disabled-notice">
                    <div class="disabled-icon">🔒</div>
                    <div class="disabled-text">Решение этой модели недоступно для просмотра</div>
                </div>"""
            else:
                panel_content = self._render_decision_content(ctx)
            panels_html.append(
                f'<div id="panel-{model_slug}" class="decision-panel{"" if not is_active else " active"}">'
                f"{panel_content}"
                f"</div>"
            )

        return f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{escape_html(display_title)} — {escape_html(display_org)}</title>
    <style>
{CSS_STYLES}
{TABS_CSS}
    </style>
</head>
<body>
    <div class="container">
        <a href="{back_url}" class="back-link">Все решения</a>

        <header class="decision-header">
            <div class="org-name">{escape_html(display_org)}</div>
            <h1 class="decision-title">{escape_html(display_title)}</h1>
            <div class="decision-meta">
                <div class="meta-item">
                    <span class="meta-label">Версий:</span>
                    <span>{len(group.packages)}</span>
                </div>
            </div>
        </header>

        <div class="model-selector">
            {"".join(tabs_html)}
        </div>

        {"".join(panels_html)}
    </div>
    <script>
{TABS_JS}
    </script>
</body>
</html>"""

    def _render_decision_content(self, ctx: DecisionContext) -> str:
        """Render decision content without the outer wrapper."""
        return f"""
        {self.html_exporter.render_verdict(ctx)}
        {self.html_exporter.render_parties(ctx)}
        {self.html_exporter.render_facts(ctx)}
        {self.html_exporter.render_norms(ctx)}
        {self.html_exporter.render_reasoning(ctx)}
        {self.html_exporter.render_citations(ctx)}
        {self.html_exporter.render_notes(ctx)}
        {self.html_exporter.render_next_steps(ctx)}
        {self.html_exporter.render_footer(ctx)}
        """

    def export_index(
        self,
        groups: list[CaseGroup],
        base_url: str = "",
        title_overrides: dict[str, str] | None = None,
    ) -> str:
        """Generate main index page listing all cases.

        Args:
            groups: List of case groups
            base_url: Base URL for links (e.g. "/cases" for API, "" for static)
            title_overrides: Dict mapping case_slug to custom title
        """
        items_html: list[str] = []

        for group in groups:
            latest = group.packages[0]
            date_str = latest.package.timestamp.strftime("%d.%m.%Y")
            summary = latest.decision.verdict_summary
            if len(summary) > 150:
                summary = summary[:150] + "..."

            models_count = len(group.packages)
            models_label = f"{models_count} модел{'ь' if models_count == 1 else 'и' if 2 <= models_count <= 4 else 'ей'}"

            if base_url:
                link_url = f"{base_url}/{group.case_slug}"
            else:
                link_url = f"{group.case_slug}/index.html"

            display_title = (
                title_overrides.get(group.case_slug, group.title)
                if title_overrides
                else group.title
            )

            items_html.append(f"""
            <li class="case-item">
                <a href="{link_url}">
                    <div class="case-title">{escape_html(display_title)}</div>
                    <div class="case-summary">{escape_html(summary)}</div>
                    <div class="case-meta">
                        <span class="case-date">{date_str}</span>
                        <span class="case-models">{models_label}</span>
                    </div>
                </a>
            </li>""")

        return f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Решения ИИ арбитра ЛПР — {escape_html(self.org_name)}</title>
    <style>
        :root {{
            --bg-primary: #fdfcfa;
            --bg-secondary: #f5f3ef;
            --text-primary: #1a1a1a;
            --text-muted: #6b6b6b;
            --accent-color: #8b4513;
            --border-color: #d4d0c8;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: "Charter", "Georgia", serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.7;
            padding: 2rem;
            max-width: 900px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            border-bottom: 2px solid var(--accent-color);
            padding-bottom: 1rem;
            margin-bottom: 2rem;
        }}
        .org-name {{
            text-align: center;
            font-size: 0.85rem;
            letter-spacing: 0.15em;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }}
        .about {{
            margin-bottom: 2.5rem;
            padding: 1.5rem;
            background: var(--bg-secondary);
            border-left: 3px solid var(--accent-color);
        }}
        .about p {{ margin-bottom: 1rem; }}
        .about p:last-child {{ margin-bottom: 0; }}
        .about-title {{
            font-weight: 600;
            margin-bottom: 1rem;
            font-size: 1.1rem;
        }}
        .easter-egg {{
            font-size: 0.85rem;
            color: var(--text-muted);
            font-style: italic;
        }}
        .disclaimer {{
            margin-top: 1.5rem;
            padding-top: 1rem;
            border-top: 1px solid var(--border-color);
            font-size: 0.9rem;
            color: var(--text-muted);
        }}
        .section-title {{
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }}
        .cases-list {{
            list-style: none;
        }}
        .case-item {{
            margin-bottom: 1rem;
        }}
        .case-item a {{
            display: block;
            padding: 1.25rem 1.5rem;
            padding-right: 3rem;
            background: var(--bg-secondary);
            border-radius: 6px;
            border: 1px solid var(--border-color);
            border-left: 3px solid var(--accent-color);
            text-decoration: none;
            color: inherit;
            transition: all 0.2s ease;
            position: relative;
        }}
        .case-item a::after {{
            content: "→";
            position: absolute;
            right: 1.25rem;
            top: 50%;
            transform: translateY(-50%);
            font-size: 1.2rem;
            color: var(--accent-light);
            opacity: 0.5;
            transition: all 0.2s ease;
        }}
        .case-item a:hover {{
            background: linear-gradient(135deg, var(--bg-secondary) 0%, #f0ede8 100%);
            border-color: var(--accent-light);
            box-shadow: 0 4px 12px rgba(139, 69, 19, 0.1);
            transform: translateX(4px);
        }}
        .case-item a:hover::after {{
            opacity: 1;
            transform: translateY(-50%) translateX(3px);
        }}
        .case-title {{
            font-weight: 600;
            color: var(--accent-color);
            margin-bottom: 0.5rem;
            font-size: 1.1rem;
        }}
        .case-item a:hover .case-title {{
            text-decoration: underline;
        }}
        .case-summary {{
            color: var(--text-primary);
            font-size: 0.95rem;
            margin-bottom: 0.75rem;
            line-height: 1.5;
        }}
        .case-meta {{
            display: flex;
            gap: 1rem;
            font-size: 0.8rem;
            color: var(--text-muted);
            align-items: center;
        }}
        .case-date {{
            display: flex;
            align-items: center;
            gap: 0.3rem;
        }}
        .case-models {{
            background: var(--bg-primary);
            padding: 0.2rem 0.6rem;
            border-radius: 12px;
            border: 1px solid var(--border-color);
            font-family: "JetBrains Mono", monospace;
            font-size: 0.7rem;
        }}
        /* Architecture diagram */
        .arch-diagram {{
            margin: 1.5rem 0;
            padding: 1.25rem;
            background: linear-gradient(135deg, #faf9f7 0%, #f0ede8 100%);
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }}
        .arch-flow {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            flex-wrap: wrap;
            margin-bottom: 1rem;
        }}
        .arch-node {{
            text-align: center;
            padding: 0.75rem 1rem;
            background: white;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            min-width: 80px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        .arch-node.arch-agent {{
            border-color: var(--accent-color);
            background: linear-gradient(135deg, #fff 0%, #fdf8f4 100%);
        }}
        .arch-icon {{
            font-size: 1.5rem;
            margin-bottom: 0.25rem;
        }}
        .arch-label {{
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-primary);
        }}
        .arch-sub {{
            font-size: 0.65rem;
            color: var(--text-muted);
            font-family: "JetBrains Mono", monospace;
            margin-top: 0.15rem;
        }}
        .arch-arrow {{
            color: var(--accent-light);
            font-size: 1.2rem;
            font-weight: 300;
        }}
        .arch-corpus {{
            text-align: center;
            padding-top: 0.75rem;
            border-top: 1px dashed var(--border-color);
        }}
        .arch-corpus-title {{
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }}
        .arch-corpus-items {{
            display: flex;
            gap: 0.5rem;
            justify-content: center;
            flex-wrap: wrap;
        }}
        .arch-corpus-items span {{
            font-size: 0.75rem;
            padding: 0.2rem 0.6rem;
            background: white;
            border: 1px solid var(--border-color);
            border-radius: 12px;
            color: var(--text-secondary);
        }}
        .tech-stack {{
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
            margin: 1rem 0;
            padding: 0.75rem 1rem;
            background: #2c2c2c;
            border-radius: 6px;
            font-family: "JetBrains Mono", "Fira Code", monospace;
            font-size: 0.75rem;
        }}
        .tech-item {{
            color: #a0a0a0;
        }}
        .tech-key {{
            color: #cd853f;
        }}
        .tech-key::after {{
            content: ":";
            margin-right: 0.4em;
        }}
        @media (max-width: 600px) {{
            .arch-flow {{
                flex-direction: column;
            }}
            .arch-arrow {{
                transform: rotate(90deg);
            }}
            .tech-stack {{
                flex-direction: column;
                gap: 0.5rem;
            }}
        }}
    </style>
</head>
<body>
    <div class="org-name">{escape_html(self.org_name)}</div>
    <h1>Решения ИИ арбитра</h1>

    <div class="about">
        <div class="about-title">Что это</div>
        <p>RAG-система для анализа партийных документов. Гибридный поиск находит релевантные нормы, LLM строит аргументацию со ссылками.</p>

        <div class="arch-diagram">
            <div class="arch-flow">
                <div class="arch-node arch-input">
                    <div class="arch-icon">📄</div>
                    <div class="arch-label">Запрос</div>
                </div>
                <div class="arch-arrow">→</div>
                <div class="arch-node arch-search">
                    <div class="arch-icon">🔍</div>
                    <div class="arch-label">Hybrid Search</div>
                    <div class="arch-sub">E5 + BM25</div>
                </div>
                <div class="arch-arrow">→</div>
                <div class="arch-node arch-agent">
                    <div class="arch-icon">🧠</div>
                    <div class="arch-label">ReAct Agent</div>
                    <div class="arch-sub">OpenAI / Local</div>
                </div>
                <div class="arch-arrow">→</div>
                <div class="arch-node arch-output">
                    <div class="arch-icon">⚖️</div>
                    <div class="arch-label">Решение</div>
                </div>
            </div>
            <div class="arch-corpus">
                <div class="arch-corpus-title">Корпус</div>
                <div class="arch-corpus-items">
                    <span>Устав</span>
                    <span>Декларация</span>
                    <span>Решения ФК</span>
                    <span>Решения ЭК</span>
                    <span>Прецеденты</span>
                </div>
            </div>
        </div>

        <div class="tech-stack">
            <div class="tech-item"><span class="tech-key">Embedding</span> multilingual-e5-base</div>
            <div class="tech-item"><span class="tech-key">Search</span> Vector + BM25 (α=0.7)</div>
            <div class="tech-item"><span class="tech-key">LLM</span> gpt-5.2-2025-12-11 | qwen/qwen3-30b-a3b-2507 </div>
        </div>

        <p class="disclaimer"><strong>Дисклеймер:</strong> демонстрация возможностей, не официальный орган партии. Ассистент для подготовки решений — не замена человеческому суждению.</p>
    </div>

    <div class="section-title">Рассмотренные дела</div>
    <ul class="cases-list">
        {"".join(items_html)}
    </ul>
</body>
</html>"""

    def export_all(self, audit_dir: Path, output_dir: Path) -> list[Path]:
        """
        Export all cases to output directory.

        Reads cases.json from output_dir for titles and metadata.

        Returns list of created files.
        """
        config = SiteConfig.load(output_dir / "cases.json")
        self.org_name = config.org_name

        contexts = self.load_packages(audit_dir)
        groups = self.group_by_case(contexts)
        created_files: list[Path] = []

        output_dir.mkdir(parents=True, exist_ok=True)

        for group in groups:
            case_dir = output_dir / group.case_slug
            case_dir.mkdir(parents=True, exist_ok=True)
            case_file = case_dir / "index.html"

            custom_title = config.get_case_title(group.case_slug, group.title)
            overrides = CaseOverrides(title=custom_title, org_name=config.org_name)
            case_html = self.export_case(group, overrides=overrides, config=config)
            case_file.write_text(case_html, encoding="utf-8")
            created_files.append(case_file)

        title_overrides = {
            slug: config.get_case_title(slug, g.title)
            for g in groups
            for slug in [g.case_slug]
        }
        index_html = self.export_index(groups, title_overrides=title_overrides)
        index_file = output_dir / "index.html"
        index_file.write_text(index_html, encoding="utf-8")
        created_files.append(index_file)

        return created_files
