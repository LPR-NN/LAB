"""Page rendering with Jinja2 templates."""

import html
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from source.contracts.audit import DecisionPackage
from source.contracts.decision import Decision
from source.contracts.request import Request

# Template directory
TEMPLATES_DIR = Path(__file__).parent / "templates"

# Create Jinja2 environment
_env = Environment(
    loader=FileSystemLoader(TEMPLATES_DIR),
    autoescape=select_autoescape(["html", "xml"]),
)


@dataclass
class SiteConfig:
    """Site configuration from cases.json."""

    org_name: str = "ЛПР Лаб"
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
        """Get rank for a model (1 = best, higher = worse)."""
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


# Labels
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


def _slugify(text: str) -> str:
    """Convert text to URL-friendly slug."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s]+", "-", text)
    return text.strip("-")[:50]


def _get_model_slug(model_id: str) -> str:
    """Convert model ID to a slug."""
    slug = model_id.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")


def _make_section_anchor(section: str) -> str:
    """Convert section reference like 'п. 3.4.1.2' to anchor '#p-3-4-1-2'."""
    match = re.search(r"(\d+(?:\.\d+)+)", section)
    if match:
        number = match.group(1)
        return f"#p-{number.replace('.', '-')}"
    return ""


def _linkify_corpus_citation(text: str) -> str:
    """Convert corpus citations to clickable links."""
    pattern = r"\[([A-ZА-ЯЁ][A-ZА-ЯЁ0-9_-]+(?:-[A-ZА-ЯЁ0-9_-]+)*)(?:,\s*([^\]]+))?\]"

    def replace_citation(match: re.Match[str]) -> str:
        citation_key = match.group(1)
        section_info = match.group(2)
        url = f"/corpus/by-key/{citation_key}"
        if section_info:
            anchor = _make_section_anchor(section_info)
            url += anchor
            return (
                f'[<a href="{url}" class="corpus-link" target="_blank">'
                f"{html.escape(citation_key)}</a>, {html.escape(section_info)}]"
            )
        else:
            return f'[<a href="{url}" class="corpus-link" target="_blank">{html.escape(citation_key)}</a>]'

    return re.sub(pattern, replace_citation, text)


def _linkify_evidence_ref(ref: str) -> str:
    """Convert evidence reference to a clickable link."""
    ref = ref.strip()
    path_match = re.match(r"^((?:assets|data)/[^\s(]+)", ref)
    if not path_match:
        return html.escape(ref)

    file_path = path_match.group(1)
    suffix = ref[len(file_path) :]

    if file_path.startswith("assets/"):
        url = f"/attachments/{file_path[7:]}"
    elif file_path.startswith("data/"):
        url = f"/corpus/{file_path[5:]}"
    else:
        return html.escape(ref)

    link = f'<a href="{url}" class="evidence-link" target="_blank">{html.escape(file_path)}</a>'
    if suffix:
        link += html.escape(suffix)
    return link


class PageRenderer:
    """Renders pages using Jinja2 templates."""

    def __init__(self, static_path: Path, audit_path: Path):
        self.static_path = static_path
        self.audit_path = audit_path
        self.config = SiteConfig.load(static_path / "cases.json")

    def render_home(self) -> str:
        """Render home page."""
        template = _env.get_template("home.html")
        return template.render()

    def render_dumatel_index(self) -> str:
        """Render Думатель index with list of cases."""
        cases = self._load_cases_summary()
        template = _env.get_template("dumatel/index.html")
        return template.render(cases=cases)

    def render_decision(self, case_slug: str) -> str | None:
        """Render decision page for a case."""
        groups = self._load_case_groups()

        # Find the group
        group = None
        for g in groups:
            if g.case_slug == case_slug:
                group = g
                break

        if not group:
            return None

        # Sort by model rank
        sorted_packages = sorted(
            group.packages,
            key=lambda ctx: self.config.get_model_rank(ctx.package.model_id),
        )

        # Prepare decision data
        decisions = []
        for ctx in sorted_packages:
            model_slug = _get_model_slug(ctx.package.model_id)
            is_disabled = self.config.is_model_disabled(ctx.package.model_id)
            decisions.append(
                {
                    "model_id": ctx.package.model_id,
                    "model_slug": model_slug,
                    "date": ctx.package.timestamp.strftime("%d.%m"),
                    "disabled": is_disabled,
                    "content": self._render_decision_content(ctx)
                    if not is_disabled
                    else "",
                }
            )

        case_title = self.config.get_case_title(case_slug, group.title)

        template = _env.get_template("dumatel/decision.html")
        return template.render(
            case_title=case_title,
            org_name=self.config.org_name,
            versions_count=len(decisions),
            decisions=decisions,
        )

    def render_putevoditel(self) -> str:
        """Render Путеводитель demo page."""
        template = _env.get_template("putevoditel.html")
        return template.render()

    def _load_case_groups(self) -> list[CaseGroup]:
        """Load and group all decisions from audit_logs."""
        if not self.audit_path.exists():
            return []

        contexts: list[DecisionContext] = []
        for json_file in self.audit_path.glob("*.json"):
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

        # Group by case
        groups: dict[str, CaseGroup] = {}
        for ctx in contexts:
            case_id = self._get_case_id(ctx)
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

    def _load_cases_summary(self) -> list[dict]:
        """Load summary of all cases for listing."""
        groups = self._load_case_groups()
        cases = []

        for group in groups:
            latest = group.packages[0]
            summary = latest.decision.verdict_summary
            if len(summary) > 150:
                summary = summary[:150] + "..."

            models_count = len(group.packages)
            if models_count == 1:
                models_label = "модель"
            elif 2 <= models_count <= 4:
                models_label = "модели"
            else:
                models_label = "моделей"

            cases.append(
                {
                    "slug": group.case_slug,
                    "title": self.config.get_case_title(group.case_slug, group.title),
                    "summary": summary,
                    "date": latest.package.timestamp.strftime("%d.%m.%Y"),
                    "models_count": models_count,
                    "models_label": models_label,
                }
            )

        return cases

    def _get_case_id(self, ctx: DecisionContext) -> str:
        """Extract case ID from a decision context."""
        import hashlib

        if ctx.request.case_id:
            return ctx.request.case_id
        if ctx.request.attachments:
            first_attachment = ctx.request.attachments[0].file_path
            return hashlib.md5(first_attachment.encode()).hexdigest()[:8]
        return hashlib.md5(ctx.request.query.encode()).hexdigest()[:8]

    def _render_decision_content(self, ctx: DecisionContext) -> str:
        """Render the content of a single decision (without wrapper)."""
        parts = []

        # Verdict
        parts.append(self._render_verdict(ctx))

        # Parties
        if ctx.request.parties:
            parts.append(self._render_parties(ctx))

        # Facts
        if ctx.decision.findings_of_fact:
            parts.append(self._render_facts(ctx))

        # Norms
        if ctx.decision.applicable_norms:
            parts.append(self._render_norms(ctx))

        # Reasoning
        if ctx.decision.reasoning:
            parts.append(self._render_reasoning(ctx))

        # Citations
        if ctx.decision.citations:
            parts.append(self._render_citations(ctx))

        # Notes
        notes = self._render_notes(ctx)
        if notes:
            parts.append(notes)

        # Next steps
        if ctx.decision.recommended_next_steps:
            parts.append(self._render_next_steps(ctx))

        # Footer
        parts.append(self._render_footer(ctx))

        return "\n".join(parts)

    def _render_verdict(self, ctx: DecisionContext) -> str:
        """Render verdict banner."""
        return f"""
        <div class="verdict-banner">
            <div class="verdict-type">{html.escape(VERDICT_LABELS.get(ctx.decision.verdict, ctx.decision.verdict))}</div>
            <div class="verdict-summary">{html.escape(ctx.decision.verdict_summary)}</div>
        </div>"""

    def _render_parties(self, ctx: DecisionContext) -> str:
        """Render parties section."""
        parties_html = []
        for party in ctx.request.parties:
            role_label = ROLE_LABELS.get(party.role, party.role)
            desc = (
                f'<div class="party-desc">{html.escape(party.description)}</div>'
                if party.description
                else ""
            )
            parties_html.append(f"""
                <div class="party-card">
                    <div class="party-role">{html.escape(role_label)}</div>
                    <div class="party-name">{html.escape(party.name)}</div>
                    {desc}
                </div>""")

        return f"""
        <section class="section">
            <h2 class="section-title"><span class="section-number">I.</span> Стороны</h2>
            <div class="parties-grid">
                {"".join(parties_html)}
            </div>
        </section>"""

    def _render_facts(self, ctx: DecisionContext) -> str:
        """Render findings of fact."""
        facts_html = []
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
                    <div class="fact-text">{html.escape(fact.fact)}</div>
                    {refs_html}
                </li>""")

        return f"""
        <section class="section">
            <h2 class="section-title"><span class="section-number">II.</span> Установленные факты</h2>
            <ol class="facts-list">
                {"".join(facts_html)}
            </ol>
        </section>"""

    def _render_norms(self, ctx: DecisionContext) -> str:
        """Render applicable norms."""
        norms_html = []
        for norm in ctx.decision.applicable_norms:
            citation_with_section = f"[{norm.citation_key}, {norm.section}]"
            citation_link = _linkify_corpus_citation(citation_with_section)
            norms_html.append(f"""
                <li class="norm-item">
                    <div class="norm-citation">{citation_link}</div>
                    <div class="norm-relevance">{html.escape(norm.relevance)}</div>
                </li>""")

        return f"""
        <section class="section">
            <h2 class="section-title"><span class="section-number">III.</span> Применимые нормы</h2>
            <ul class="norms-list">
                {"".join(norms_html)}
            </ul>
        </section>"""

    def _render_reasoning(self, ctx: DecisionContext) -> str:
        """Render reasoning chain."""
        steps_html = []
        for step in ctx.decision.reasoning:
            norm_html = _linkify_corpus_citation(html.escape(step.norm_or_precedent))
            steps_html.append(f"""
                <div class="reasoning-step reasoning-fact">
                    <div class="reasoning-label">Факт</div>
                    <div class="reasoning-content">{html.escape(step.fact)}</div>
                </div>
                <div class="reasoning-step reasoning-norm">
                    <div class="reasoning-label">Норма</div>
                    <div class="reasoning-content">{norm_html}</div>
                </div>
                <div class="reasoning-step reasoning-conclusion">
                    <div class="reasoning-label">Вывод</div>
                    <div class="reasoning-content">{html.escape(step.conclusion)}</div>
                </div>""")

        return f"""
        <section class="section">
            <h2 class="section-title"><span class="section-number">IV.</span> Обоснование</h2>
            <div class="reasoning-chain">
                {"".join(steps_html)}
            </div>
        </section>"""

    def _render_citations(self, ctx: DecisionContext) -> str:
        """Render citations."""
        citations_html = []
        for citation in ctx.decision.citations:
            citation_with_section = f"[{citation.doc_id}, {citation.section}]"
            doc_link = _linkify_corpus_citation(citation_with_section)
            citations_html.append(f"""
                <li class="citation-item">
                    <div class="citation-ref">{doc_link}</div>
                    <blockquote class="citation-text">«{html.escape(citation.quoted_text)}»</blockquote>
                </li>""")

        return f"""
        <section class="section">
            <h2 class="section-title"><span class="section-number">V.</span> Цитаты</h2>
            <ul class="citations-list">
                {"".join(citations_html)}
            </ul>
        </section>"""

    def _render_notes(self, ctx: DecisionContext) -> str:
        """Render uncertainty and minority view."""
        notes_html = []

        if ctx.decision.uncertainty:
            notes_html.append(f"""
            <div class="note-box uncertainty">
                <div class="note-title">Неопределённость</div>
                <div class="note-content">{html.escape(ctx.decision.uncertainty)}</div>
            </div>""")

        if ctx.decision.minority_view:
            notes_html.append(f"""
            <div class="note-box minority">
                <div class="note-title">Альтернативная позиция</div>
                <div class="note-content">{html.escape(ctx.decision.minority_view)}</div>
            </div>""")

        if not notes_html:
            return ""

        return f"""
        <section class="section">
            <h2 class="section-title"><span class="section-number">VI.</span> Примечания</h2>
            {"".join(notes_html)}
        </section>"""

    def _render_next_steps(self, ctx: DecisionContext) -> str:
        """Render recommended next steps."""
        steps_html = []
        for step in ctx.decision.recommended_next_steps:
            steps_html.append(f'<li class="next-step-item">{html.escape(step)}</li>')

        return f"""
        <section class="section">
            <h2 class="section-title"><span class="section-number">VII.</span> Рекомендуемые действия</h2>
            <ol class="next-steps-list">
                {"".join(steps_html)}
            </ol>
        </section>"""

    def _render_footer(self, ctx: DecisionContext) -> str:
        """Render decision footer."""
        timestamp = ctx.package.timestamp
        date_str = timestamp.strftime("%d.%m.%Y")
        short_id = ctx.package.package_id[:8]

        return f"""
        <footer class="decision-footer">
            <div>
                <div class="footer-date">{date_str}</div>
                <div class="footer-signature">Думатель</div>
            </div>
            <div class="tech-widget">
                <div><span class="tech-label">model:</span> <span class="tech-value">{html.escape(ctx.package.model_id)}</span></div>
                <div><span class="tech-label">agent:</span> <span class="tech-value">{html.escape(ctx.package.agent_version)}</span></div>
                <div><span class="tech-label">id:</span> <span class="tech-value">{html.escape(short_id)}</span></div>
            </div>
        </footer>"""
