"""Chat functionality with RAG and rate limiting."""

import html
import json
import logging
import re
from collections import defaultdict
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kungfu import Nothing, Ok, Option, Some
from openai import AsyncOpenAI
from pydantic import BaseModel

from funcai import Dialogue, message
from funcai.core.provider import ABCAIProvider
from source.corpus.index import CorpusIndex
from source.corpus.loader import DocumentLoader
from source.providers.factory import ProviderFactory
from source.settings import ProviderType, get_settings

logger = logging.getLogger(__name__)


@dataclass
class ChatConfig:
    """Chat configuration loaded from cases.json."""

    models: list[dict[str, str]] = field(default_factory=list)  # type: ignore[type-arg]
    daily_limit: int = 5
    max_question_length: int = 2000

    @classmethod
    def load(cls, config_path: Path) -> "ChatConfig":
        """Load config from cases.json."""
        if not config_path.exists():
            return cls()
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            return cls(
                models=data.get("chat_models", []),
                daily_limit=data.get("chat_daily_limit", 5),
                max_question_length=data.get("chat_max_question_length", 2000),
            )
        except Exception:
            return cls()

    def is_model_allowed(self, provider: str, model: str) -> bool:
        """Check if provider/model combination is in whitelist."""
        for m in self.models:
            if m.get("provider") == provider and m.get("model") == model:
                return True
        return False

    def get_default_model(self) -> dict[str, str] | None:
        """Get the first available model as default."""
        return self.models[0] if self.models else None


class RateLimiter:
    """
    In-memory rate limiter for chat requests.

    Tracks requests per username per day.
    Resets at midnight UTC.
    """

    def __init__(self, daily_limit: int = 5):
        self.daily_limit = daily_limit
        # username -> list of request timestamps
        self._requests: dict[str, list[datetime]] = defaultdict(list)

    def _clean_old_requests(self, username: str) -> None:
        """Remove requests from previous days."""
        today = datetime.now(UTC).date()
        self._requests[username] = [
            ts for ts in self._requests[username] if ts.date() == today
        ]

    def check_limit(self, username: str) -> tuple[bool, int]:
        """
        Check if user can make a request.

        Returns:
            (allowed, remaining) - whether request is allowed and how many remain
        """
        self._clean_old_requests(username)
        used = len(self._requests[username])
        remaining = max(0, self.daily_limit - used)
        return remaining > 0, remaining

    def record_request(self, username: str) -> int:
        """
        Record a request and return remaining count.

        Should be called AFTER successful request processing.
        """
        self._clean_old_requests(username)
        self._requests[username].append(datetime.now(UTC))
        remaining = max(0, self.daily_limit - len(self._requests[username]))
        return remaining

    def get_remaining(self, username: str) -> int:
        """Get remaining requests for user."""
        self._clean_old_requests(username)
        used = len(self._requests[username])
        return max(0, self.daily_limit - used)


class ChatResponse(BaseModel):
    """Response from chat API."""

    answer: str
    sources: list[dict[str, str]]
    remaining: int


CHAT_SYSTEM_PROMPT = """Ты — помощник по партийным документам Либертарианской партии России.
Отвечай на вопросы, используя ТОЛЬКО информацию из предоставленного контекста.

## Формат ответа

Структурируй ответ ЧЁТКО:
1. **Краткий ответ** — 1-2 предложения с сутью (если возможно)
2. **Детали** — раскрытие по пунктам, если нужно
3. **Источники** — в конце, списком

## Правила форматирования

- Пиши чисто и по делу, без воды
- Используй нумерованные/маркированные списки для перечислений
- Ссылки на источники: **[КЛЮЧ, п. X.Y]** — только ключ и номер пункта
- НЕ копируй технические артефакты из документов: {{#org...}}, {{.section-number...}}, ::: div, []{{.done}}, {{.subtitle}} и т.п.
- НЕ цитируй заголовки редакций типа "Исправления в пунктуации (Имя)"
- Если цитируешь текст — приводи только суть нормы, без разметки

## Стиль

- Лаконично: не повторяй вопрос, не пиши "на основании документов"
- Профессионально: юридический стиль, без эмоций
- Если информации нет — скажи прямо: "В документах не нашлось информации о..."

Контекст из базы знаний:
{context}
"""


# Patterns for cleaning org-mode/pandoc artifacts from content
_CLEANUP_PATTERNS = [
    # Org-mode anchors: {#org1234567}
    (re.compile(r"\{#org[a-f0-9]+\}"), ""),
    # Pandoc div markers: ::: {#outline-container-org... .outline-2}
    (re.compile(r"::+\s*\{[^}]*\}"), ""),
    # Closing div markers: :::
    (re.compile(r"^::+$", re.MULTILINE), ""),
    # Section number spans: {.section-number-2}
    (re.compile(r"\{\.section-number-\d+\}"), ""),
    # Done/todo markers: {.done .___...}
    (re.compile(r"\{\.[a-z_]+(?:\s+\.[a-z_]+)*\}"), ""),
    # Subtitle spans: {.subtitle}
    (re.compile(r"\{\.subtitle\}"), ""),
    # Empty square brackets from cleaned spans: []
    (re.compile(r"\[\]"), ""),
    # Pandoc postamble div: ::: {#postamble ...} ... :::
    (re.compile(r"::: \{#postamble[^}]*\}.*?:::", re.DOTALL), ""),
    # Content div wrapper: :::::: {#content .content}
    (re.compile(r":{2,}\s*\{#content[^}]*\}"), ""),
    # Closing content div
    (re.compile(r"^:{2,}$", re.MULTILINE), ""),
    # Multiple blank lines → single
    (re.compile(r"\n{3,}"), "\n\n"),
]


def _clean_context(text: str) -> str:
    """Remove org-mode/pandoc artifacts from context text."""
    for pattern, replacement in _CLEANUP_PATTERNS:
        text = pattern.sub(replacement, text)
    return text.strip()


class ChatService:
    """
    Service for RAG-based chat.

    Combines corpus search with LLM for answering questions.
    """

    def __init__(
        self,
        corpus: CorpusIndex,
        config: ChatConfig,
        rate_limiter: RateLimiter,
    ):
        self.corpus = corpus
        self.config = config
        self.rate_limiter = rate_limiter

    def validate_request(
        self,
        question: str,
        provider: str,
        model: str,
        username: str,
    ) -> str | None:
        """
        Validate chat request.

        Returns error message or None if valid.
        """
        # Check rate limit FIRST (before any processing)
        allowed, _ = self.rate_limiter.check_limit(username)
        if not allowed:
            return "Лимит вопросов исчерпан. Попробуйте завтра."

        # Validate question length
        if len(question) > self.config.max_question_length:
            return f"Вопрос слишком длинный (макс. {self.config.max_question_length} символов)"

        if len(question.strip()) < 3:
            return "Вопрос слишком короткий"

        # Validate provider/model against whitelist
        if not self.config.is_model_allowed(provider, model):
            return "Недопустимая модель"

        return None

    def _build_context(
        self, question: str, top_k: int = 5
    ) -> tuple[str, list[dict[str, str]]]:
        """
        Build context from corpus search.

        Returns:
            (context_text, sources_list)
        """
        results = self.corpus.search_chunks(question, top_k=top_k)

        context_parts: list[str] = []
        sources: list[dict[str, str]] = []
        seen_docs: set[str] = set()

        for result in results:
            chunk = result.chunk
            doc_key = f"{chunk.citation_key}:{chunk.section or 'full'}"

            if doc_key in seen_docs:
                continue
            seen_docs.add(doc_key)

            # Clean content from org-mode/pandoc artifacts
            clean_content = _clean_context(chunk.content)

            # Build context entry
            section_str = f", {chunk.section}" if chunk.section else ""
            context_parts.append(
                f"[{chunk.citation_key}{section_str}]\n{clean_content}\n"
            )

            # Track source
            sources.append(
                {
                    "doc_id": chunk.doc_id,
                    "citation_key": chunk.citation_key,
                    "section": chunk.section or "",
                    "title": chunk.title or "",
                }
            )

        return "\n---\n".join(context_parts), sources

    async def answer(
        self,
        question: str,
        provider_type: ProviderType,
        model: str,
        username: str,
    ) -> ChatResponse:
        """
        Answer a question using RAG.

        Args:
            question: User question
            provider_type: LLM provider type
            model: Model identifier
            username: User for rate limiting

        Returns:
            ChatResponse with answer, sources, and remaining requests
        """
        settings = get_settings()

        # Build context from corpus
        context, sources = self._build_context(question, top_k=5)

        if not context:
            # No relevant documents found
            remaining = self.rate_limiter.record_request(username)
            return ChatResponse(
                answer="К сожалению, я не нашёл релевантных документов по вашему вопросу. "
                "Попробуйте переформулировать или задать более конкретный вопрос.",
                sources=[],
                remaining=remaining,
            )

        # Create provider
        api_key = settings.get_api_key(provider_type)
        provider: ABCAIProvider[Any] = ProviderFactory.create(
            provider_type=provider_type,
            model=model,
            api_key=api_key,
        )

        # Build dialogue
        system_prompt = CHAT_SYSTEM_PROMPT.format(context=context)
        dialogue = Dialogue(
            [
                message.system(text=system_prompt),
                message.user(text=question),
            ]
        )

        # Get response
        result = await dialogue.interpret(provider)

        match result:
            case Ok(response):
                answer_text = response.message.text.unwrap_or(
                    "Не удалось получить ответ"
                )
            case _:
                answer_text = (
                    "Произошла ошибка при обработке запроса. Попробуйте позже."
                )

        # Record request AFTER successful processing
        remaining = self.rate_limiter.record_request(username)

        return ChatResponse(
            answer=answer_text,
            sources=sources,
            remaining=remaining,
        )

    async def answer_stream(
        self,
        question: str,
        provider_type: ProviderType,
        model: str,
        username: str,
    ) -> AsyncGenerator[str, None]:
        """
        Answer a question using RAG with streaming response.

        Yields SSE-formatted chunks: data: {"type": "...", ...}

        Args:
            question: User question
            provider_type: LLM provider type
            model: Model identifier
            username: User for rate limiting

        Yields:
            SSE data chunks
        """
        settings = get_settings()

        # Status update: searching corpus
        logger.info("[stream] Starting for user=%s, model=%s", username, model)
        yield json.dumps({"type": "status", "status": "Поиск по базе знаний..."})

        # Build context from corpus
        context, sources = self._build_context(question, top_k=5)
        logger.info("[stream] Found %d sources", len(sources))

        if not context:
            remaining = self.rate_limiter.record_request(username)
            yield json.dumps({"type": "sources", "sources": []})
            yield json.dumps(
                {
                    "type": "content",
                    "content": "К сожалению, я не нашёл релевантных документов по вашему вопросу.",
                }
            )
            yield json.dumps({"type": "done", "remaining": remaining})
            return

        # Send sources first
        yield json.dumps({"type": "sources", "sources": sources})

        # Status update: calling LLM
        yield json.dumps({"type": "status", "status": "Запрос к модели..."})

        # Get API key and base URL based on provider
        api_key = settings.get_api_key(provider_type)
        base_url: Option[str] = Nothing()

        if provider_type == "openrouter":
            base_url = Some("https://openrouter.ai/api/v1")
        elif provider_type == "lmstudio":
            base_url = Some("http://localhost:1234/v1")

        # Create OpenAI client (works for OpenAI-compatible APIs)
        client = AsyncOpenAI(
            api_key=api_key or "dummy",
            base_url=base_url.unwrap_or_none(),
        )

        system_prompt = CHAT_SYSTEM_PROMPT.format(context=context)
        logger.info(
            "[stream] Calling %s/%s, context=%d chars", provider_type, model, len(context)
        )

        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                stream=True,
            )
            logger.info("[stream] Stream started, receiving chunks...")

            chunk_count = 0
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    chunk_count += 1
                    yield json.dumps({"type": "content", "content": content})

            logger.info("[stream] Stream complete, %d chunks received", chunk_count)

        except Exception as e:
            logger.exception("Streaming error for user=%s model=%s", username, model)
            yield json.dumps(
                {"type": "error", "error": "Ошибка при обработке запроса"}
            )

        # Record request and send done
        remaining = self.rate_limiter.record_request(username)
        yield json.dumps({"type": "done", "remaining": remaining})
        logger.info("[stream] Done, remaining=%d", remaining)


def render_chat_page(
    config: ChatConfig,
    remaining: int,
    org_name: str = "LLM инициатива",
) -> str:
    """Render the chat page HTML."""

    # Build model options
    model_options: list[str] = []
    for m in config.models:
        label = html.escape(m.get("label", m.get("model", "Unknown")))
        provider = html.escape(m.get("provider", ""))
        model = html.escape(m.get("model", ""))
        model_options.append(f'<option value="{provider}|{model}">{label}</option>')

    model_select = (
        "\n".join(model_options)
        if model_options
        else '<option value="">Нет доступных моделей</option>'
    )

    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Чат с арбитром — {html.escape(org_name)}</title>
    <style>
        :root {{
            --bg-primary: #fdfcfa;
            --bg-secondary: #f5f3ef;
            --bg-accent: #e8e4dc;
            --text-primary: #1a1a1a;
            --text-secondary: #4a4a4a;
            --text-muted: #6b6b6b;
            --border-color: #d4d0c8;
            --accent-color: #8b4513;
            --accent-light: #cd853f;
            --success: #2d5a3d;
            --error: #8b1a1a;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: "Charter", "Georgia", serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }}

        .header {{
            background: var(--bg-secondary);
            border-bottom: 2px solid var(--accent-color);
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .header-left {{
            display: flex;
            align-items: center;
            gap: 1rem;
        }}

        .back-link {{
            color: var(--accent-color);
            text-decoration: none;
            font-size: 0.9rem;
        }}

        .back-link:hover {{
            text-decoration: underline;
        }}

        .header-title {{
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--text-primary);
        }}

        .header-right {{
            display: flex;
            align-items: center;
            gap: 1rem;
        }}

        .limit-badge {{
            font-size: 0.8rem;
            color: var(--text-muted);
            background: var(--bg-primary);
            padding: 0.3rem 0.8rem;
            border-radius: 12px;
            border: 1px solid var(--border-color);
        }}

        .limit-badge strong {{
            color: var(--accent-color);
        }}

        .model-select {{
            padding: 0.4rem 0.8rem;
            border: 1px solid var(--border-color);
            border-radius: 4px;
            background: var(--bg-primary);
            font-family: inherit;
            font-size: 0.85rem;
            color: var(--text-primary);
            cursor: pointer;
        }}

        .main {{
            flex: 1;
            display: flex;
            flex-direction: column;
            max-width: 900px;
            width: 100%;
            margin: 0 auto;
            padding: 2rem;
        }}

        .chat-area {{
            flex: 1;
            overflow-y: auto;
            margin-bottom: 1.5rem;
        }}

        .guide {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-left: 4px solid var(--accent-color);
            border-radius: 0 8px 8px 0;
            padding: 1.5rem;
            margin-bottom: 1rem;
        }}

        .guide-title {{
            font-size: 1rem;
            font-weight: 600;
            color: var(--accent-color);
            margin-bottom: 1rem;
        }}

        .guide p {{
            margin-bottom: 0.8rem;
            color: var(--text-secondary);
        }}

        .guide ul {{
            list-style: none;
            margin: 1rem 0;
        }}

        .guide li {{
            padding: 0.3rem 0;
            padding-left: 1.2rem;
            position: relative;
            color: var(--text-secondary);
        }}

        .guide li::before {{
            content: "•";
            position: absolute;
            left: 0;
            color: var(--accent-light);
        }}

        .guide-section {{
            margin-top: 1.2rem;
            padding-top: 1rem;
            border-top: 1px solid var(--border-color);
        }}

        .guide-section-title {{
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }}

        .guide-note {{
            font-size: 0.85rem;
            color: var(--text-muted);
            font-style: italic;
        }}

        .message {{
            margin-bottom: 1.5rem;
            padding: 1rem 1.25rem;
            border-radius: 8px;
        }}

        .message-user {{
            background: var(--bg-accent);
            border: 1px solid var(--border-color);
            margin-left: 2rem;
        }}

        .message-assistant {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            margin-right: 2rem;
        }}

        .message-label {{
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }}

        .message-content {{
            white-space: pre-wrap;
            word-wrap: break-word;
        }}

        .sources {{
            margin-top: 1rem;
            padding-top: 0.75rem;
            border-top: 1px solid var(--border-color);
        }}

        .sources-title {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }}

        .source-link {{
            display: inline-block;
            font-size: 0.8rem;
            color: var(--accent-color);
            text-decoration: none;
            margin-right: 1rem;
            margin-bottom: 0.3rem;
            font-family: "JetBrains Mono", monospace;
        }}

        .source-link:hover {{
            text-decoration: underline;
        }}

        .input-area {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
        }}

        .input-wrapper {{
            display: flex;
            gap: 0.75rem;
        }}

        .question-input {{
            flex: 1;
            padding: 0.75rem 1rem;
            border: 1px solid var(--border-color);
            border-radius: 4px;
            font-family: inherit;
            font-size: 1rem;
            resize: none;
            min-height: 60px;
            background: var(--bg-primary);
        }}

        .question-input:focus {{
            outline: none;
            border-color: var(--accent-color);
        }}

        .send-btn {{
            padding: 0.75rem 1.5rem;
            background: var(--accent-color);
            color: white;
            border: none;
            border-radius: 4px;
            font-family: inherit;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s;
            align-self: flex-end;
        }}

        .send-btn:hover:not(:disabled) {{
            background: var(--accent-light);
        }}

        .send-btn:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
        }}

        .input-footer {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 0.5rem;
            font-size: 0.8rem;
            color: var(--text-muted);
        }}

        .char-counter {{
            font-family: "JetBrains Mono", monospace;
        }}

        .loading {{
            display: none;
            align-items: center;
            gap: 0.5rem;
            color: var(--text-muted);
        }}

        .loading.visible {{
            display: flex;
        }}

        .spinner {{
            width: 16px;
            height: 16px;
            border: 2px solid var(--border-color);
            border-top-color: var(--accent-color);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }}

        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}

        .error {{
            background: #fef2f2;
            border: 1px solid #fecaca;
            color: var(--error);
            padding: 0.75rem 1rem;
            border-radius: 4px;
            margin-bottom: 1rem;
            display: none;
        }}

        .error.visible {{
            display: block;
        }}

        @media (max-width: 600px) {{
            .header {{
                flex-direction: column;
                gap: 1rem;
                padding: 1rem;
            }}

            .main {{
                padding: 1rem;
            }}

            .message-user {{
                margin-left: 0;
            }}

            .message-assistant {{
                margin-right: 0;
            }}

            .input-wrapper {{
                flex-direction: column;
            }}

            .send-btn {{
                align-self: stretch;
            }}
        }}
    </style>
</head>
<body>
    <header class="header">
        <div class="header-left">
            <a href="/" class="back-link">← Назад</a>
            <span class="header-title">Чат с арбитром</span>
        </div>
        <div class="header-right">
            <span class="limit-badge">Осталось: <strong id="remaining">{remaining}</strong>/{config.daily_limit}</span>
            <select class="model-select" id="model-select">
                {model_select}
            </select>
        </div>
    </header>

    <main class="main">
        <div class="chat-area" id="chat-area">
            <div class="guide" id="guide">
                <div class="guide-title">Как использовать</div>
                <p>Задайте вопрос по партийным документам — устав, решения ЭК/ФК, этический кодекс и прецеденты.</p>

                <ul>
                    <li>Какие полномочия у Этического комитета?</li>
                    <li>Как обжаловать решение Федерального комитета?</li>
                    <li>Какие санкции предусмотрены за нарушение устава?</li>
                </ul>

                <div class="guide-section">
                    <div class="guide-section-title">Вложения</div>
                    <p>Прикладывать файлы нельзя. Если нужно обсудить изображение — используйте другую нейросеть (ChatGPT, Claude) для описания, затем вставьте текстовое описание в вопрос.</p>
                </div>

                <p class="guide-note">Лимит: {config.daily_limit} вопросов в день (LLM запросы платные)</p>
            </div>
        </div>

        <div class="error" id="error"></div>

        <div class="input-area">
            <div class="input-wrapper">
                <textarea
                    class="question-input"
                    id="question-input"
                    placeholder="Задайте вопрос..."
                    maxlength="{config.max_question_length}"
                    rows="2"
                ></textarea>
                <button class="send-btn" id="send-btn">Отправить</button>
            </div>
            <div class="input-footer">
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <span id="loading-status">Обработка...</span>
                </div>
                <div class="char-counter">
                    <span id="char-count">0</span>/{config.max_question_length}
                </div>
            </div>
        </div>
    </main>

    <script>
        const chatArea = document.getElementById('chat-area');
        const guide = document.getElementById('guide');
        const errorEl = document.getElementById('error');
        const questionInput = document.getElementById('question-input');
        const sendBtn = document.getElementById('send-btn');
        const modelSelect = document.getElementById('model-select');
        const remainingEl = document.getElementById('remaining');
        const charCount = document.getElementById('char-count');
        const loading = document.getElementById('loading');
        const loadingStatus = document.getElementById('loading-status');

        let isProcessing = false;
        let currentMessageDiv = null;
        let currentContentDiv = null;
        let currentSources = null;

        function escapeHtml(text) {{
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }}

        function updateCharCount() {{
            charCount.textContent = questionInput.value.length;
        }}

        function showError(message) {{
            errorEl.textContent = message;
            errorEl.classList.add('visible');
        }}

        function hideError() {{
            errorEl.classList.remove('visible');
        }}

        function setLoading(state, status) {{
            isProcessing = state;
            sendBtn.disabled = state;
            loading.classList.toggle('visible', state);
            if (status) {{
                loadingStatus.textContent = status;
            }} else {{
                loadingStatus.textContent = 'Обработка...';
            }}
        }}

        function setStatus(status) {{
            loadingStatus.textContent = status;
            console.log('[chat] Status:', status);
        }}

        function addUserMessage(content) {{
            guide.style.display = 'none';

            const messageDiv = document.createElement('div');
            messageDiv.className = 'message message-user';
            messageDiv.innerHTML = '<div class="message-label">Вы</div>' +
                '<div class="message-content">' + escapeHtml(content) + '</div>';
            chatArea.appendChild(messageDiv);
            chatArea.scrollTop = chatArea.scrollHeight;
        }}

        function createAssistantMessage() {{
            guide.style.display = 'none';

            currentMessageDiv = document.createElement('div');
            currentMessageDiv.className = 'message message-assistant';
            currentMessageDiv.innerHTML = '<div class="message-label">Арбитр</div>' +
                '<div class="message-content"></div>';
            currentContentDiv = currentMessageDiv.querySelector('.message-content');
            chatArea.appendChild(currentMessageDiv);
            chatArea.scrollTop = chatArea.scrollHeight;
        }}

        function appendToAssistantMessage(text) {{
            if (currentContentDiv) {{
                currentContentDiv.textContent += text;
                chatArea.scrollTop = chatArea.scrollHeight;
            }}
        }}

        function finalizeAssistantMessage() {{
            if (currentMessageDiv && currentSources && currentSources.length > 0) {{
                let sourcesHtml = '<div class="sources"><div class="sources-title">Источники</div>';
                currentSources.forEach(function(s) {{
                    const key = escapeHtml(s.citation_key);
                    const section = s.section ? ', ' + escapeHtml(s.section) : '';
                    const url = '/corpus/by-key/' + encodeURIComponent(s.citation_key);
                    sourcesHtml += '<a href="' + url + '" class="source-link" target="_blank">[' + key + section + ']</a>';
                }});
                sourcesHtml += '</div>';
                currentMessageDiv.insertAdjacentHTML('beforeend', sourcesHtml);
            }}
            currentMessageDiv = null;
            currentContentDiv = null;
            currentSources = null;
        }}

        async function sendQuestion() {{
            const question = questionInput.value.trim();
            if (!question || isProcessing) return;

            const remaining = parseInt(remainingEl.textContent);
            if (remaining <= 0) {{
                showError('Лимит вопросов исчерпан. Попробуйте завтра.');
                return;
            }}

            hideError();
            setLoading(true, 'Отправка запроса...');

            const modelValue = modelSelect.value;
            const [provider, model] = modelValue.split('|');
            console.log('[chat] Sending question to', provider, model);

            addUserMessage(question);
            questionInput.value = '';
            updateCharCount();

            try {{
                const response = await fetch('/api/chat/stream', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json',
                    }},
                    body: JSON.stringify({{
                        question: question,
                        provider: provider,
                        model: model
                    }})
                }});

                console.log('[chat] Response status:', response.status);

                if (!response.ok) {{
                    const data = await response.json();
                    throw new Error(data.detail || 'Ошибка сервера');
                }}

                createAssistantMessage();

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                let chunkCount = 0;

                while (true) {{
                    const {{ done, value }} = await reader.read();
                    if (done) {{
                        console.log('[chat] Stream done, total chunks:', chunkCount);
                        break;
                    }}

                    buffer += decoder.decode(value, {{ stream: true }});

                    // Process complete SSE events
                    const lines = buffer.split('\\n');
                    buffer = lines.pop() || '';

                    for (const line of lines) {{
                        if (line.startsWith('data: ')) {{
                            try {{
                                const data = JSON.parse(line.slice(6));
                                console.log('[chat] Event:', data.type);

                                if (data.type === 'status') {{
                                    setStatus(data.status);
                                }} else if (data.type === 'sources') {{
                                    currentSources = data.sources;
                                    console.log('[chat] Sources:', currentSources.length);
                                }} else if (data.type === 'content') {{
                                    chunkCount++;
                                    appendToAssistantMessage(data.content);
                                }} else if (data.type === 'done') {{
                                    remainingEl.textContent = data.remaining;
                                    console.log('[chat] Done, remaining:', data.remaining);
                                }} else if (data.type === 'error') {{
                                    console.error('[chat] Error:', data.error);
                                    appendToAssistantMessage('\\n\\nОшибка: ' + data.error);
                                }}
                            }} catch (e) {{
                                console.warn('[chat] Parse error:', e, line);
                            }}
                        }}
                    }}
                }}

                finalizeAssistantMessage();

            }} catch (err) {{
                console.error('[chat] Fetch error:', err);
                if (currentContentDiv && !currentContentDiv.textContent) {{
                    currentMessageDiv.remove();
                    currentMessageDiv = null;
                    currentContentDiv = null;
                }}
                showError(err.message || 'Произошла ошибка');
            }} finally {{
                setLoading(false);
            }}
        }}

        questionInput.addEventListener('input', updateCharCount);

        questionInput.addEventListener('keydown', function(e) {{
            if (e.key === 'Enter' && !e.shiftKey) {{
                e.preventDefault();
                sendQuestion();
            }}
        }});

        sendBtn.addEventListener('click', sendQuestion);

        updateCharCount();
    </script>
</body>
</html>"""


# Global instances (initialized lazily or via preload)
_corpus: CorpusIndex | None = None
_chat_service: ChatService | None = None
_rate_limiter: RateLimiter | None = None
_data_path: Path | None = None


def preload_corpus(data_path: Path) -> None:
    """
    Preload corpus and embeddings at startup.

    Call this during app lifespan to avoid slow first request.
    """
    global _corpus, _data_path
    _data_path = data_path

    if _corpus is not None:
        return  # Already loaded

    logger.info("Preloading corpus...")
    settings = get_settings()
    loader = DocumentLoader(data_path)
    documents = loader.load_corpus_sync(strict=False)
    logger.info("Loaded %d documents", len(documents))

    # Create hybrid index if possible
    if settings.search_mode == "hybrid":
        logger.info("Creating hybrid index with embeddings...")
        from source.corpus.embeddings import SentenceTransformerProvider

        embedding_provider = SentenceTransformerProvider(name=settings.embedding_model)
        _corpus = CorpusIndex(
            documents=documents,
            embedding_provider=embedding_provider,
            mode="hybrid",
            alpha=settings.hybrid_alpha,
            cache_dir=settings.vector_cache_dir,
        )
    else:
        logger.info("Creating TF-IDF index...")
        _corpus = CorpusIndex(documents=documents, mode="tfidf")

    logger.info("Corpus preloaded successfully")


def get_corpus(data_path: Path) -> CorpusIndex:
    """Get or create corpus index."""
    global _corpus
    if _corpus is None:
        preload_corpus(data_path)
    return _corpus  # type: ignore[return-value]


def get_rate_limiter(daily_limit: int) -> RateLimiter:
    """Get or create rate limiter."""
    global _rate_limiter
    if _rate_limiter is None or _rate_limiter.daily_limit != daily_limit:
        _rate_limiter = RateLimiter(daily_limit=daily_limit)
    return _rate_limiter


def get_chat_service(
    data_path: Path,
    config: ChatConfig,
) -> ChatService:
    """Get or create chat service."""
    global _chat_service
    if _chat_service is None:
        corpus = get_corpus(data_path)
        rate_limiter = get_rate_limiter(config.daily_limit)
        _chat_service = ChatService(corpus, config, rate_limiter)
    return _chat_service


def render_chat_demo_page(org_name: str = "LLM инициатива") -> str:
    """Render demo chat page with donation call-to-action."""

    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ЛПРБаза — {html.escape(org_name)}</title>
    <style>
        :root {{
            --white: #FFFFFF;
            --off-white: #FAFAFA;
            --black: #000000;
            --gray: #666666;
            --light-gray: #E5E5E5;
            --red: #E53935;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
            background: var(--white);
            color: var(--black);
            line-height: 1.5;
            min-height: 100vh;
        }}

        /* Header */
        .header {{
            border-bottom: 1px solid var(--black);
            padding: 1.25rem 2rem;
            position: sticky;
            top: 0;
            background: var(--white);
            z-index: 100;
        }}

        .header-inner {{
            max-width: 900px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .header-left {{
            display: flex;
            align-items: center;
            gap: 2rem;
        }}

        .back-link {{
            color: var(--gray);
            text-decoration: none;
            font-size: 0.9rem;
            transition: color 0.15s;
        }}

        .back-link:hover {{
            color: var(--black);
        }}

        .logo {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }}

        .logo-square {{
            width: 16px;
            height: 16px;
            background: var(--red);
        }}

        .logo-text {{
            font-size: 1.25rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }}

        .status {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.8rem;
            color: var(--gray);
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}

        .status-dot {{
            width: 6px;
            height: 6px;
            background: var(--red);
        }}

        /* Main */
        .main {{
            max-width: 900px;
            margin: 0 auto;
            padding: 6rem 2rem 4rem;
            position: relative;
        }}

        /* Decorative square */
        .decor-square {{
            position: absolute;
            width: 120px;
            height: 120px;
            background: var(--red);
            top: 4rem;
            right: 2rem;
            z-index: -1;
        }}

        /* Hero */
        .hero {{
            margin-bottom: 6rem;
            max-width: 600px;
        }}

        .hero-title {{
            font-size: 3.5rem;
            font-weight: 700;
            line-height: 1.05;
            letter-spacing: -0.03em;
            margin-bottom: 1.5rem;
        }}

        .hero-subtitle {{
            font-size: 1.1rem;
            color: var(--gray);
            line-height: 1.6;
        }}

        /* Features list */
        .features {{
            margin-bottom: 5rem;
            border-top: 1px solid var(--black);
        }}

        .feature {{
            display: flex;
            align-items: baseline;
            gap: 1.5rem;
            padding: 1rem 0;
            border-bottom: 1px solid var(--light-gray);
        }}

        .feature-marker {{
            width: 8px;
            height: 8px;
            background: var(--black);
            flex-shrink: 0;
            margin-top: 0.5rem;
        }}

        .feature-marker.special {{
            background: var(--red);
        }}

        .feature-title {{
            font-weight: 600;
            min-width: 180px;
        }}

        .feature-desc {{
            color: var(--gray);
            font-size: 0.95rem;
        }}

        /* Chat demo section */
        .chat-demo {{
            margin-bottom: 5rem;
            border: 2px solid var(--black);
            background: var(--off-white);
        }}

        .chat-header {{
            padding: 1rem 1.5rem;
            border-bottom: 1px solid var(--light-gray);
            display: flex;
            align-items: center;
            gap: 0.75rem;
            background: var(--white);
        }}

        .chat-header-dot {{
            width: 8px;
            height: 8px;
            background: var(--red);
        }}

        .chat-header-title {{
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}

        .chat-messages {{
            padding: 2rem 1.5rem;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            min-height: 300px;
        }}

        .msg {{
            max-width: 85%;
            padding: 1rem 1.25rem;
            line-height: 1.6;
        }}

        .msg-user {{
            align-self: flex-end;
            background: var(--black);
            color: var(--white);
        }}

        .msg-assistant {{
            align-self: flex-start;
            background: var(--white);
            border: 1px solid var(--light-gray);
        }}

        .msg-assistant strong {{
            font-weight: 600;
        }}

        .msg-sources {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin-top: 1rem;
            padding-top: 0.75rem;
            border-top: 1px solid var(--light-gray);
        }}

        .msg-source {{
            font-size: 0.7rem;
            font-family: "SF Mono", "Monaco", monospace;
            padding: 0.2rem 0.4rem;
            background: var(--off-white);
            border: 1px solid var(--light-gray);
        }}

        .chat-input {{
            padding: 1rem 1.5rem;
            border-top: 1px solid var(--light-gray);
            background: var(--white);
            display: flex;
            gap: 1rem;
            align-items: center;
        }}

        .chat-input-field {{
            flex: 1;
            padding: 0.75rem 1rem;
            border: 1px solid var(--light-gray);
            font-family: inherit;
            font-size: 0.95rem;
            color: var(--gray);
            background: var(--off-white);
        }}

        .chat-input-field::placeholder {{
            color: var(--gray);
        }}

        .chat-input-btn {{
            padding: 0.75rem 1.5rem;
            background: var(--light-gray);
            border: none;
            font-family: inherit;
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--gray);
            cursor: not-allowed;
        }}

        .msg-source {{
            font-size: 0.7rem;
            font-family: "SF Mono", "Monaco", monospace;
            padding: 0.2rem 0.4rem;
            background: var(--off-white);
            border: 1px solid var(--light-gray);
            color: var(--gray);
        }}

        /* Poll section */
        .poll {{
            text-align: center;
            padding: 3rem 2rem;
            border-top: 1px solid var(--black);
        }}

        .poll-question {{
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
        }}

        .poll-buttons {{
            display: flex;
            gap: 1rem;
            justify-content: center;
            flex-wrap: wrap;
        }}

        .poll-btn {{
            padding: 0.75rem 2rem;
            font-family: inherit;
            font-size: 0.9rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s;
            border: 2px solid var(--black);
        }}

        .poll-btn-yes {{
            background: var(--black);
            color: var(--white);
        }}

        .poll-btn-yes:hover {{
            background: var(--white);
            color: var(--black);
        }}

        .poll-btn-no {{
            background: var(--white);
            color: var(--black);
        }}

        .poll-btn-no:hover {{
            background: var(--light-gray);
        }}

        .poll-step {{
            display: none;
        }}

        .poll-step.active {{
            display: block;
        }}

        .poll-thanks {{
            padding: 2rem 0;
        }}

        .poll-thanks-icon {{
            width: 16px;
            height: 16px;
            background: var(--red);
            margin: 0 auto 1rem;
        }}

        .poll-thanks-text {{
            color: var(--gray);
            font-size: 0.9rem;
        }}

        /* Responsive */
        @media (max-width: 700px) {{
            .hero-title {{
                font-size: 2.5rem;
            }}

            .decor-square {{
                width: 60px;
                height: 60px;
                top: 2rem;
                right: 1rem;
            }}

            .feature {{
                flex-direction: column;
                gap: 0.5rem;
            }}

            .feature-title {{
                min-width: auto;
            }}

            .main {{
                padding: 3rem 1.5rem 2rem;
            }}
        }}
    </style>
</head>
<body>
    <header class="header">
        <div class="header-inner">
            <div class="header-left">
                <a href="/" class="back-link">Назад</a>
                <div class="logo">
                    <div class="logo-square"></div>
                    <span class="logo-text">ЛПРБаза</span>
                </div>
            </div>
            <div class="status">
                <div class="status-dot"></div>
                <span>Готово</span>
            </div>
        </div>
    </header>

    <main class="main">
        <div class="decor-square"></div>

        <section class="hero">
            <h1 class="hero-title">Поиск по документам партии</h1>
            <p class="hero-subtitle">Устав, решения комитетов, прецеденты. Ответ с цитатами за секунды.</p>
        </section>

        <section class="features">
            <div class="feature">
                <div class="feature-marker"></div>
                <div class="feature-title">Документы</div>
                <div class="feature-desc">Устав, декларация, положения, регламенты</div>
            </div>
            <div class="feature">
                <div class="feature-marker"></div>
                <div class="feature-title">Решения</div>
                <div class="feature-desc">ЭК, ФК, апелляции, прецеденты</div>
            </div>
            <div class="feature">
                <div class="feature-marker special"></div>
                <div class="feature-title">Реестр</div>
                <div class="feature-desc">Члены, статусы, регионы</div>
            </div>
            <div class="feature">
                <div class="feature-marker"></div>
                <div class="feature-title">Поиск</div>
                <div class="feature-desc">Семантический + ключевые слова</div>
            </div>
        </section>

        <section class="chat-demo">
            <div class="chat-header">
                <div class="chat-header-dot"></div>
                <span class="chat-header-title">ЛПРБаза</span>
            </div>
            <div class="chat-messages">
                <div class="msg msg-user">Какие санкции предусмотрены за нарушение этического кодекса?</div>
                <div class="msg msg-assistant">
                    Согласно документам, за нарушения этического кодекса предусмотрены:<br><br>
                    <strong>Предупреждение</strong> — за первичные незначительные нарушения<br>
                    <strong>Приостановление членства</strong> — до 6 месяцев за грубые нарушения<br>
                    <strong>Исключение</strong> — за систематические или тяжкие нарушения<br><br>
                    Решения ЭК можно обжаловать в ФК в течение 30 дней.
                    <div class="msg-sources">
                        <span class="msg-source">УСТАВ, п. 5.3</span>
                        <span class="msg-source">ЭК-РЕГЛАМЕНТ, п. 2.1</span>
                        <span class="msg-source">КОДЕКС, п. 4.2</span>
                    </div>
                </div>
            </div>
            <div class="chat-input">
                <input type="text" class="chat-input-field" placeholder="Задайте вопрос..." disabled>
                <button class="chat-input-btn" disabled>Отправить</button>
            </div>
        </section>

        <section class="poll" id="poll">
            <div class="poll-step active" id="step1">
                <div class="poll-question">Вам нужен такой инструмент?</div>
                <div class="poll-buttons">
                    <button class="poll-btn poll-btn-yes" id="needYes">Да, нужен</button>
                    <button class="poll-btn poll-btn-no" id="needNo">Нет</button>
                </div>
            </div>

            <div class="poll-step" id="step2">
                <div class="poll-question">Готовы поддержать финансово?</div>
                <div class="poll-buttons">
                    <button class="poll-btn poll-btn-yes" id="payYes">Да</button>
                    <button class="poll-btn poll-btn-no" id="payNo">Пока нет</button>
                </div>
            </div>

            <div class="poll-step" id="thanks">
                <div class="poll-thanks">
                    <div class="poll-thanks-icon"></div>
                    <div class="poll-question">Спасибо!</div>
                    <p class="poll-thanks-text">Ваш голос учтён.</p>
                </div>
            </div>
        </section>
    </main>

    <script>
        const data = {{ need: null, pay: null }};

        function showStep(id) {{
            document.querySelectorAll('.poll-step').forEach(s => s.classList.remove('active'));
            document.getElementById(id).classList.add('active');
        }}

        function save() {{
            fetch('/api/poll', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify(data)
            }});
        }}

        document.getElementById('needYes').addEventListener('click', () => {{
            data.need = true;
            showStep('step2');
        }});

        document.getElementById('needNo').addEventListener('click', () => {{
            data.need = false;
            save();
            showStep('thanks');
        }});

        document.getElementById('payYes').addEventListener('click', () => {{
            data.pay = true;
            save();
            showStep('thanks');
        }});

        document.getElementById('payNo').addEventListener('click', () => {{
            data.pay = false;
            save();
            showStep('thanks');
        }});
    </script>
</body>
</html>"""


__all__ = [
    "ChatConfig",
    "ChatService",
    "ChatResponse",
    "RateLimiter",
    "get_chat_service",
    "get_corpus",
    "get_rate_limiter",
    "preload_corpus",
    "render_chat_page",
    "render_chat_demo_page",
]
