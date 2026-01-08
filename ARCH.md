# AI Committee Member — Архитектура

## Структура проекта

```
AFK_ai_fk/
├── source/                 # Основной Python пакет
│   ├── api/               # Python API (CommitteeMember класс)
│   ├── cli/               # Командная строка (Click CLI)
│   ├── committee/         # Ядро принятия решений (AI агент)
│   ├── corpus/            # Загрузка и индексация документов
│   ├── contracts/         # Pydantic модели данных
│   ├── tools/             # Инструменты для AI агента
│   ├── providers/         # Абстракция LLM (Anthropic/OpenAI)
│   ├── subagents/         # Суб-агенты (research)
│   ├── audit/             # Аудит и верификация решений
│   ├── export/            # HTML экспорт решений
│   └── reasoning/         # Утилиты для резолюции конфликтов норм
│
├── funcai/                # Мини-фреймворк для агентов
│   ├── core/              # Dialogue, Message, Provider
│   ├── agents/            # Tool, ABCAgent, ReActAgent
│   └── std/               # OpenAI/Anthropic провайдеры
│
├── data/                  # Корпус документов (с YAML frontmatter)
├── audit_logs/            # JSON пакеты с историей решений
├── public/                # HTML экспорт для веб-просмотра
└── main.py               # CLI точка входа
```

---

## Обзор потока данных

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│   Request   │────▶│ CommitteeAgent│────▶│  LLM Provider   │────▶│   Decision   │
│   (JSON)    │     │  (ReAct loop) │     │ (Anthropic/OAI) │     │  (Pydantic)  │
└─────────────┘     └──────────────┘     └─────────────────┘     └──────────────┘
                           │                                            │
                           ▼                                            ▼
                    ┌─────────────┐                              ┌─────────────┐
                    │   Tools     │                              │ AuditPackage│
                    │ (12 штук)   │                              │   (JSON)    │
                    └─────────────┘                              └─────────────┘
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
     ┌───────────┐  ┌───────────┐  ┌───────────┐
     │ CorpusIndex│  │ SubAgent  │  │ Precedents│
     │  (Search)  │  │ (Research)│  │  (Search) │
     └───────────┘  └───────────┘  └───────────┘
```

---

## 1. Агентская система (funcai)

### 1.1 ReAct Agent — Ядро

**Файл**: `funcai/std/react_agent.py`

ReAct (Reason + Act) — основной паттерн агента:

```
┌────────────────────────────────────────────────────────────┐
│                     ReAct Loop                             │
│                                                            │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│  │  Send    │───▶│ LLM resp │───▶│ Has tool │───Yes───┐   │
│  │ messages │    │          │    │  calls?  │         │   │
│  └──────────┘    └──────────┘    └──────────┘         │   │
│       ▲                               │               │   │
│       │                              No               ▼   │
│       │                               │        ┌──────────┐│
│       │                               ▼        │ Execute  ││
│       │                        ┌──────────┐    │  tools   ││
│       │                        │  Return  │    └──────────┘│
│       │                        │ response │         │      │
│       │                        └──────────┘         │      │
│       └─────────────────────────────────────────────┘      │
│                                                            │
│  max_steps = 20 (ограничение итераций)                    │
└────────────────────────────────────────────────────────────┘
```

**Состояние шага** (`StepState`):

- `messages: tuple[Message, ...]` — накопленная история диалога
- `tool_calls_made: tuple[ToolCall, ...]` — все вызовы инструментов
- `step: int` — номер текущего шага

**Выход агента** (`AgentResponse[S]`):

- `message: Message` — финальный ответ LLM
- `parsed: S` — распарсенный результат (Decision)
- `tool_calls_made: list[ToolCall]` — полная история вызовов
- `steps: int` — количество шагов

### 1.2 Tool — Инструмент агента

**Файл**: `funcai/agents/tool.py`

```python
@dataclass(frozen=True)
class Tool:
    name: str                    # search_corpus
    description: str             # "Search the normative corpus..."
    parameters: type[BaseModel]  # Pydantic-схема аргументов
    fn: Callable[..., Any]       # Функция-реализация
    return_type: Option[type]    # Тип возврата
```

Декоратор `@tool`:

```python
@tool("Search the normative corpus for documents matching the query")
def search_corpus(query: str, doc_types: str | None = None) -> list[SearchResult]:
    ...
```

### 1.3 Provider — Абстракция LLM

**Файлы**: `funcai/core/provider.py`, `source/providers/`

```
┌─────────────────────────────────────────────────────────┐
│                  ABCAIProvider[E]                       │
│                                                         │
│  send_messages(                                         │
│      messages: list[Message],                           │
│      tools: list[Tool] = [],                           │
│      schema: Option[type[S]] = Nothing(),              │
│  ) -> Result[AIResponse[S], E]                         │
└─────────────────────────────────────────────────────────┘
                         ▲
           ┌─────────────┼─────────────┐
           │             │             │
   ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
   │ AnthropicProv │ │  OpenAIProv   │ │LocalOpenAIProv│
   │   (Claude)    │ │   (GPT-5)     │ │  (LMStudio)   │
   └───────────────┘ └───────────────┘ └───────────────┘
```

**Поддерживаемые провайдеры**:

- `anthropic` — Claude (sonnet/opus)
- `openai` — GPT-4o и другие
- `openrouter` — OpenAI-совместимый API
- `lmstudio` — локальные модели

---

## 2. CommitteeAgent — Доменный агент

**Файл**: `source/committee/agent.py`

CommitteeAgent расширяет ReActAgent доменной логикой:

```
┌─────────────────────────────────────────────────────────────┐
│                    CommitteeAgent                           │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │                 System Prompt                          │ │
│  │  - Principles (impartiality, legal reasoning, etc.)   │ │
│  │  - Jurisdiction rules (lex superior/specialis/poster) │ │
│  │  - Decision process (facts → norms → reasoning)       │ │
│  │  - Search limits (6 search calls max)                 │ │
│  │  - Citation format [KEY, Section] "quoted"            │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    Tools (12)                        │   │
│  │                                                      │   │
│  │  Corpus:     search_corpus, get_document,           │   │
│  │              search_document_section, cite_fragment │   │
│  │                                                      │   │
│  │  Precedents: find_precedents, compare_with_precedent│   │
│  │                                                      │   │
│  │  Reasoning:  check_norm_conflicts,                  │   │
│  │              apply_priority_rules,                  │   │
│  │              verify_reasoning_chain                 │   │
│  │                                                      │   │
│  │  Research:   research_documents (sub-agent)         │   │
│  │                                                      │   │
│  │  Validation: validate_request, check_decision_ready │   │
│  │                                                      │   │
│  │  Attachments: read_attachment                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Output: Decision (structured Pydantic model)              │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Система поиска (CorpusIndex)

**Файлы**: `source/corpus/index.py`, `source/corpus/embeddings/`

### 3.1 Режимы поиска

```
┌─────────────────────────────────────────────────────────────┐
│                      CorpusIndex                            │
│                                                             │
│  mode: "tfidf" | "vector" | "hybrid"                       │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   TF-IDF    │  │   Vector    │  │       Hybrid        │ │
│  │             │  │             │  │                     │ │
│  │ • Быстрый   │  │ • Семантика │  │ • Vector + BM25     │ │
│  │ • Ключевые  │  │ • Embeddings│  │ • alpha=0.7         │ │
│  │   слова     │  │ • GPU/CPU   │  │ • Лучшее качество   │ │
│  │ • Без deps  │  │ • ~500ms    │  │ • RRF fusion        │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│                                                             │
│  Embedding models:                                          │
│  • intfloat/multilingual-e5-base (default)                 │
│  • OpenAI text-embedding-3-small                           │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Chunking (разбиение документов)

**Файл**: `source/corpus/embeddings/chunker.py`

```
┌─────────────────────────────────────────────────────────────┐
│                    DocumentChunker                          │
│                                                             │
│  Документ (10K символов)                                   │
│  ─────────────────────────                                 │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐          │
│  │ Chunk 1 │ │ Chunk 2 │ │ Chunk 3 │ │ Chunk 4 │          │
│  │ ~1500   │ │ ~1500   │ │ ~1500   │ │ ~1500   │          │
│  │ chars   │ │ chars   │ │ chars   │ │ chars   │          │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘          │
│       │           │           │           │                │
│       ▼           ▼           ▼           ▼                │
│  [embedding] [embedding] [embedding] [embedding]           │
│                                                             │
│  Параметры:                                                │
│  • chunk_size: 1500 символов                               │
│  • chunk_overlap: 200 символов                             │
│  • section_aware: True (по заголовкам ##)                  │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Hybrid Search

**Файл**: `source/corpus/embeddings/hybrid.py`

```
Query: "правила исключения за грубое нарушение"
       │
       ├──────────────────┬──────────────────┐
       ▼                  ▼                  │
┌─────────────┐    ┌─────────────┐          │
│   Vector    │    │    BM25     │          │
│   Search    │    │   Search    │          │
│             │    │             │          │
│ Semantic    │    │ Keyword     │          │
│ similarity  │    │ matching    │          │
└─────────────┘    └─────────────┘          │
       │                  │                  │
       ▼                  ▼                  │
   [0.85, 0.72,      [0.91, 0.45,           │
    0.68, ...]        0.38, ...]            │
       │                  │                  │
       └────────┬─────────┘                  │
                ▼                            │
       ┌─────────────────┐                   │
       │  Score Fusion   │◀──────────────────┘
       │                 │   alpha=0.7
       │ Weighted:       │
       │ s = α·vec +     │
       │     (1-α)·bm25  │
       │                 │
       │ или RRF:        │
       │ s = Σ 1/(k+rank)│
       └─────────────────┘
                │
                ▼
       Ranked Results
```

---

## 4. Инструменты агента (Tools)

### 4.1 Corpus Tools

**Файл**: `source/tools/corpus_tools.py`

| Tool                      | Описание                                           | Token Budget |
| ------------------------- | -------------------------------------------------- | ------------ |
| `search_corpus`           | Семантический поиск, возвращает chunks ~1500 chars | Нет          |
| `get_document`            | Полный текст документа                             | ✓ Потребляет |
| `search_document_section` | Поиск конкретного раздела (п. 2.3.2.2)             | Нет          |
| `cite_fragment`           | Создание формальной цитаты                         | Нет          |

### 4.2 Precedent Tools

**Файл**: `source/tools/precedent_tools.py`

| Tool                     | Описание                                    |
| ------------------------ | ------------------------------------------- |
| `find_precedents`        | Семантический поиск похожих дел (по фактам) |
| `compare_with_precedent` | Сравнение текущего дела с прецедентом       |

**Важно**: факты должны быть на русском для лучшего matching!

### 4.3 Reasoning Tools

**Файл**: `source/tools/reasoning_tools.py`

| Tool                     | Описание                                    |
| ------------------------ | ------------------------------------------- |
| `check_norm_conflicts`   | Проверка конфликтов между нормами           |
| `apply_priority_rules`   | Применение lex superior/specialis/posterior |
| `verify_reasoning_chain` | Валидация цепочки рассуждений               |

### 4.4 Research Sub-Agent

**Файл**: `source/subagents/research.py`

```
┌─────────────────────────────────────────────────────────────┐
│                  ResearchSubAgent                           │
│                                                             │
│  Запускается из main агента через research_documents()     │
│                                                             │
│  Input:  query + doc_ids (max 3)                           │
│  Output: ResearchResult                                     │
│          - summaries[]                                      │
│          - synthesis                                        │
│          - recommended_citations[]                          │
│          - gaps[]                                           │
│                                                             │
│  Преимущества:                                              │
│  • Изолированный контекст (не засоряет main agent)         │
│  • Полные документы читаются внутри sub-agent              │
│  • Возвращает только summary → экономия токенов            │
│                                                             │
│  Tools (внутренние):                                        │
│  • get_full_document (max 50K chars, truncates)            │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Контракты данных (Contracts)

### 5.1 Request — Входной запрос

**Файл**: `source/contracts/request.py`

```python
class Request(BaseModel):
    query: str                    # Вопрос/жалоба
    case_type: str                # ethics, discipline, arbitration
    requested_remedy: str         # Что просит заявитель
    parties: list[Party]          # Стороны дела
    attachments: list[Attachment] # Приложенные файлы
    jurisdiction: list[str]       # Приоритет норм
    additional_context: str       # Установленные факты
```

### 5.2 Decision — Выходное решение

**Файл**: `source/contracts/decision.py`

```python
class Decision(BaseModel):
    verdict: Literal["decision", "sanction", "clarification",
                     "refusal", "needs_more_info"]
    verdict_summary: str

    findings_of_fact: list[FindingOfFact]  # Факты + evidence_refs
    applicable_norms: list[ApplicableNorm] # Нормы + priority
    reasoning: list[ReasoningStep]         # Fact → Norm → Conclusion
    citations: list[Citation]              # [KEY, Section] "quote"

    uncertainty: str              # Что осталось неясным
    minority_view: str | None     # Альтернативная интерпретация
```

### 5.3 CorpusDocument — Документ корпуса

**Файл**: `source/contracts/corpus.py`

```python
class CorpusDocument(BaseModel):
    doc_id: str           # MAIN-DOCS-CHARTER-1
    citation_key: str     # UST-B1CAE0
    doc_type: str         # charter, code, decision, precedent
    priority: int         # 100=charter, 80=regulations, 50=decision

    metadata: DocumentMetadata
    content: str
    content_hash: str
    file_path: str
```

---

## 6. Аудит и воспроизводимость

**Файлы**: `source/audit/package.py`, `source/audit/storage.py`

### 6.1 DecisionPackage

```
┌─────────────────────────────────────────────────────────────┐
│                    DecisionPackage                          │
│                                                             │
│  package_id: UUID                                           │
│  timestamp: datetime (UTC)                                  │
│                                                             │
│  request_json: str        # Исходный запрос                │
│  decision_json: str       # Финальное решение              │
│                                                             │
│  corpus_snapshot: []      # Хеши использованных документов │
│  cited_fragments: []      # Все цитаты                     │
│  tool_calls: ToolCallHistory  # Вся история вызовов        │
│                                                             │
│  model_id: str            # claude-sonnet-4-20250514       │
│  provider: str            # anthropic                      │
│  temperature: float       # 0.0 (deterministic)            │
│  mode: str                # deterministic/creative         │
│                                                             │
│  content_hash: SHA256     # Для верификации целостности    │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Верификация

```bash
# Проверить целостность пакета
python main.py verify-audit <package_id>
```

---

## 7. CLI команды

```bash
# Основные команды
python main.py decide request.json          # Принять решение
python main.py intake request.json          # Проверить полноту
python main.py compare request.json         # Сравнить с прецедентами
python main.py redteam request.json dec.json # Adversarial анализ
python main.py chat                         # Интерактивный режим

# Корпус
python main.py corpus-stats                 # Статистика
python main.py annotate                     # LLM-аннотация метаданных
python main.py dedup --delete               # Удалить дубликаты

# Экспорт
python main.py export --all                 # HTML для всех решений
python main.py serve                        # Веб-сервер

# Опции поиска
python main.py -s hybrid decide ...         # Гибридный поиск
python main.py -s vector decide ...         # Только embeddings
python main.py --embedding-model "..." ...  # Кастомная модель
```

---

## 9. Конфигурация

### 9.1 Переменные окружения

```bash
# API ключи
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
OPENROUTER_API_KEY=...

# Настройки поиска
SEARCH_MODE=hybrid              # tfidf, vector, hybrid
EMBEDDING_MODEL=intfloat/multilingual-e5-base

# Веб-сервер (ОБЯЗАТЕЛЬНО задать перед запуском!)
AUTH_USERNAME=your-username-here
AUTH_PASSWORD=your-strong-password-here-min-16-chars

# ========== ЭКОНОМИЯ БЮДЖЕТА ==========

# Sub-agent модель (дешёвая для research операций)
SUBAGENT_PROVIDER=openai        # openai, anthropic
SUBAGENT_MODEL=gpt-4o-mini      # gpt-4o-mini / claude-3-5-haiku-20241022

# Лимиты поиска
MAX_SEARCH_CALLS=10             # Максимум поисков (было 6)

# Re-ranking (улучшает качество, локально)
USE_RERANKER=false              # true для включения
RERANKER_MODEL=BAAI/bge-reranker-base
```

### 9.2 settings.py

```python
class Settings(BaseSettings):
    search_mode: SearchMode = "hybrid"
    embedding_model: str = "intfloat/multilingual-e5-base"
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    # Cost optimization settings
    subagent_provider: ProviderType | None = None   # Cheap model provider
    subagent_model: str | None = None               # e.g., "gpt-4o-mini"
    max_search_calls: int = 10                      # Search limit
    use_reranker: bool = False                      # Enable cross-encoder
    reranker_model: str = "BAAI/bge-reranker-base"
```

---

## 10. Типичный flow решения

```
1. Load Request (JSON)
   ├── parties, attachments, query, case_type

2. Create Dialogue
   ├── System prompt (principles, tools, limits)
   └── User message (formatted request)

3. ReAct Loop (max 20 steps)
   │
   ├── Step 1: read_attachment(evidence.md)
   ├── Step 2: search_corpus("правила исключения", doc_types="charter")
   ├── Step 3: search_document_section("CHARTER-1", "2.3.2.2")
   ├── Step 4: find_precedents("Член опубликовал оскорбления...", "ethics")
   ├── Step 5: research_documents("санкции за оскорбления", "DEC-10,DEC-15")
   │           └── SubAgent: get_full_document × 2 → ResearchResult
   ├── Step 6: check_norm_conflicts("CHARTER-1,CODE-1")
   └── Step 7: [No tools] → Final answer

4. Parse Decision (Pydantic)
   ├── verdict, findings_of_fact, applicable_norms
   ├── reasoning (fact → norm → conclusion)
   └── citations, uncertainty, minority_view

5. Build AuditPackage
   ├── corpus_snapshot (hashes)
   ├── tool_calls history
   └── content_hash (SHA256)

6. Save & Export
   ├── audit_logs/{uuid}.json
   └── public/{case}/index.html
```
