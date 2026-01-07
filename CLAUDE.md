# CLAUDE.md

Инструкции для AI-ассистентов по работе с проектом AFK (AI Committee Member).

## Что это

Агентская система на базе LLM для принятия юридически обоснованных решений в политических организациях. Использует библиотеку `funcai` для оркестрации агентов.

**Основной flow:**

1. Загрузка корпуса документов (Устав, кодексы, прецеденты) → индексация
2. Получение запроса (Request) с фактами и вопросом
3. Агент ищет нормы, прецеденты, строит цепочку рассуждений
4. Выдаёт структурированное решение (Decision) с цитатами
5. Сохраняет audit package для воспроизводимости

**Режимы поиска:**

- `tfidf` — быстрый keyword-based поиск (default, без доп. зависимостей)
- `vector` — семантический поиск через embeddings
- `hybrid` — комбинация vector + BM25 (рекомендуется для лучшего качества)

## Структура проекта

```
source/
├── api/           # Python API (CommitteeMember)
├── cli/           # Click CLI (main.py)
├── committee/     # Ядро: agent.py, prompts.py, commands.py
├── corpus/        # Загрузка документов и индексация
│   ├── embeddings/  # Vector search подсистема
│   │   ├── providers.py   # ABCEmbeddingProvider, SentenceTransformer, OpenAI
│   │   ├── chunker.py     # Разбиение документов на чанки
│   │   ├── vector_index.py # ChromaDB vector index
│   │   └── hybrid.py      # Hybrid search (vector + BM25)
│   ├── index.py     # Unified CorpusIndex (фасад)
│   └── tfidf_index.py # Legacy TF-IDF индекс
├── contracts/     # Pydantic модели: Request, Decision, Audit
├── tools/         # Инструменты агента (corpus, precedents, attachments)
├── providers/     # Абстракция LLM (Anthropic/OpenAI)
├── audit/         # Сохранение и верификация решений
├── export/        # HTML экспорт решений
├── subagents/     # Субагенты (research)
└── web/           # FastAPI сервер

data/              # Корпус документов (markdown с YAML frontmatter)
audit_logs/        # JSON пакеты решений
public/            # Статический HTML сайт с решениями
requests/          # JSON файлы запросов
```

## Ключевые файлы

| Файл                           | Назначение                                   |
| ------------------------------ | -------------------------------------------- |
| `source/committee/prompts.py`  | Системные промпты агента                     |
| `source/committee/agent.py`    | CommitteeAgent, dialogue, tools              |
| `source/contracts/decision.py` | Структура решения                            |
| `source/contracts/request.py`  | Структура запроса                            |
| `source/tools/*.py`            | Инструменты: corpus, precedents, attachments |
| `source/export/html.py`        | HTML экспорт (CSS inline)                    |
| `source/corpus/index.py`       | Unified index (TF-IDF / Vector / Hybrid)     |
| `source/corpus/embeddings/`    | Vector search подсистема                     |

## Команды

```bash
# Установка (базовая, только TF-IDF)
uv sync

# Установка с vector search (рекомендуется)
uv sync --extra vector

# Принятие решения (TF-IDF, default)
python main.py decide requests/appeal-generalov.json --verbose

# Принятие решения с semantic search (hybrid mode)
python main.py --search-mode hybrid decide requests/appeal-generalov.json --verbose

# Выбор embedding модели
python main.py --search-mode hybrid --embedding-model "intfloat/multilingual-e5-large" decide ...

# Экспорт в HTML
python main.py export --all

# Запуск веб-сервера
python main.py serve -p 8080

# Статистика корпуса
python main.py corpus-stats

# Дедупликация
python main.py dedup --verbose --delete
```

## Режимы поиска

| Режим    | Описание                          | Зависимости               |
| -------- | --------------------------------- | ------------------------- |
| `tfidf`  | Keyword search, быстрый           | Нет (встроено)            |
| `vector` | Semantic search через embeddings  | sentence-transformers, chromadb |
| `hybrid` | Vector + BM25 (лучшее качество)   | sentence-transformers, chromadb |

**Настройка через env:**

```bash
SEARCH_MODE=hybrid
EMBEDDING_MODEL=intfloat/multilingual-e5-base
HYBRID_ALPHA=0.7  # Вес vector vs BM25 (0-1)
```

**Рекомендуемые embedding модели:**

- `intfloat/multilingual-e5-base` — хороший баланс (default)
- `intfloat/multilingual-e5-large` — лучшее качество
- `BAAI/bge-m3` — поддержка длинных текстов
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` — быстрая

## Стиль кода

**Python:**

- Импорты только в начале файла, не внутри функций
- Типизация везде: `str | None`, `list[str]`, `dict[str, Any]`, `Annotated`
- Без `getattr/setattr/hasattr` и метамагии
- Pydantic для всех контрактов данных

**Frontend (public/):**

- Модульная структура
- Разметка через template/компоненты
- Без inline-обработчиков, только addEventListener
- ES-модули, async/await

## Промпты агента

Промпты в `source/committee/prompts.py`:

- `COMMITTEE_SYSTEM_PROMPT` — основной системный промпт
- `DECIDE_PROMPT` — инструкции для принятия решения
- Требования к агенту:
  - Каждый юридический тезис → норма или "норма не найдена"
  - Каждый факт → evidence_refs (источник)
  - Прецеденты → детальное сравнение по 6 критериям
  - Резолютивка → проверка полномочий органа

## Инструменты агента

```python
# source/tools/
corpus_tools.py      # search_corpus, get_document, search_document_section
precedent_tools.py   # find_precedents, compare_with_precedent
attachment_tools.py  # read_attachment
reasoning_tools.py   # check_norm_conflicts, apply_priority_rules
validation_tools.py  # validate_request, check_decision_readiness
research_tools.py    # research_documents (субагент)
```

**Поиск:**

- В режиме `tfidf`: keyword-based, запросы 3-6 ключевых слов, на русском
- В режиме `hybrid`/`vector`: семантический поиск, можно писать естественным языком

## Контракты

**Request** (вход):

- query, case_type, parties, attachments, additional_context
- additional_context → факты, которые агент принимает как установленные

**Decision** (выход):

- verdict: decision | sanction | clarification | refusal | needs_more_info
- findings_of_fact (каждый с evidence_refs и confidence)
- applicable_norms
- reasoning (fact → norm → conclusion)
- citations, uncertainty, minority_view

## HTML экспорт

`source/export/html.py`:

- CSS inline (переменные в :root)
- Confidence badges — компактные, в углу факта
- Footer — tech-widget справа (model, agent, id)
- Поддержка нескольких моделей на одно дело (tabs)

## Git

**Только read-only операции:**

- `git status`, `git log`, `git show`, `git diff`, `git ls-files`
- Изменения предлагать текстовым diff/patch
- Без commit/push/merge/rebase

## Миграции

Можно создавать миграции, но не применять их автоматически.

## Частые задачи

**Добавить новый инструмент агенту:**

1. Создать функцию с `@tool` декоратором в `source/tools/`
2. Добавить в `make_*_tools()` → list[Tool]
3. Подключить в `source/committee/agent.py` → `build_tools()`

**Изменить поведение агента:**

1. Редактировать промпты в `source/committee/prompts.py`
2. Для структурных изменений → контракты в `source/contracts/`

**Добавить документ в корпус:**

1. Создать .md файл в `data/` с YAML frontmatter
2. Запустить `python main.py corpus-stats` для проверки

**Обновить HTML экспорт:**

1. CSS в `CSS_STYLES` строке
2. Методы `render_*()` в классе `HTMLExporter`
