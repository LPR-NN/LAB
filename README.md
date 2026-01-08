# AI Committee Member

AI-система для принятия обоснованных решений в политических организациях на базе funcai.

## Возможности

- **Беспристрастные решения** — AI анализирует факты и нормы без предвзятости
- **Юридическое обоснование** — каждое решение содержит цепочку рассуждений со ссылками
- **Прецедентный анализ** — сравнение с предыдущими решениями
- **Аудит решений** — полная воспроизводимость с сохранением контекста
- **Разрешение конфликтов норм** — lex superior, lex specialis, lex posterior

## Установка

```bash
# Базовая (TF-IDF поиск)
uv sync

# С семантическим поиском (рекомендуется)
uv sync --extra vector
```

## Быстрый старт

### Python API

```python
import asyncio
from source.api.committee import CommitteeMember
from source.contracts.request import Request, Party

async def main():
    # Инициализация
    committee = CommitteeMember(
        corpus_path="data/",
        provider="anthropic",  # или "openai"
        model="claude-sonnet-4-20250514",
    )

    # Создание запроса
    request = Request(
        query="Следует ли применить взыскание за неуплату взносов?",
        case_type="discipline",
        requested_remedy="decision",
        parties=[
            Party(name="ФК", role="complainant"),
            Party(name="Член X", role="respondent"),
        ],
    )

    # Принятие решения
    decision, audit_id = await committee.decide(request)

    print(f"Verdict: {decision.verdict}")
    print(f"Summary: {decision.verdict_summary}")
    print(f"Audit ID: {audit_id}")

asyncio.run(main())
```

### CLI

```bash
# Справка
python main.py --help

# Проверка запроса
python main.py intake examples/request-example.json

# Принятие решения
python main.py decide examples/request-example.json

# Интерактивный режим
python main.py chat

# Статистика корпуса
python main.py corpus-stats
```

## Структура корпуса

Документы в `data/` в формате Markdown с опциональным YAML frontmatter:

```markdown
---
doc_id: CHARTER-001
title: Устав Партии
doc_type: charter
effective_date: 2024-01-01
citation_key: UST-1
priority: 100
status: active
---

# Устав Партии

## Статья 1. Общие положения

...
```

Если frontmatter отсутствует, система автоматически извлечёт метаданные.

### Типы документов (по приоритету)

| Тип           | Приоритет | Описание       |
| ------------- | --------- | -------------- |
| charter       | 100       | Устав          |
| regulations   | 80        | Регламент      |
| code          | 70        | Кодекс         |
| decision      | 50        | Решение органа |
| precedent     | 40        | Прецедент      |
| clarification | 30        | Разъяснение    |

## Команды

| Команда    | Описание                     |
| ---------- | ---------------------------- |
| `/intake`  | Проверка полноты данных      |
| `/decide`  | Принятие решения             |
| `/cite`    | Показать цитаты              |
| `/appeal`  | Черновик апелляции           |
| `/compare` | Сравнение с прецедентами     |
| `/redteam` | Самопроверка на предвзятость |

## Формат запроса

```json
{
  "query": "Что нужно решить",
  "case_type": "discipline",
  "jurisdiction": ["charter", "regulations", "decisions"],
  "parties": [
    { "name": "Сторона A", "role": "complainant" },
    { "name": "Сторона B", "role": "respondent" }
  ],
  "requested_remedy": "decision",
  "attachments": [
    {
      "file_path": "evidence/doc1.md",
      "relevance_note": "Подтверждает факт X"
    }
  ]
}
```

## Формат решения

```json
{
  "verdict": "decision",
  "verdict_summary": "Краткое резюме",
  "findings_of_fact": [...],
  "applicable_norms": [...],
  "reasoning": [
    {
      "fact": "Установленный факт",
      "norm_or_precedent": "[UST-1, ст. 5]",
      "conclusion": "Вывод"
    }
  ],
  "citations": [...],
  "uncertainty": "Что остаётся неясным"
}
```

## Конфигурация

Переменные окружения:

```bash
# API ключи
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Режим поиска (опционально)
SEARCH_MODE=hybrid                          # tfidf | vector | hybrid
EMBEDDING_MODEL=intfloat/multilingual-e5-base
HYBRID_ALPHA=0.7                            # 0-1, выше = больше семантики
```

## Лицензия

MIT

---

## Шпаргалка по командам

### Глобальные флаги (перед командой)

```bash
-c, --corpus PATH        # Путь к корпусу (default: data/)
-p, --provider NAME      # openai | anthropic | openrouter | lmstudio
-m, --model NAME         # Модель LLM
-s, --search-mode MODE   # tfidf | vector | hybrid
--embedding-model NAME   # Модель эмбеддингов (для vector/hybrid)
--alpha FLOAT            # Вес семантики в hybrid (0-1)
```

### Установка

```bash
# Только TF-IDF (быстро, без ML зависимостей)
uv sync

# С семантическим поиском
uv sync --extra vector
```

---

### 0. Подготовка корпуса

```bash
# Дедупликация (найти и удалить дубликаты)
python main.py dedup --verbose           # Только показать
python main.py dedup --verbose --delete  # Удалить

# Найти битые файлы (mojibake)
python main.py find-corrupted

# Автоматическое исправление кодировки
python main.py fix-encoding              # Только показать
python main.py fix-encoding --dry-run=false  # Применить
```

### 1. Аннотация метаданных

```bash
# Простой режим (один вызов на файл) — для сильных моделей
python main.py -p openai -m gpt-5-nano-2025-08-07 annotate

# Agent режим (5 шагов на файл) — для слабых/локальных моделей
python main.py -p lmstudio -m "qwen3-30b-a3b" annotate --agent --debug

# Принудительно переаннотировать всё
python main.py -p openai -m gpt-5-mini-2025-08-07 annotate --force

# Параметры
--force          # Переаннотировать даже уже размеченные
--dry-run        # Только показать что будет сделано
--batch-size N   # Параллельность (default: 5, для agent лучше 1-2)
--agent          # Многошаговый режим
--debug          # Детальный вывод
```

### 2. Статистика корпуса

```bash
# TF-IDF режим (default)
python main.py corpus-stats

# С семантическим индексом
python main.py -s hybrid corpus-stats
python main.py -s vector --embedding-model intfloat/multilingual-e5-large corpus-stats
```

---

### 3. Проверка запроса (intake)

```bash
# Базовый
python main.py -p openai -m gpt-5-nano-2025-08-07 intake requests/appeal-generalov.json

# С hybrid search
python main.py -p openai -m gpt-5-nano-2025-08-07 -s hybrid intake requests/appeal-generalov.json
```

### 4. Принятие решения (decide)

```bash
# TF-IDF поиск (default, быстро)
python main.py -p openai -m gpt-5-nano-2025-08-07 decide requests/appeal-generalov.json

# Verbose (показывает tool calls)
python main.py -p openai -m gpt-5-nano-2025-08-07 decide requests/appeal-generalov.json --verbose

# Debug (полный вывод tool calls)
python main.py -p openai -m gpt-5-nano-2025-08-07 decide requests/appeal-generalov.json --verbose --debug

# С семантическим поиском (hybrid)
python main.py -p openai -m gpt-5-mini-2025-08-07 -s hybrid decide requests/appeal-generalov.json --verbose

# Кастомная embedding модель
python main.py -p anthropic -m claude-sonnet-4-20250514 \
  -s hybrid --embedding-model intfloat/multilingual-e5-large --alpha 0.8 \
  decide requests/appeal-generalov.json --verbose

# Сохранить решение в файл
python main.py -p openai -m gpt-5-mini-2025-08-07 decide requests/appeal-generalov.json -o decision.json

# Без аудита
python main.py -p openai -m gpt-5-nano-2025-08-07 decide requests/appeal-generalov.json --no-audit
```

### 5. Сравнение с прецедентами

```bash
# Автопоиск прецедентов
python main.py -p openai -m gpt-5-nano-2025-08-07 compare requests/appeal-generalov.json

# Указать конкретные прецеденты
python main.py -p openai -m gpt-5-nano-2025-08-07 compare requests/appeal-generalov.json \
  -p DEC-123 -p DEC-456
```

### 6. Red team (самопроверка)

```bash
python main.py -p openai -m gpt-5-nano-2025-08-07 redteam requests/appeal-generalov.json decision.json
```

---

### 7. Cервер

```bash
# Запуск веб-сервера
python main.py serve -p 8080
python main.py serve -p 8080 --reload  # Dev mode с auto-reload
```

### 8. Верификация аудита

```bash
python main.py verify-audit abc123-def456-...
```

---

## Режимы поиска

| Режим    | Флаг        | Зависимости                     | Когда использовать                          |
| -------- | ----------- | ------------------------------- | ------------------------------------------- |
| `tfidf`  | `-s tfidf`  | Нет                             | Быстрые эксперименты, точные ключевые слова |
| `vector` | `-s vector` | sentence-transformers, chromadb | Семантический поиск                         |
| `hybrid` | `-s hybrid` | sentence-transformers, chromadb | **Лучшее качество** (семантика + keywords)  |

## Embedding модели

| Модель                                                        | Размер | Качество | Скорость |
| ------------------------------------------------------------- | ------ | -------- | -------- |
| `intfloat/multilingual-e5-base`                               | 560MB  | ⭐⭐⭐   | ⭐⭐⭐   |
| `intfloat/multilingual-e5-large`                              | 1.1GB  | ⭐⭐⭐⭐ | ⭐⭐     |
| `BAAI/bge-m3`                                                 | 2.2GB  | ⭐⭐⭐⭐ | ⭐       |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 470MB  | ⭐⭐     | ⭐⭐⭐⭐ |

---

## Примеры экспериментов

```bash
# Сравнить TF-IDF vs Hybrid на одном запросе
python main.py -p openai -m gpt-5-nano-2025-08-07 -s tfidf decide requests/appeal-generalov.json -o result-tfidf.json
python main.py -p openai -m gpt-5-nano-2025-08-07 -s hybrid decide requests/appeal-generalov.json -o result-hybrid.json

# Разные модели LLM
python main.py -p openai -m gpt-5-nano-2025-08-07 decide requests/appeal-generalov.json -o result-nano.json
python main.py -p openai -m gpt-5-mini-2025-08-07 decide requests/appeal-generalov.json -o result-mini.json
python main.py -p anthropic -m claude-sonnet-4-20250514 decide requests/appeal-generalov.json -o result-sonnet.json
```


python main.py \
  -p openai \
  -m gpt-5-nano-2025-08-07 \
  decide requests/appeal-generalov.json \
  -v -d
