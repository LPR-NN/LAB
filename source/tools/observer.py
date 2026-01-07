"""Tool call observer for logging and auditing.

Includes Writer monad integration for FP-style structured logging.
"""

import hashlib
import inspect
import json
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Callable, cast

from combinators.writer import LazyCoroResultWriter, Log, writer_ok
from kungfu import Ok, Result
from pydantic import BaseModel

from funcai.agents.tool import Tool


class ToolCallRecord(BaseModel):
    """Record of a single tool call."""

    tool_name: str
    arguments: dict[str, Any]
    result_preview: str
    success: bool
    error: str | None = None
    duration_ms: float
    timestamp: datetime
    cached: bool = False


class ToolCallHistory(BaseModel):
    """Complete history of tool calls during an agent run."""

    calls: list[ToolCallRecord] = []
    total_calls: int = 0
    total_duration_ms: float = 0.0
    cache_hits: int = 0


def _make_cache_key(tool_name: str, arguments: dict[str, Any]) -> str:
    """Create a hash key for tool call caching."""
    data = json.dumps({"tool": tool_name, "args": arguments}, sort_keys=True)
    return hashlib.md5(data.encode()).hexdigest()


SEARCH_TOOLS = {"search_corpus", "find_precedents"}
SOFT_LIMIT_CALLS = 12
HARD_LIMIT_CALLS = 18
SEARCH_HARD_LIMIT = 6

SOFT_LIMIT_MESSAGE = (
    "⚠️  {count} tool calls made. "
    "You have gathered enough information. "
    "FORMULATE YOUR DECISION NOW using the documents already retrieved. "
    "Additional searches are unlikely to find new relevant information."
)
HARD_LIMIT_MESSAGE = (
    "🛑 HARD LIMIT REACHED ({count} tool calls). "
    "STOP ALL SEARCHES IMMEDIATELY. "
    "Formulate your final decision using ONLY the information already gathered. "
    "Any further search calls will be BLOCKED."
)


def _normalize_query(query: str) -> set[str]:
    """Extract normalized keywords from a query for similarity detection."""
    import re

    words = re.findall(r"\b[а-яёА-ЯЁa-zA-Z0-9]{3,}\b", query.lower())
    return set(words)


def _queries_similar(q1: str, q2: str, threshold: float = 0.6) -> bool:
    """Check if two queries are similar based on keyword overlap."""
    words1 = _normalize_query(q1)
    words2 = _normalize_query(q2)
    if not words1 or not words2:
        return False
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return (intersection / union) >= threshold if union > 0 else False


@dataclass
class BlockedResult:
    """Result returned when a tool call is blocked due to limits."""

    reason: str
    cached_result: Any | None = None


@dataclass
class ToolObserver:
    """
    Observer that wraps tools to log their execution.

    Usage:
        observer = ToolObserver()
        wrapped_tools = observer.wrap_tools(tools)
        # Use wrapped_tools with funcai agent
        # After execution:
        history = observer.get_history()
    """

    on_call: Callable[[str, dict[str, Any]], None] | None = None
    on_result: Callable[[str, Any, float], None] | None = None
    on_error: Callable[[str, Exception], None] | None = None
    on_cache_hit: Callable[[str, dict[str, Any]], None] | None = None
    on_warning: Callable[[str], None] | None = None
    enable_cache: bool = True
    cacheable_tools: set[str] = field(
        default_factory=lambda: {
            "search_corpus",
            "get_document",
            "search_document_section",
            "find_precedents",
        }
    )

    _calls: list[ToolCallRecord] = field(default_factory=list)
    _cache: dict[str, Any] = field(default_factory=dict)
    _cache_hits: int = 0
    _search_queries: list[str] = field(default_factory=list)
    _warnings_issued: set[str] = field(default_factory=set)

    def wrap_tools(self, tools: list[Tool]) -> list[Tool]:
        """Wrap all tools with observation logging."""
        return [self._wrap_tool(tool) for tool in tools]

    def _wrap_tool(self, tool: Tool) -> Tool:
        """Wrap a single tool with observation logging and caching."""
        original_fn = tool.fn
        tool_name = tool.name
        param_model = tool.parameters
        is_cacheable = tool_name in self.cacheable_tools
        is_async = inspect.iscoroutinefunction(original_fn)

        if is_async:

            async def async_wrapped_fn(**kwargs: Any) -> Any:
                start_time = time.perf_counter()
                timestamp = datetime.now(timezone.utc)

                arguments = self._extract_arguments(param_model, kwargs)

                if self.enable_cache and is_cacheable:
                    cache_key = _make_cache_key(tool_name, arguments)
                    if cache_key in self._cache:
                        self._cache_hits += 1
                        cached_result = self._cache[cache_key]
                        duration_ms = (time.perf_counter() - start_time) * 1000

                        if self.on_cache_hit:
                            self.on_cache_hit(tool_name, arguments)

                        record = ToolCallRecord(
                            tool_name=tool_name,
                            arguments=arguments,
                            result_preview=self._preview_result(cached_result),
                            success=True,
                            error=None,
                            duration_ms=duration_ms,
                            timestamp=timestamp,
                            cached=True,
                        )
                        self._calls.append(record)

                        if self.on_result:
                            self.on_result(tool_name, cached_result, duration_ms)

                        return cached_result
                else:
                    cache_key = None

                blocked = self._check_limits_and_warn(tool_name, arguments)
                if blocked is not None:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    result = (
                        blocked.cached_result
                        if blocked.cached_result is not None
                        else []
                    )

                    record = ToolCallRecord(
                        tool_name=tool_name,
                        arguments=arguments,
                        result_preview=f"BLOCKED: {blocked.reason}",
                        success=True,
                        error=None,
                        duration_ms=duration_ms,
                        timestamp=timestamp,
                        cached=True,
                    )
                    self._calls.append(record)

                    if self.on_result:
                        self.on_result(tool_name, result, duration_ms)

                    return result

                if self.on_call:
                    self.on_call(tool_name, arguments)

                try:
                    result = await original_fn(**kwargs)
                    duration_ms = (time.perf_counter() - start_time) * 1000

                    if cache_key is not None:
                        self._cache[cache_key] = result

                    result_preview = self._preview_result(result)

                    record = ToolCallRecord(
                        tool_name=tool_name,
                        arguments=arguments,
                        result_preview=result_preview,
                        success=True,
                        error=None,
                        duration_ms=duration_ms,
                        timestamp=timestamp,
                        cached=False,
                    )
                    self._calls.append(record)

                    if self.on_result:
                        self.on_result(tool_name, result, duration_ms)

                    return result

                except Exception as e:
                    duration_ms = (time.perf_counter() - start_time) * 1000

                    record = ToolCallRecord(
                        tool_name=tool_name,
                        arguments=arguments,
                        result_preview="",
                        success=False,
                        error=str(e),
                        duration_ms=duration_ms,
                        timestamp=timestamp,
                        cached=False,
                    )
                    self._calls.append(record)

                    if self.on_error:
                        self.on_error(tool_name, e)

                    raise

            return replace(tool, fn=async_wrapped_fn)

        def wrapped_fn(**kwargs: Any) -> Any:
            start_time = time.perf_counter()
            timestamp = datetime.now(timezone.utc)

            arguments = self._extract_arguments(param_model, kwargs)

            if self.enable_cache and is_cacheable:
                cache_key = _make_cache_key(tool_name, arguments)
                if cache_key in self._cache:
                    self._cache_hits += 1
                    cached_result = self._cache[cache_key]
                    duration_ms = (time.perf_counter() - start_time) * 1000

                    if self.on_cache_hit:
                        self.on_cache_hit(tool_name, arguments)

                    record = ToolCallRecord(
                        tool_name=tool_name,
                        arguments=arguments,
                        result_preview=self._preview_result(cached_result),
                        success=True,
                        error=None,
                        duration_ms=duration_ms,
                        timestamp=timestamp,
                        cached=True,
                    )
                    self._calls.append(record)

                    if self.on_result:
                        self.on_result(tool_name, cached_result, duration_ms)

                    return cached_result
            else:
                cache_key = None

            blocked = self._check_limits_and_warn(tool_name, arguments)
            if blocked is not None:
                duration_ms = (time.perf_counter() - start_time) * 1000
                result = (
                    blocked.cached_result if blocked.cached_result is not None else []
                )

                record = ToolCallRecord(
                    tool_name=tool_name,
                    arguments=arguments,
                    result_preview=f"BLOCKED: {blocked.reason}",
                    success=True,
                    error=None,
                    duration_ms=duration_ms,
                    timestamp=timestamp,
                    cached=True,
                )
                self._calls.append(record)

                if self.on_result:
                    self.on_result(tool_name, result, duration_ms)

                return result

            if self.on_call:
                self.on_call(tool_name, arguments)

            try:
                result = original_fn(**kwargs)
                duration_ms = (time.perf_counter() - start_time) * 1000

                if cache_key is not None:
                    self._cache[cache_key] = result

                result_preview = self._preview_result(result)

                record = ToolCallRecord(
                    tool_name=tool_name,
                    arguments=arguments,
                    result_preview=result_preview,
                    success=True,
                    error=None,
                    duration_ms=duration_ms,
                    timestamp=timestamp,
                    cached=False,
                )
                self._calls.append(record)

                if self.on_result:
                    self.on_result(tool_name, result, duration_ms)

                return result

            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000

                record = ToolCallRecord(
                    tool_name=tool_name,
                    arguments=arguments,
                    result_preview="",
                    success=False,
                    error=str(e),
                    duration_ms=duration_ms,
                    timestamp=timestamp,
                    cached=False,
                )
                self._calls.append(record)

                if self.on_error:
                    self.on_error(tool_name, e)

                raise

        return replace(tool, fn=wrapped_fn)

    def _check_limits_and_warn(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> BlockedResult | None:
        """Check tool call limits and block if exceeded.

        Returns:
            BlockedResult if the call should be blocked, None otherwise.
        """
        total_calls = len(self._calls)

        if (
            total_calls >= SOFT_LIMIT_CALLS
            and "soft_limit" not in self._warnings_issued
        ):
            self._warnings_issued.add("soft_limit")
            if self.on_warning:
                self.on_warning(SOFT_LIMIT_MESSAGE.format(count=total_calls))

        if (
            total_calls >= HARD_LIMIT_CALLS
            and "hard_limit" not in self._warnings_issued
        ):
            self._warnings_issued.add("hard_limit")
            if self.on_warning:
                self.on_warning(HARD_LIMIT_MESSAGE.format(count=total_calls))

        if total_calls >= HARD_LIMIT_CALLS and tool_name in SEARCH_TOOLS:
            return BlockedResult(
                reason="Hard limit reached. No more searches allowed. "
                "Formulate your decision with the information already gathered.",
                cached_result=[],
            )

        if tool_name in SEARCH_TOOLS:
            query = arguments.get("query", "")
            if query:
                search_count = len(self._search_queries)

                if search_count >= SEARCH_HARD_LIMIT:
                    if self.on_warning:
                        self.on_warning(
                            f"🛑 Search limit ({SEARCH_HARD_LIMIT}) reached. "
                            "Formulate decision with available information."
                        )
                    return BlockedResult(
                        reason=f"Search limit reached ({SEARCH_HARD_LIMIT} queries). "
                        "Use the information already gathered to formulate your decision.",
                        cached_result=[],
                    )

                for prev_query in self._search_queries:
                    if _queries_similar(query, prev_query, threshold=0.65):
                        cache_key = _make_cache_key(tool_name, {"query": prev_query})
                        cached = self._cache.get(cache_key)
                        if self.on_warning:
                            self.on_warning(
                                f"🔄 Similar query blocked: '{query[:40]}...' ≈ '{prev_query[:40]}...'. "
                                "Returning cached result."
                            )
                        return BlockedResult(
                            reason="Similar query already made. Use different keywords.",
                            cached_result=cached,
                        )

                self._search_queries.append(query)

        return None

    def _extract_arguments(
        self, param_model: type[BaseModel], kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract argument names and values from a tool call."""
        arguments: dict[str, Any] = {}

        for key, value in kwargs.items():
            arguments[key] = self._serialize_arg(value)

        return arguments

    def _serialize_arg(
        self, value: Any
    ) -> str | int | float | bool | list[Any] | dict[str, Any] | None:
        """Serialize an argument value for logging."""
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            if len(value) > 200:
                return value[:200] + "..."
            return value
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, list):
            result: list[Any] = []
            for i, v in enumerate(value):
                if i >= 10:
                    break
                result.append(self._serialize_arg(v))
            return result
        if isinstance(value, tuple):
            result_t: list[Any] = []
            for i, v in enumerate(value):
                if i >= 10:
                    break
                result_t.append(self._serialize_arg(v))
            return result_t
        if isinstance(value, dict):
            result_d: dict[str, Any] = {}
            count = 0
            for k, v in value.items():
                if count >= 10:
                    break
                result_d[str(k)] = self._serialize_arg(v)
                count += 1
            return result_d
        if isinstance(value, BaseModel):
            return value.model_dump()
        return str(value)[:200]

    def _preview_result(self, result: Any, max_length: int = 300) -> str:
        """Create a preview string of the result."""
        if result is None:
            return "null"
        if isinstance(result, BaseModel):
            preview = result.model_dump_json()
        elif isinstance(result, (list, dict)):
            preview = json.dumps(result, ensure_ascii=False, default=str)
        else:
            preview = str(result)

        if len(preview) > max_length:
            return preview[:max_length] + "..."
        return preview

    def get_history(self) -> ToolCallHistory:
        """Get the complete tool call history."""
        total_duration = sum(call.duration_ms for call in self._calls)
        return ToolCallHistory(
            calls=self._calls,
            total_calls=len(self._calls),
            total_duration_ms=total_duration,
            cache_hits=self._cache_hits,
        )

    def clear(self) -> None:
        """Clear the tool call history and cache."""
        self._calls = []
        self._cache = {}
        self._cache_hits = 0
        self._search_queries = []
        self._warnings_issued = set()

    def get_summary(self) -> str:
        """Get a human-readable summary of tool calls."""
        if not self._calls:
            return "No tool calls recorded"

        lines = [f"Tool calls: {len(self._calls)}"]
        if self._cache_hits > 0:
            lines.append(f"Cache hits: {self._cache_hits}")

        tool_counts: dict[str, int] = {}
        cached_counts: dict[str, int] = {}

        for call in self._calls:
            tool_counts[call.tool_name] = tool_counts.get(call.tool_name, 0) + 1
            if call.cached:
                cached_counts[call.tool_name] = cached_counts.get(call.tool_name, 0) + 1

        for tool_name, count in sorted(tool_counts.items()):
            cached = cached_counts.get(tool_name, 0)
            if cached > 0:
                lines.append(f"  - {tool_name}: {count} ({cached} cached)")
            else:
                lines.append(f"  - {tool_name}: {count}")

        total_ms = sum(call.duration_ms for call in self._calls)
        lines.append(f"Total duration: {total_ms:.1f}ms")

        return "\n".join(lines)


def create_progress_observer(
    on_detail: Callable[[str], None],
    debug: bool = False,
) -> ToolObserver:
    """
    Create an observer that reports to progress tracker.

    Args:
        on_detail: Callback to add detail to progress (e.g., tracker.add_detail)
        debug: If True, output full tool results instead of summaries

    Returns:
        Configured ToolObserver with caching enabled
    """

    def on_call(tool_name: str, arguments: dict[str, Any]) -> None:
        args_str = ", ".join(f"{k}={_short_value(v)}" for k, v in arguments.items())
        on_detail(f"🔧 {tool_name}({args_str})")

    def on_result(tool_name: str, result: Any, duration_ms: float) -> None:
        if debug:
            full_output = _format_debug_result(result)
            on_detail(f"   → ({duration_ms:.0f}ms)")
            for line in full_output.split("\n"):
                on_detail(f"      {line}")
        else:
            preview = _result_summary(result)
            on_detail(f"   → {preview} ({duration_ms:.0f}ms)")

    def on_error(tool_name: str, error: Exception) -> None:
        on_detail(f"   ✗ Error: {str(error)[:100]}")

    def on_cache_hit(tool_name: str, arguments: dict[str, Any]) -> None:
        args_str = ", ".join(f"{k}={_short_value(v)}" for k, v in arguments.items())
        on_detail(f"📦 {tool_name}({args_str}) [cached]")

    def on_warning(message: str) -> None:
        on_detail(message)

    return ToolObserver(
        on_call=on_call,
        on_result=on_result,
        on_error=on_error,
        on_cache_hit=on_cache_hit,
        on_warning=on_warning,
        enable_cache=True,
    )


def _short_value(value: Any, max_len: int = 50) -> str:
    """Create a short string representation of a value."""
    if value is None:
        return "null"
    if isinstance(value, str):
        if len(value) > max_len:
            return f'"{value[:max_len]}..."'
        return f'"{value}"'
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return f"[{len(cast(list[Any], value))} items]"
    if isinstance(value, dict):
        return f"{{{len(cast(dict[Any, Any], value))} keys}}"
    return str(value)[:max_len]


def _result_summary(result: Any) -> str:
    """Create a brief summary of a tool result."""
    if result is None:
        return "No result"
    if isinstance(result, list):
        return f"List with {len(cast(list[Any], result))} items"
    if isinstance(result, BaseModel):
        type_name = type(result).__name__
        doc_id = getattr(result, "doc_id", None)
        if doc_id is not None:
            return f"{type_name}[{doc_id}]"
        return type_name
    if isinstance(result, str):
        if len(result) > 100:
            return f"Text ({len(result)} chars)"
        return result
    return type(result).__name__


def _format_debug_result(result: Any, max_length: int = 2000) -> str:
    """Format full result for debug output."""
    if result is None:
        return "null"

    if isinstance(result, str):
        if len(result) > max_length:
            return result[:max_length] + f"\n... [truncated, {len(result)} total chars]"
        return result

    if isinstance(result, BaseModel):
        json_str = result.model_dump_json(indent=2)
        if len(json_str) > max_length:
            return (
                json_str[:max_length]
                + f"\n... [truncated, {len(json_str)} total chars]"
            )
        return json_str

    if isinstance(result, list):
        items: list[str] = []
        for i, item in enumerate(result):
            if isinstance(item, BaseModel):
                items.append(f"[{i}] {item.model_dump_json()}")
            else:
                items.append(f"[{i}] {item}")
        output = "\n".join(items)
        if len(output) > max_length:
            return output[:max_length] + f"\n... [truncated, {len(output)} total chars]"
        return output

    if isinstance(result, dict):
        json_str = json.dumps(result, ensure_ascii=False, indent=2, default=str)
        if len(json_str) > max_length:
            return (
                json_str[:max_length]
                + f"\n... [truncated, {len(json_str)} total chars]"
            )
        return json_str

    return str(result)[:max_length]


# =============================================================================
# Writer Monad Integration for FP-style structured logging
# =============================================================================


@dataclass(frozen=True)
class ToolCallLogEntry:
    """
    Structured log entry for Writer monad.

    Can be used with LazyCoroResultWriter for FP-style logging
    that doesn't rely on side effects.
    """

    tool_name: str
    duration_ms: float
    success: bool
    timestamp: datetime
    cached: bool = False
    arguments_preview: str = ""
    result_preview: str = ""
    error: str | None = None


def tool_call_to_log_entry(record: ToolCallRecord) -> ToolCallLogEntry:
    """Convert a ToolCallRecord to a ToolCallLogEntry for Writer monad."""
    args_preview = ", ".join(
        f"{k}={_short_value(v)}" for k, v in record.arguments.items()
    )
    return ToolCallLogEntry(
        tool_name=record.tool_name,
        duration_ms=record.duration_ms,
        success=record.success,
        timestamp=record.timestamp,
        cached=record.cached,
        arguments_preview=args_preview,
        result_preview=record.result_preview[:100] if record.result_preview else "",
        error=record.error,
    )


def history_to_log(history: ToolCallHistory) -> Log[ToolCallLogEntry]:
    """Convert ToolCallHistory to a Log for Writer monad operations."""
    entries = [tool_call_to_log_entry(record) for record in history.calls]
    return Log.of(*entries)


def create_log_entry(
    tool_name: str,
    duration_ms: float = 0.0,
    success: bool = True,
    cached: bool = False,
    arguments_preview: str = "",
    result_preview: str = "",
    error: str | None = None,
) -> ToolCallLogEntry:
    """Create a new log entry with current timestamp."""
    return ToolCallLogEntry(
        tool_name=tool_name,
        duration_ms=duration_ms,
        success=success,
        timestamp=datetime.now(timezone.utc),
        cached=cached,
        arguments_preview=arguments_preview,
        result_preview=result_preview,
        error=error,
    )


def tell_tool_call(
    tool_name: str,
    duration_ms: float = 0.0,
    success: bool = True,
) -> LazyCoroResultWriter[None, Any, ToolCallLogEntry]:
    """
    Create a Writer that logs a tool call.

    Usage:
        pipeline = (
            ast_w(tell_tool_call("search_corpus", 150.0, True))
            .then(lambda _: do_something())
            .with_log(create_log_entry("complete"))
        )
    """
    entry = create_log_entry(tool_name, duration_ms, success)
    return writer_ok(None, entry)


class WriterObserver:
    """
    Observer that collects tool calls as Writer monad logs.

    This is a pure FP alternative to callback-based ToolObserver.
    Instead of side effects, it accumulates logs that can be
    extracted with get_log().

    Usage:
        observer = WriterObserver()
        wrapped_tools = observer.wrap_tools(tools)
        # Use tools...
        log = observer.get_log()  # Log[ToolCallLogEntry]
    """

    def __init__(self) -> None:
        self._entries: list[ToolCallLogEntry] = []

    def wrap_tools(self, tools: list[Tool]) -> list[Tool]:
        """Wrap tools to collect logs."""
        return [self._wrap_tool(tool) for tool in tools]

    def _wrap_tool(self, tool: Tool) -> Tool:
        """Wrap a single tool to collect logs."""
        original_fn = tool.fn
        tool_name = tool.name
        is_async = inspect.iscoroutinefunction(original_fn)

        if is_async:

            async def async_wrapped(**kwargs: Any) -> Any:
                start = time.perf_counter()
                timestamp = datetime.now(timezone.utc)
                args_preview = ", ".join(
                    f"{k}={_short_value(v)}" for k, v in kwargs.items()
                )

                try:
                    result = await original_fn(**kwargs)
                    duration_ms = (time.perf_counter() - start) * 1000

                    self._entries.append(
                        ToolCallLogEntry(
                            tool_name=tool_name,
                            duration_ms=duration_ms,
                            success=True,
                            timestamp=timestamp,
                            cached=False,
                            arguments_preview=args_preview,
                            result_preview=_result_summary(result),
                            error=None,
                        )
                    )
                    return result

                except Exception as e:
                    duration_ms = (time.perf_counter() - start) * 1000
                    self._entries.append(
                        ToolCallLogEntry(
                            tool_name=tool_name,
                            duration_ms=duration_ms,
                            success=False,
                            timestamp=timestamp,
                            cached=False,
                            arguments_preview=args_preview,
                            result_preview="",
                            error=str(e),
                        )
                    )
                    raise

            return replace(tool, fn=async_wrapped)

        def sync_wrapped(**kwargs: Any) -> Any:
            start = time.perf_counter()
            timestamp = datetime.now(timezone.utc)
            args_preview = ", ".join(
                f"{k}={_short_value(v)}" for k, v in kwargs.items()
            )

            try:
                result = original_fn(**kwargs)
                duration_ms = (time.perf_counter() - start) * 1000

                self._entries.append(
                    ToolCallLogEntry(
                        tool_name=tool_name,
                        duration_ms=duration_ms,
                        success=True,
                        timestamp=timestamp,
                        cached=False,
                        arguments_preview=args_preview,
                        result_preview=_result_summary(result),
                        error=None,
                    )
                )
                return result

            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000
                self._entries.append(
                    ToolCallLogEntry(
                        tool_name=tool_name,
                        duration_ms=duration_ms,
                        success=False,
                        timestamp=timestamp,
                        cached=False,
                        arguments_preview=args_preview,
                        result_preview="",
                        error=str(e),
                    )
                )
                raise

        return replace(tool, fn=sync_wrapped)

    def get_log(self) -> Log[ToolCallLogEntry]:
        """Get accumulated logs as a Writer Log."""
        return Log.of(*self._entries)

    def get_entries(self) -> list[ToolCallLogEntry]:
        """Get raw log entries."""
        return self._entries.copy()

    def clear(self) -> None:
        """Clear accumulated logs."""
        self._entries = []

    def to_writer[T](
        self, value: T
    ) -> LazyCoroResultWriter[T, Any, Log[ToolCallLogEntry]]:
        """
        Wrap a value with accumulated logs as a Writer.

        Usage:
            result = await some_operation()
            writer_result = observer.to_writer(result)
        """
        log = self.get_log()

        async def run() -> tuple[Result[T, Any], Log[ToolCallLogEntry]]:
            return Ok(value), log

        return LazyCoroResultWriter[T, Any, Log[ToolCallLogEntry]](run)


__all__ = [
    "ToolObserver",
    "ToolCallRecord",
    "ToolCallHistory",
    "BlockedResult",
    "create_progress_observer",
    # Writer monad integration
    "ToolCallLogEntry",
    "WriterObserver",
    "tool_call_to_log_entry",
    "history_to_log",
    "create_log_entry",
    "tell_tool_call",
]
