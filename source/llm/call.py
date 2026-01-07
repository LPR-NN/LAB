"""LLM call abstractions with retry, timeout, and error handling.

This module provides FP-style wrappers around LLM calls using combinators
for retry logic, timeouts, and proper error handling.
"""

from dataclasses import dataclass
from typing import Any

from combinators import ast
from combinators.control import RetryPolicy
from kungfu import Error, LazyCoroResult, Ok, Result
from pydantic import BaseModel

from funcai.agents.abc import AgentError, AgentResponse
from funcai.agents.agent import agent
from funcai.agents.tool import Tool
from funcai.core.dialogue import Dialogue
from funcai.core.provider import ABCAIProvider


@dataclass(frozen=True)
class LLMError:
    """Unified error type for LLM operations."""

    message: str
    retryable: bool
    original: Any | None = None

    def __str__(self) -> str:
        return self.message


# Error patterns that indicate retryable errors
RETRYABLE_PATTERNS = [
    "rate_limit",
    "rate limit",
    "overloaded",
    "timeout",
    "529",
    "503",
    "502",
    "too many requests",
    "temporarily unavailable",
    "service unavailable",
]


def is_retryable_error(error: Any) -> bool:
    """
    Determine if an error is retryable.

    Checks error message for patterns indicating transient failures
    like rate limits, overloaded servers, timeouts.
    """
    msg = str(error).lower()
    return any(pattern in msg for pattern in RETRYABLE_PATTERNS)


def _wrap_llm_error(error: Any) -> LLMError:
    """Wrap any error into LLMError with retryable detection."""
    return LLMError(
        message=str(error),
        retryable=is_retryable_error(error),
        original=error,
    )


def llm_interpret[T: BaseModel](
    dialogue: Dialogue,
    provider: ABCAIProvider[Any],
    schema: type[T],
    *,
    timeout_seconds: float = 120,
    retry_times: int = 3,
) -> LazyCoroResult[T, LLMError]:
    """
    LLM interpretation with retry and timeout.

    Uses combinators for:
    - Exponential backoff retry with jitter on retryable errors
    - Timeout protection
    - Unified error handling

    Args:
        dialogue: The conversation to interpret
        provider: LLM provider (OpenAI, Anthropic, etc.)
        schema: Pydantic model for structured output
        timeout_seconds: Maximum time for the call
        retry_times: Number of retry attempts

    Returns:
        LazyCoroResult[T, LLMError] - lazy computation that can be composed

    Example:
        result = await llm_interpret(dialogue, provider, MySchema)
        match result:
            case Ok(parsed): ...
            case Error(e): print(e.message)
    """
    # Get the base interpretation as LazyCoroResult
    base_interp = dialogue.interpret(provider, schema)

    # Build the pipeline using AST/Flow API
    pipeline = (
        ast(base_interp)
        .retry(
            policy=RetryPolicy[LLMError].exponential_jitter(
                times=retry_times,
                initial=1.0,
                multiplier=2.0,
                max_delay=30.0,
                jitter_factor=0.3,
                retry_on=is_retryable_error,
            )
        )
        .timeout(seconds=timeout_seconds)
        .lower()
    )

    # Map errors to LLMError
    async def run() -> Result[T, LLMError]:
        result = await pipeline
        match result:
            case Ok(value):
                return Ok(value)
            case Error(e):
                return Error(_wrap_llm_error(e))

    return LazyCoroResult(run)


def llm_agent_call[S: BaseModel](
    dialogue: Dialogue,
    provider: ABCAIProvider[Any],
    tools: list[Tool],
    *,
    max_steps: int = 10,
    schema: type[S],
    timeout_seconds: float = 300,
    retry_times: int = 3,
) -> LazyCoroResult[AgentResponse[S], LLMError]:
    """
    LLM agent call with retry and timeout.

    Wraps funcai's agent() with:
    - Exponential backoff retry with jitter on retryable errors
    - Timeout protection for entire agent loop
    - Unified error handling

    Args:
        dialogue: The conversation to process
        provider: LLM provider
        tools: List of tools available to the agent
        max_steps: Maximum agent iterations
        schema: Pydantic model for structured output
        timeout_seconds: Maximum time for entire agent loop
        retry_times: Number of retry attempts

    Returns:
        LazyCoroResult[AgentResponse[S], LLMError]

    Example:
        result = await llm_agent_call(
            dialogue, provider, tools,
            max_steps=10, schema=Decision
        )
    """
    # Get the base agent call as LazyCoroResult
    base_interp = agent(dialogue, provider, tools, max_steps, schema=schema)

    # Build the pipeline
    pipeline = (
        ast(base_interp)
        .retry(
            policy=RetryPolicy[AgentError].exponential_jitter(
                times=retry_times,
                initial=2.0,
                multiplier=2.0,
                max_delay=60.0,
                jitter_factor=0.3,
                retry_on=lambda e: is_retryable_error(e),
            )
        )
        .timeout(seconds=timeout_seconds)
        .lower()
    )

    # Map errors to LLMError
    async def run() -> Result[AgentResponse[S], LLMError]:
        result = await pipeline
        match result:
            case Ok(response):
                return Ok(response)
            case Error(e):
                return Error(_wrap_llm_error(e))

    return LazyCoroResult(run)


__all__ = [
    "LLMError",
    "is_retryable_error",
    "llm_interpret",
    "llm_agent_call",
]
