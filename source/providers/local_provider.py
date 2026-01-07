"""OpenAI-compatible provider for local LLM servers (LM Studio, Ollama, etc.)."""

import json
from typing import Any

from funcai.agents.tool import Tool
from funcai.core.message import Message, Role, assistant
from funcai.core.provider import ABCAIProvider, AIResponse
from funcai.core.types import ToolCall
from kungfu import Error, Nothing, Ok, Option, Result, Some
from openai import APIError, AsyncOpenAI
from pydantic import BaseModel


class LocalProviderError:
    """Error from local LLM provider."""

    def __init__(self, message: str, code: str | None = None):
        self.message = message
        self.code = code

    def __str__(self) -> str:
        return self.message


def _tool_to_openai(tool: Tool) -> dict[str, Any]:
    """Convert funcai Tool to OpenAI format."""
    schema = tool.parameters.model_json_schema()
    schema["additionalProperties"] = False

    if "properties" in schema:
        schema["required"] = list(schema.get("properties", {}).keys())

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": schema,
        },
    }


def _message_to_openai(message: Message) -> dict[str, Any]:
    """Convert funcai Message to OpenAI format."""
    if message.role == Role.TOOL:
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id.unwrap_or_none(),
            "content": message.text.unwrap_or(""),
        }

    if message.role == Role.ASSISTANT and message.has_tool_calls:
        return {
            "role": "assistant",
            "content": message.text.unwrap_or_none(),
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in message.tool_calls
            ],
        }

    return {"role": message.role.value, "content": message.text.unwrap_or("")}


class LocalOpenAIProvider(ABCAIProvider[LocalProviderError]):
    """
    OpenAI-compatible provider for local servers.

    Works with:
    - LM Studio (default: http://localhost:1234/v1)
    - Ollama (http://localhost:11434/v1)
    - Any OpenAI-compatible API
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "lm-studio",
        temperature: Option[float] = Nothing(),
    ) -> None:
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature

    async def send_messages[S: BaseModel](
        self,
        messages: list[Message],
        *,
        schema: Option[type[S]] = Nothing(),
        tools: list[Tool] | None = None,
    ) -> Result[AIResponse[S], LocalProviderError]:
        openai_messages = [_message_to_openai(m) for m in messages]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
        }

        match self.temperature:
            case Some(temp):
                kwargs["temperature"] = temp
            case _:
                pass

        # Local models often don't support structured outputs, use JSON mode
        match schema:
            case Some(s):
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "schema": s.model_json_schema(),
                    },
                }
            case _:
                pass

        if tools:
            kwargs["tools"] = [_tool_to_openai(t) for t in tools]

        try:
            response = await self.client.chat.completions.create(**kwargs)
        except APIError as e:
            return Error(LocalProviderError(str(e), e.code))

        choice = response.choices[0]
        msg = choice.message

        tool_calls: list[ToolCall] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                )

        response_msg = assistant(text=msg.content, tool_calls=tool_calls)

        # Parse structured response if schema provided
        parsed: Option[S] = Nothing()
        match schema:
            case Some(s) if msg.content:
                try:
                    data = json.loads(msg.content)
                    parsed = Some(s.model_validate(data))
                except (json.JSONDecodeError, Exception):
                    pass
            case _:
                pass

        meta: dict[str, Any] = {
            "model": response.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens
                if response.usage
                else None,
                "completion_tokens": response.usage.completion_tokens
                if response.usage
                else None,
                "total_tokens": response.usage.total_tokens if response.usage else None,
            },
            "finish_reason": choice.finish_reason,
        }

        return Ok(
            AIResponse(
                message=response_msg,
                tool_calls=tool_calls,
                parsed=parsed,
                meta=meta,
            )
        )


__all__ = ["LocalOpenAIProvider", "LocalProviderError"]
