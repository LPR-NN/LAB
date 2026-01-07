"""Anthropic Claude provider for funcai."""

from dataclasses import dataclass, field
from typing import Any

import httpx
from kungfu import Error, Result, Ok, Option, Nothing, Some, from_optional
from anthropic import AsyncAnthropic, APIError
from pydantic import BaseModel

from funcai.core.message import Message, Role, assistant
from funcai.core.provider import ABCAIProvider, AIResponse
from funcai.core.types import ToolCall
from funcai.agents.tool import Tool


@dataclass(frozen=True)
class AnthropicError:
    """Error from Anthropic API."""

    message: str
    request: httpx.Request
    body: Option[object] = field(default_factory=Nothing)

    @classmethod
    def from_api_error(cls, e: APIError) -> "AnthropicError":
        return cls(
            message=e.message,
            request=e.request,
            body=from_optional(e.body),
        )


def _tool_to_anthropic(tool: Tool) -> dict[str, Any]:
    """Convert funcai Tool to Anthropic format."""
    schema = tool.parameters.model_json_schema()
    # Remove $defs if present (Anthropic doesn't need them at top level)
    schema.pop("$defs", None)

    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": schema,
    }


def _schema_to_tool[S: BaseModel](schema: type[S]) -> dict[str, Any]:
    """Convert Pydantic schema to tool for structured output."""
    json_schema = schema.model_json_schema()
    json_schema.pop("$defs", None)

    return {
        "name": f"respond_with_{schema.__name__.lower()}",
        "description": f"Use this tool to respond with a structured {schema.__name__}. "
        f"Always use this tool to provide your final answer.",
        "input_schema": json_schema,
    }


def _message_to_anthropic(message: Message) -> dict[str, Any]:
    """Convert funcai Message to Anthropic format."""
    # Tool result - must be user message with tool_result content
    if message.role == Role.TOOL:
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id.unwrap(),
                    "content": message.text.unwrap_or(""),
                }
            ],
        }

    # Assistant with tool calls
    if message.role == Role.ASSISTANT and message.has_tool_calls:
        content: list[dict[str, Any]] = []

        # Add text if present
        match message.text:
            case Some(text) if text:
                content.append({"type": "text", "text": text})
            case _:
                pass

        # Add tool use blocks
        for tc in message.tool_calls:
            content.append(
                {
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments,
                }
            )

        return {"role": "assistant", "content": content}

    # Regular text message
    return {"role": message.role.value, "content": message.text.unwrap_or("")}


def _extract_system_message(
    messages: list[Message],
) -> tuple[str | None, list[Message]]:
    """
    Extract system message from the list.

    Anthropic requires system message as a separate parameter.
    Returns (system_text, remaining_messages).
    """
    system_text = None
    remaining: list[Message] = []

    for msg in messages:
        if msg.role == Role.SYSTEM:
            # Concatenate multiple system messages
            text = msg.text.unwrap_or("")
            if system_text is None:
                system_text = text
            else:
                system_text = f"{system_text}\n\n{text}"
        else:
            remaining.append(msg)

    return system_text, remaining


def _parse_tool_calls_from_response(content: list[dict[str, Any]]) -> list[ToolCall]:
    """Parse tool calls from Anthropic response content blocks."""
    tool_calls: list[ToolCall] = []

    for block in content:
        if block.get("type") == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=block["id"],
                    name=block["name"],
                    arguments=block["input"],
                )
            )

    return tool_calls


def _extract_text_from_response(content: list[dict[str, Any]]) -> str | None:
    """Extract text content from Anthropic response."""
    texts: list[str] = []

    for block in content:
        if block.get("type") == "text":
            texts.append(block["text"])

    return "\n".join(texts) if texts else None


class AnthropicProvider(ABCAIProvider[AnthropicError]):
    """Anthropic Claude provider for funcai."""

    def __init__(
        self,
        model: str,
        api_key: Option[str] = Nothing(),
        temperature: Option[float] = Nothing(),
        max_tokens: int = 4096,
    ) -> None:
        self.client = AsyncAnthropic(api_key=api_key.unwrap_or_none())
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def send_messages[S: BaseModel](
        self,
        messages: list[Message],
        *,
        schema: Option[type[S]] = Nothing(),
        tools: list[Tool] | None = None,
    ) -> Result[AIResponse[S], AnthropicError]:
        # Extract system message
        system_text, conversation = _extract_system_message(messages)

        # Convert remaining messages
        anthropic_messages = [_message_to_anthropic(m) for m in conversation]

        # Build request kwargs
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": self.max_tokens,
        }

        # Add system if present
        if system_text:
            kwargs["system"] = system_text

        # Add temperature if specified
        match self.temperature:
            case Some(temp):
                kwargs["temperature"] = temp
            case _:
                pass

        # Build tools list
        anthropic_tools: list[dict[str, Any]] = []

        if tools:
            anthropic_tools.extend([_tool_to_anthropic(t) for t in tools])

        # For structured output, add schema as a tool
        schema_tool_name: str | None = None
        match schema:
            case Some(s):
                schema_tool = _schema_to_tool(s)
                schema_tool_name = schema_tool["name"]
                anthropic_tools.append(schema_tool)
                # Force the model to use the schema tool
                kwargs["tool_choice"] = {"type": "tool", "name": schema_tool_name}
            case _:
                pass

        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        try:
            response = await self.client.messages.create(**kwargs)
        except APIError as e:
            return Error(AnthropicError.from_api_error(e))

        # Parse response content
        content = response.content
        content_dicts = [
            block.model_dump() if hasattr(block, "model_dump") else block
            for block in content
        ]

        # Extract tool calls and text
        tool_calls = _parse_tool_calls_from_response(content_dicts)
        text = _extract_text_from_response(content_dicts)

        # Handle structured output via schema tool
        parsed: Option[S] = Nothing()
        filtered_tool_calls: list[ToolCall] = []

        for tc in tool_calls:
            if schema_tool_name and tc.name == schema_tool_name:
                # This is the structured output, parse it
                match schema:
                    case Some(s):
                        try:
                            parsed_obj = s.model_validate(tc.arguments)
                            parsed = Some(parsed_obj)
                        except Exception:
                            # Validation failed, keep as regular tool call
                            filtered_tool_calls.append(tc)
                    case _:
                        filtered_tool_calls.append(tc)
            else:
                filtered_tool_calls.append(tc)

        # Create response message
        response_msg = assistant(text=text, tool_calls=filtered_tool_calls)

        # Build metadata
        meta: dict[str, Any] = {
            "model": response.model,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            "stop_reason": response.stop_reason,
        }

        return Ok(
            AIResponse(
                message=response_msg,
                tool_calls=filtered_tool_calls,
                parsed=parsed,
                meta=meta,
            )
        )


__all__ = ["AnthropicProvider", "AnthropicError"]
