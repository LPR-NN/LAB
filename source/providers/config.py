"""Provider configuration."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    """Configuration for an LLM provider."""

    provider_type: Literal["openai", "anthropic", "openrouter", "lmstudio"] = Field(
        description="Type of LLM provider"
    )

    model: str = Field(
        description="Model identifier (e.g., 'gpt-4o', 'claude-sonnet-4-20250514')"
    )

    api_key: str | None = Field(
        default=None,
        description="API key (if None, will use environment variable)",
    )

    temperature: float = Field(
        default=0.0, description="Temperature for generation (0.0 for deterministic)"
    )

    max_tokens: int = Field(default=4096, description="Maximum tokens in response")

    extra: dict[str, Any] = Field(
        default_factory=dict, description="Provider-specific extra options"
    )

    @property
    def is_deterministic(self) -> bool:
        return self.temperature == 0.0
