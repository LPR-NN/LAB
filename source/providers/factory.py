"""Provider factory for creating LLM providers."""

from typing import Any, Literal

from kungfu import Nothing, Option, Some

from funcai.core.provider import ABCAIProvider
from funcai.std.openai_provider import OpenAIProvider
from source.providers.anthropic_provider import AnthropicProvider
from source.providers.config import ProviderConfig
from source.providers.local_provider import LocalOpenAIProvider


class ProviderFactory:
    """Factory for creating LLM providers."""

    @staticmethod
    def create(
        provider_type: Literal["openai", "anthropic", "openrouter", "lmstudio"],
        model: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        base_url: str | None = None,
    ) -> ABCAIProvider[Any]:
        """
        Create an LLM provider.

        Args:
            provider_type: Type of provider ("openai", "anthropic", "openrouter", "lmstudio")
            model: Model identifier
            api_key: API key (optional, will use env var if None)
            temperature: Temperature for generation (0.0 for deterministic)
            max_tokens: Maximum tokens in response (Anthropic only)
            base_url: Custom base URL (for lmstudio/local providers)

        Returns:
            Configured provider instance
        """
        api_key_opt: Option[str] = Some(api_key) if api_key else Nothing()
        temp_opt: Option[float] = Some(temperature) if temperature > 0 else Nothing()

        match provider_type:
            case "openai":
                return OpenAIProvider(
                    model=model,
                    api_key=api_key_opt,
                    temperature=temp_opt,
                )
            case "openrouter":
                # OpenRouter is OpenAI-compatible with custom base URL
                return LocalOpenAIProvider(
                    model=model,
                    base_url="https://openrouter.ai/api/v1",
                    api_key=api_key or "",
                    temperature=temp_opt,
                )
            case "lmstudio":
                # LM Studio uses OpenAI-compatible API on localhost
                lmstudio_url = base_url or "http://localhost:1234/v1"
                return LocalOpenAIProvider(
                    model=model,
                    base_url=lmstudio_url,
                    api_key="lm-studio",  # dummy key
                    temperature=temp_opt,
                )
            case "anthropic":
                return AnthropicProvider(
                    model=model,
                    api_key=api_key_opt,
                    temperature=temp_opt,
                    max_tokens=max_tokens,
                )

    @staticmethod
    def from_config(config: ProviderConfig) -> ABCAIProvider[Any]:
        """
        Create a provider from configuration.

        Args:
            config: Provider configuration

        Returns:
            Configured provider instance
        """
        return ProviderFactory.create(
            provider_type=config.provider_type,
            model=config.model,
            api_key=config.api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            **config.extra,
        )

    @staticmethod
    def default_openai(model: str = "gpt-4o") -> ABCAIProvider[Any]:
        """Create default OpenAI provider with deterministic settings."""
        return ProviderFactory.create("openai", model, temperature=0.0)

    @staticmethod
    def default_anthropic(
        model: str = "claude-sonnet-4-20250514",
    ) -> ABCAIProvider[Any]:
        """Create default Anthropic provider with deterministic settings."""
        return ProviderFactory.create("anthropic", model, temperature=0.0)


__all__ = ["ProviderFactory"]
