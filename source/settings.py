import json
from typing import Literal

from kungfu import cache
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderType = Literal["openai", "anthropic", "openrouter", "lmstudio"]
SearchMode = Literal["tfidf", "vector", "hybrid"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")

    # Multiple users: JSON dict {"user1": "pass1", "user2": "pass2"}
    auth_users: dict[str, str] = Field(default_factory=dict, alias="AUTH_USERS")

    @field_validator("auth_users", mode="before")
    @classmethod
    def parse_auth_users(cls, v: str | dict[str, str] | None) -> dict[str, str]:
        if v is None:
            return {}
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            if not v.strip():
                return {}
            return json.loads(v)
        return {}

    # Search limits
    max_search_calls: int = Field(
        default=10,
        alias="MAX_SEARCH_CALLS",
        description="Maximum number of search_corpus/find_precedents calls",
    )

    # Re-ranking settings (improves search quality at no API cost)
    use_reranker: bool = Field(
        default=False,
        alias="USE_RERANKER",
        description="Enable cross-encoder re-ranking for search results",
    )
    reranker_model: str = Field(
        default="BAAI/bge-reranker-base",
        alias="RERANKER_MODEL",
        description="Cross-encoder model for re-ranking (local, no API)",
    )

    # Embedding settings
    embedding_model: str = Field(
        default="intfloat/multilingual-e5-base",
        alias="EMBEDDING_MODEL",
        description="Sentence transformer model for embeddings",
    )
    search_mode: SearchMode = Field(
        default="hybrid",
        alias="SEARCH_MODE",
        description="Search mode: tfidf, vector, or hybrid",
    )
    hybrid_alpha: float = Field(
        default=0.7,
        alias="HYBRID_ALPHA",
        description="Weight for vector similarity in hybrid mode (0-1)",
    )
    vector_cache_dir: str | None = Field(
        default=".cache/embeddings",
        alias="VECTOR_CACHE_DIR",
        description="Directory for embedding cache (pickle). Set to empty string to disable.",
    )

    # Chat settings
    chat_daily_limit: int = Field(
        default=5,
        alias="CHAT_DAILY_LIMIT",
        description="Maximum chat questions per user per day",
    )
    chat_max_question_length: int = Field(
        default=2000,
        alias="CHAT_MAX_QUESTION_LENGTH",
        description="Maximum length of chat question in characters",
    )

    def get_api_key(self, provider: ProviderType) -> str | None:
        match provider:
            case "openai":
                return self.openai_api_key
            case "anthropic":
                return self.anthropic_api_key
            case "openrouter":
                return self.openrouter_api_key
            case "lmstudio":
                return None  # LM Studio doesn't need API key


@cache
def get_settings() -> Settings:
    return Settings()
