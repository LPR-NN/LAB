from typing import Literal

from kungfu import cache
from pydantic import Field
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

    auth_username: str = Field(init=False, alias="AUTH_USERNAME")
    auth_password: str = Field(init=False, alias="AUTH_PASSWORD")

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
        default=None,
        alias="VECTOR_CACHE_DIR",
        description="Directory for embedding cache (pickle)",
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
