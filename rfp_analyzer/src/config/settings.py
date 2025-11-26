"""Application settings using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # OpenAI Configuration
    openai_api_key: SecretStr = Field(..., description="OpenAI API Key")
    openai_model: str = Field(default="gpt-4o", description="OpenAI model name")
    openai_embedding_model: str = Field(
        default="text-embedding-3-large", description="OpenAI embedding model"
    )
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)

    # LangSmith Configuration
    langchain_tracing_v2: bool = Field(default=True)
    langchain_api_key: SecretStr | None = Field(default=None)
    langchain_project: str = Field(default="rfp-analyzer")

    # Application Settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")
    environment: Literal["development", "staging", "production"] = Field(default="development")

    # Vector Store Settings
    chroma_persist_directory: Path = Field(default=Path("./data/chroma_db"))
    collection_name: str = Field(default="rfp_documents")

    # Retrieval Settings
    top_k_results: int = Field(default=10, ge=1, le=100)
    similarity_threshold: float = Field(default=0.3, ge=0.0, le=1.0)  # Lower threshold for better recall
    chunk_size: int = Field(default=1000, ge=100, le=4000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)

    # Data Directory
    data_directory: Path = Field(default=Path("./data"))

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

