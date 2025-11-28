"""Application settings using Pydantic BaseSettings.

All configuration should be accessed through get_settings() function.
Never read os.environ directly in application code.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """PostgreSQL settings."""

    model_config = SettingsConfigDict(env_prefix="POSTGRES_")

    host: str = "localhost"
    port: int = 5432
    user: str = "rag_user"
    password: SecretStr = SecretStr("rag_password")
    database: str = "rag_db"
    pool_size: int = 5
    max_overflow: int = 10

    @property
    def async_url(self) -> str:
        """Async database URL for SQLAlchemy."""
        return (
            f"postgresql+asyncpg://{self.user}:"
            f"{self.password.get_secret_value()}@"
            f"{self.host}:{self.port}/{self.database}"
        )

    @property
    def sync_url(self) -> str:
        """Sync database URL for migrations."""
        return (
            f"postgresql://{self.user}:"
            f"{self.password.get_secret_value()}@"
            f"{self.host}:{self.port}/{self.database}"
        )


class QdrantSettings(BaseSettings):
    """Qdrant vector database settings."""

    model_config = SettingsConfigDict(env_prefix="QDRANT_")

    host: str = "localhost"
    port: int = 6333
    grpc_port: int = 6334
    collection_name: str = "documents"
    use_grpc: bool = True
    api_key: SecretStr | None = None

    @property
    def url(self) -> str:
        """REST API URL."""
        return f"http://{self.host}:{self.port}"


class RedisSettings(BaseSettings):
    """Redis settings."""

    model_config = SettingsConfigDict(env_prefix="REDIS_")

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: SecretStr | None = None
    cache_ttl: int = 3600
    semantic_cache_threshold: float = 0.95

    @property
    def url(self) -> str:
        """Redis connection URL."""
        if self.password:
            return (
                f"redis://:{self.password.get_secret_value()}@"
                f"{self.host}:{self.port}/{self.db}"
            )
        return f"redis://{self.host}:{self.port}/{self.db}"


class MinIOSettings(BaseSettings):
    """MinIO (S3 compatible) object storage settings."""

    model_config = SettingsConfigDict(env_prefix="MINIO_")

    endpoint: str = "localhost:9000"
    access_key: str = "minioadmin"
    secret_key: SecretStr = SecretStr("minioadmin")
    bucket_name: str = "documents"
    secure: bool = False

    @property
    def endpoint_url(self) -> str:
        """Full endpoint URL with protocol."""
        protocol = "https" if self.secure else "http"
        return f"{protocol}://{self.endpoint}"


class OpenAISettings(BaseSettings):
    """OpenAI API settings."""

    model_config = SettingsConfigDict(
        env_prefix="OPENAI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: SecretStr = SecretStr("")
    model: str = "gpt-4o"
    fallback_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    max_retries: int = 3
    timeout: float = 30.0


class CohereSettings(BaseSettings):
    """Cohere API settings."""

    model_config = SettingsConfigDict(
        env_prefix="COHERE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: SecretStr = SecretStr("")
    rerank_model: str = "rerank-v3.5"
    max_retries: int = 3
    timeout: float = 30.0


class LangfuseSettings(BaseSettings):
    """Langfuse observability settings."""

    model_config = SettingsConfigDict(env_prefix="LANGFUSE_")

    public_key: str = ""
    secret_key: SecretStr = SecretStr("")
    host: str = "http://localhost:3000"
    enabled: bool = True


class ChunkingSettings(BaseSettings):
    """Document chunking settings."""

    model_config = SettingsConfigDict(env_prefix="CHUNKING_")

    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: list[str] = Field(default=["\n\n", "\n", ". ", " ", ""])


class SearchSettings(BaseSettings):
    """Search and retrieval settings."""

    model_config = SettingsConfigDict(env_prefix="SEARCH_")

    default_top_k: int = 10
    rerank_top_n: int = 5
    hybrid_alpha: float = 0.5
    min_score_threshold: float = 0.0


class Settings(BaseSettings):
    """Root application settings.

    Access via get_settings() to ensure singleton behavior.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    app_name: str = "Enterprise Agentic RAG"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"
    environment: Literal["development", "staging", "production"] = "development"

    # Provider selection (for future AWS Bedrock migration)
    provider: Literal["openai", "bedrock"] = "openai"

    # Nested settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    minio: MinIOSettings = Field(default_factory=MinIOSettings)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    cohere: CohereSettings = Field(default_factory=CohereSettings)
    langfuse: LangfuseSettings = Field(default_factory=LangfuseSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is a valid Python logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            msg = f"Invalid log level: {v}. Must be one of {valid_levels}"
            raise ValueError(msg)
        return v.upper()


@lru_cache
def get_settings() -> Settings:
    """Get singleton settings instance.

    Returns:
        Settings instance loaded from environment variables.
    """
    return Settings()
