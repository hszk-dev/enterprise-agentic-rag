# Phase 1 Detailed Design - Core RAG with Hybrid Search

**Last Updated:** 2025-11-28

## 1. Domain Entities (詳細設計)

### 1.1 Value Objects

```python
# src/domain/value_objects.py

from enum import Enum
from dataclasses import dataclass

class DocumentStatus(str, Enum):
    """ドキュメント処理ステータス"""
    PENDING = "pending"           # アップロード直後
    PROCESSING = "processing"     # 処理中
    COMPLETED = "completed"       # 処理完了
    FAILED = "failed"             # 処理失敗

class ContentType(str, Enum):
    """サポートするコンテンツタイプ"""
    PDF = "application/pdf"
    DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    TXT = "text/plain"
    MD = "text/markdown"
    HTML = "text/html"

@dataclass(frozen=True)
class TokenUsage:
    """LLMトークン使用量"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str = "gpt-5"

    @property
    def estimated_cost_usd(self) -> float:
        """GPT-5の概算コスト (2025年11月時点)"""
        # GPT-5: $1.25/1M input, $10/1M output
        # GPT-5 mini: $0.25/1M input, $2/1M output
        if "mini" in self.model.lower():
            input_cost = (self.prompt_tokens / 1_000_000) * 0.25
            output_cost = (self.completion_tokens / 1_000_000) * 2
        elif "nano" in self.model.lower():
            input_cost = (self.prompt_tokens / 1_000_000) * 0.05
            output_cost = (self.completion_tokens / 1_000_000) * 0.40
        else:  # GPT-5
            input_cost = (self.prompt_tokens / 1_000_000) * 1.25
            output_cost = (self.completion_tokens / 1_000_000) * 10
        return input_cost + output_cost

@dataclass(frozen=True)
class ChunkMetadata:
    """チャンクのメタデータ"""
    page_number: int | None = None
    section_title: str | None = None
    source_url: str | None = None
```

### 1.2 Entities

```python
# src/domain/entities.py

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4
from typing import Any

@dataclass
class Document:
    """
    アップロードされたドキュメントを表すエンティティ。

    Invariants:
    - id は常に有効なUUID
    - filename は空文字列ではない
    - size_bytes は0以上
    """
    id: UUID
    filename: str
    content_type: ContentType
    size_bytes: int
    status: DocumentStatus = DocumentStatus.PENDING
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    error_message: str | None = None
    chunk_count: int = 0
    file_path: str | None = None  # BlobStorage上のパス (MinIO/S3)

    @classmethod
    def create(
        cls,
        filename: str,
        content_type: ContentType,
        size_bytes: int,
        metadata: dict[str, Any] | None = None,
    ) -> "Document":
        """ファクトリメソッド"""
        return cls(
            id=uuid4(),
            filename=filename,
            content_type=content_type,
            size_bytes=size_bytes,
            metadata=metadata or {},
        )

    def mark_processing(self) -> None:
        """処理開始"""
        self.status = DocumentStatus.PROCESSING
        self.updated_at = datetime.utcnow()

    def mark_completed(self, chunk_count: int) -> None:
        """処理完了"""
        self.status = DocumentStatus.COMPLETED
        self.chunk_count = chunk_count
        self.updated_at = datetime.utcnow()

    def mark_failed(self, error_message: str) -> None:
        """処理失敗"""
        self.status = DocumentStatus.FAILED
        self.error_message = error_message
        self.updated_at = datetime.utcnow()

    def set_file_path(self, file_path: str) -> None:
        """BlobStorage上のパスを設定"""
        self.file_path = file_path
        self.updated_at = datetime.utcnow()


@dataclass
class Chunk:
    """
    ドキュメントを分割したチャンク。
    検索の基本単位となる。
    """
    id: UUID
    document_id: UUID
    content: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: dict[str, Any] = field(default_factory=dict)
    dense_embedding: list[float] | None = None
    sparse_embedding: dict[int, float] | None = None  # SPLADE形式

    @classmethod
    def create(
        cls,
        document_id: UUID,
        content: str,
        chunk_index: int,
        start_char: int,
        end_char: int,
        metadata: dict[str, Any] | None = None,
    ) -> "Chunk":
        return cls(
            id=uuid4(),
            document_id=document_id,
            content=content,
            chunk_index=chunk_index,
            start_char=start_char,
            end_char=end_char,
            metadata=metadata or {},
        )

    @property
    def char_count(self) -> int:
        return len(self.content)


@dataclass
class Query:
    """ユーザーからの検索クエリ"""
    id: UUID
    text: str
    top_k: int = 10
    rerank_top_n: int = 5
    alpha: float = 0.5  # Hybrid search weight (0=sparse, 1=dense)
    filters: dict[str, Any] | None = None
    include_metadata: bool = True
    user_id: str | None = None
    session_id: str | None = None

    @classmethod
    def create(
        cls,
        text: str,
        top_k: int = 10,
        rerank_top_n: int = 5,
        alpha: float = 0.5,
        filters: dict[str, Any] | None = None,
    ) -> "Query":
        return cls(
            id=uuid4(),
            text=text,
            top_k=top_k,
            rerank_top_n=rerank_top_n,
            alpha=alpha,
            filters=filters,
        )


@dataclass
class SearchResult:
    """検索結果"""
    chunk: Chunk
    score: float                    # Hybrid search スコア (Qdrant実装依存)
    rerank_score: float | None = None  # Rerank後のスコア (0.0-1.0)
    rank: int = 0                   # 順位

    @property
    def final_score(self) -> float:
        """最終スコア（rerank後があればそちら）"""
        return self.rerank_score if self.rerank_score is not None else self.score

    @property
    def display_score(self) -> float:
        """
        UI表示用の正規化スコア (0.0-1.0)

        Note:
        - Rerank スコアはすでに0-1の範囲
        - Hybrid search スコアはQdrantの実装により分布が異なる場合があり、
          必要に応じて正規化が必要
        """
        if self.rerank_score is not None:
            return self.rerank_score
        # Hybrid searchスコアは正規化が必要な場合あり
        return min(1.0, max(0.0, self.score))


@dataclass
class GenerationResult:
    """LLM生成結果"""
    id: UUID
    query: Query
    answer: str
    sources: list[SearchResult]
    usage: TokenUsage
    model: str
    latency_ms: float
    created_at: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def create(
        cls,
        query: Query,
        answer: str,
        sources: list[SearchResult],
        usage: TokenUsage,
        model: str,
        latency_ms: float,
    ) -> "GenerationResult":
        return cls(
            id=uuid4(),
            query=query,
            answer=answer,
            sources=sources,
            usage=usage,
            model=model,
            latency_ms=latency_ms,
        )
```

### 1.3 Exceptions

```python
# src/domain/exceptions.py

class DomainError(Exception):
    """ドメイン層の基底例外"""
    def __init__(self, message: str, code: str | None = None):
        self.message = message
        self.code = code or self.__class__.__name__
        super().__init__(self.message)


class DocumentNotFoundError(DomainError):
    """ドキュメントが見つからない"""
    def __init__(self, document_id: str):
        super().__init__(f"Document not found: {document_id}", "DOCUMENT_NOT_FOUND")


class DocumentProcessingError(DomainError):
    """ドキュメント処理エラー"""
    pass


class UnsupportedContentTypeError(DomainError):
    """サポートされていないコンテンツタイプ"""
    def __init__(self, content_type: str):
        super().__init__(
            f"Unsupported content type: {content_type}",
            "UNSUPPORTED_CONTENT_TYPE"
        )


class StorageError(DomainError):
    """ストレージエラー（BlobStorage）"""
    pass


class EmbeddingError(DomainError):
    """埋め込み生成エラー"""
    pass


class SearchError(DomainError):
    """検索エラー"""
    pass


class RerankError(DomainError):
    """Rerankエラー"""
    pass


class LLMError(DomainError):
    """LLM呼び出しエラー"""
    pass


class RateLimitError(LLMError):
    """レート制限エラー"""
    pass


class ConfigurationError(DomainError):
    """設定エラー"""
    pass
```

---

## 2. Domain Interfaces

```python
# src/domain/interfaces.py

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable, BinaryIO
from uuid import UUID

from .entities import Document, Chunk, Query, SearchResult, GenerationResult
from .value_objects import DocumentStatus


@runtime_checkable
class DocumentRepository(Protocol):
    """ドキュメントメタデータの永続化"""

    async def save(self, document: Document) -> Document:
        """ドキュメントを保存"""
        ...

    async def get_by_id(self, document_id: UUID) -> Document | None:
        """IDでドキュメントを取得"""
        ...

    async def list(
        self,
        status: DocumentStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Document]:
        """ドキュメント一覧を取得"""
        ...

    async def update(self, document: Document) -> Document:
        """ドキュメントを更新"""
        ...

    async def delete(self, document_id: UUID) -> bool:
        """ドキュメントを削除"""
        ...


@runtime_checkable
class BlobStorage(Protocol):
    """
    元ファイルの永続化（MinIO/S3）

    Note:
    - 元ファイルを保持することで、将来的なモデル変更時に再処理が可能
    - Phase 1: MinIO、本番: AWS S3
    """

    async def upload(
        self,
        file: BinaryIO,
        filename: str,
        content_type: str,
    ) -> str:
        """
        ファイルをアップロードし、パスを返す

        Args:
            file: アップロードするファイル
            filename: ファイル名
            content_type: MIMEタイプ

        Returns:
            BlobStorage上のパス（例: "documents/uuid/filename.pdf"）
        """
        ...

    async def download(self, path: str) -> BinaryIO:
        """ファイルをダウンロード"""
        ...

    async def delete(self, path: str) -> bool:
        """ファイルを削除"""
        ...

    async def exists(self, path: str) -> bool:
        """ファイルの存在確認"""
        ...


@runtime_checkable
class VectorStore(Protocol):
    """ベクトルストア操作"""

    async def upsert_chunks(self, chunks: list[Chunk]) -> None:
        """チャンクをベクトルストアに追加/更新"""
        ...

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Dense Vector検索"""
        ...

    async def hybrid_search(
        self,
        query_text: str,
        query_dense_embedding: list[float],
        query_sparse_embedding: dict[int, float],
        top_k: int = 10,
        alpha: float = 0.5,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Hybrid Search (Dense + Sparse)"""
        ...

    async def delete_by_document_id(self, document_id: UUID) -> int:
        """ドキュメントIDでチャンクを削除。削除件数を返す"""
        ...

    async def get_collection_stats(self) -> dict:
        """コレクション統計を取得"""
        ...


@runtime_checkable
class EmbeddingService(Protocol):
    """Dense埋め込みベクトル生成"""

    @property
    def dimension(self) -> int:
        """埋め込みベクトルの次元数"""
        ...

    async def embed_text(self, text: str) -> list[float]:
        """単一テキストの埋め込み生成"""
        ...

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """複数テキストの埋め込み生成（バッチ処理）"""
        ...


@runtime_checkable
class SparseEmbeddingService(Protocol):
    """Sparse Embedding生成（SPLADE等）"""

    async def embed_text(self, text: str) -> dict[int, float]:
        """単一テキストのSparse埋め込み生成"""
        ...

    async def embed_texts(self, texts: list[str]) -> list[dict[int, float]]:
        """複数テキストのSparse埋め込み生成"""
        ...


@runtime_checkable
class Reranker(Protocol):
    """Re-ranking サービス"""

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_n: int = 5,
    ) -> list[SearchResult]:
        """検索結果をリランク"""
        ...


@runtime_checkable
class LLMService(Protocol):
    """LLM呼び出し"""

    async def generate(
        self,
        prompt: str,
        context: list[str],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> GenerationResult:
        """回答を生成"""
        ...

    async def generate_stream(
        self,
        prompt: str,
        context: list[str],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        """ストリーミング生成（AsyncGenerator）"""
        ...


class DocumentParser(ABC):
    """ドキュメントパーサーの抽象基底クラス"""

    @abstractmethod
    async def parse(self, file_path: str) -> str:
        """ドキュメントをテキストに変換"""
        ...

    @abstractmethod
    def supports(self, content_type: str) -> bool:
        """指定されたコンテンツタイプをサポートするか"""
        ...


class ChunkingService(ABC):
    """チャンキングサービスの抽象基底クラス"""

    @abstractmethod
    def chunk(self, text: str, document_id: UUID, metadata: dict | None = None) -> list[Chunk]:
        """テキストをチャンクに分割"""
        ...
```

---

## 3. Configuration (Pydantic Settings)

```python
# config/settings.py

from functools import lru_cache
from typing import Literal
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """PostgreSQL設定"""
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
        return (
            f"postgresql+asyncpg://{self.user}:"
            f"{self.password.get_secret_value()}@"
            f"{self.host}:{self.port}/{self.database}"
        )


class QdrantSettings(BaseSettings):
    """Qdrant設定"""
    model_config = SettingsConfigDict(env_prefix="QDRANT_")

    host: str = "localhost"
    port: int = 6333
    grpc_port: int = 6334
    collection_name: str = "documents"
    use_grpc: bool = True
    api_key: SecretStr | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


class RedisSettings(BaseSettings):
    """Redis設定"""
    model_config = SettingsConfigDict(env_prefix="REDIS_")

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: SecretStr | None = None

    @property
    def url(self) -> str:
        if self.password:
            return f"redis://:{self.password.get_secret_value()}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


class MinIOSettings(BaseSettings):
    """MinIO (S3互換) 設定"""
    model_config = SettingsConfigDict(env_prefix="MINIO_")

    endpoint: str = "localhost:9000"
    access_key: str = "minioadmin"
    secret_key: SecretStr = SecretStr("minioadmin")
    bucket_name: str = "documents"
    secure: bool = False  # True for HTTPS

    @property
    def endpoint_url(self) -> str:
        protocol = "https" if self.secure else "http"
        return f"{protocol}://{self.endpoint}"


class OpenAISettings(BaseSettings):
    """OpenAI設定"""
    model_config = SettingsConfigDict(env_prefix="OPENAI_")

    api_key: SecretStr
    model: str = "gpt-5"
    fallback_model: str = "gpt-5-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    max_retries: int = 3
    timeout: float = 30.0


class CohereSettings(BaseSettings):
    """Cohere設定"""
    model_config = SettingsConfigDict(env_prefix="COHERE_")

    api_key: SecretStr
    rerank_model: str = "rerank-v3.5"
    max_retries: int = 3
    timeout: float = 30.0


class LangfuseSettings(BaseSettings):
    """Langfuse設定"""
    model_config = SettingsConfigDict(env_prefix="LANGFUSE_")

    public_key: str | None = None
    secret_key: SecretStr | None = None
    host: str = "http://localhost:3000"
    enabled: bool = True


class ChunkingSettings(BaseSettings):
    """Chunking設定"""
    model_config = SettingsConfigDict(env_prefix="CHUNKING_")

    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: list[str] = Field(default=["\n\n", "\n", ". ", " ", ""])


class SearchSettings(BaseSettings):
    """検索設定"""
    model_config = SettingsConfigDict(env_prefix="SEARCH_")

    default_top_k: int = 10
    rerank_top_n: int = 5
    hybrid_alpha: float = 0.5  # 0=sparse only, 1=dense only
    min_score_threshold: float = 0.0


class Settings(BaseSettings):
    """アプリケーション設定（ルート）"""
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
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}")
        return v.upper()


@lru_cache
def get_settings() -> Settings:
    """設定のシングルトンインスタンスを取得"""
    return Settings()
```

---

## 4. Ingestion Service with Compensating Transactions

```python
# src/application/services/ingestion_service.py

import logging
from typing import BinaryIO
from uuid import UUID

from src.domain.entities import Document, Chunk
from src.domain.interfaces import (
    DocumentRepository,
    BlobStorage,
    VectorStore,
    EmbeddingService,
    SparseEmbeddingService,
    DocumentParser,
    ChunkingService,
)
from src.domain.exceptions import DocumentProcessingError

logger = logging.getLogger(__name__)


class IngestionService:
    """
    ドキュメント取り込みサービス

    Saga Pattern (簡易版) を使用して、
    複数のストレージ間の整合性を保つ
    """

    def __init__(
        self,
        document_repo: DocumentRepository,
        blob_storage: BlobStorage,
        vector_store: VectorStore,
        dense_embedding: EmbeddingService,
        sparse_embedding: SparseEmbeddingService,
        parser: DocumentParser,
        chunker: ChunkingService,
    ):
        self.document_repo = document_repo
        self.blob_storage = blob_storage
        self.vector_store = vector_store
        self.dense_embedding = dense_embedding
        self.sparse_embedding = sparse_embedding
        self.parser = parser
        self.chunker = chunker

    async def ingest_document(
        self,
        document: Document,
        file: BinaryIO,
    ) -> Document:
        """
        ドキュメントを取り込む

        Flow:
        1. Document作成 (status=PENDING) → すでに作成済み
        2. ファイル保存 → MinIO
        3. Document.file_path 更新
        4. status=PROCESSING に更新
        5. パース → チャンキング
        6. Embedding生成
        7. Qdrant Upsert
        8. status=COMPLETED に更新

        失敗時の補償トランザクション:
        - Step 7 失敗 → status=FAILED + Qdrantから不完全なチャンク削除
        """
        try:
            # Step 2: ファイル保存
            file_path = await self.blob_storage.upload(
                file=file,
                filename=f"{document.id}/{document.filename}",
                content_type=document.content_type.value,
            )
            document.set_file_path(file_path)

            # Step 3-4: DB更新
            document.mark_processing()
            await self.document_repo.update(document)

            # Step 5: パース & チャンキング
            text = await self.parser.parse(file_path)
            chunks = self.chunker.chunk(
                text=text,
                document_id=document.id,
                metadata={"filename": document.filename},
            )

            # Step 6-7: Embedding生成 & Qdrant Upsert
            await self._embed_and_store(chunks)

            # Step 8: 完了
            document.mark_completed(len(chunks))
            await self.document_repo.update(document)

            logger.info(
                f"Document {document.id} ingested successfully: "
                f"{len(chunks)} chunks"
            )

        except Exception as e:
            logger.error(f"Failed to ingest document {document.id}: {e}")
            await self._handle_failure(document, e)
            raise DocumentProcessingError(str(e))

        return document

    async def _embed_and_store(self, chunks: list[Chunk]) -> None:
        """チャンクの埋め込み生成とベクトルストアへの保存"""
        texts = [chunk.content for chunk in chunks]

        # Dense & Sparse Embedding を並列生成
        dense_embeddings = await self.dense_embedding.embed_texts(texts)
        sparse_embeddings = await self.sparse_embedding.embed_texts(texts)

        # チャンクに埋め込みを設定
        for chunk, dense_emb, sparse_emb in zip(
            chunks, dense_embeddings, sparse_embeddings
        ):
            chunk.dense_embedding = dense_emb
            chunk.sparse_embedding = sparse_emb

        # ベクトルストアに保存
        await self.vector_store.upsert_chunks(chunks)

    async def _handle_failure(
        self,
        document: Document,
        error: Exception,
    ) -> None:
        """
        補償トランザクション: 失敗時のクリーンアップ
        """
        # Document を FAILED に更新
        document.mark_failed(str(error))
        try:
            await self.document_repo.update(document)
        except Exception as repo_error:
            logger.error(
                f"Failed to update document status: {repo_error}"
            )

        # Qdrant の不完全なチャンクを削除
        try:
            deleted_count = await self.vector_store.delete_by_document_id(
                document.id
            )
            if deleted_count > 0:
                logger.info(
                    f"Cleaned up {deleted_count} chunks for document {document.id}"
                )
        except Exception as cleanup_error:
            logger.warning(
                f"Failed to cleanup chunks for {document.id}: {cleanup_error}"
            )

    async def delete_document(self, document_id: UUID) -> bool:
        """
        ドキュメントとその関連データを削除

        削除順序:
        1. Qdrant のチャンク削除
        2. MinIO のファイル削除
        3. PostgreSQL のメタデータ削除
        """
        document = await self.document_repo.get_by_id(document_id)
        if not document:
            return False

        # Step 1: Qdrant のチャンク削除
        await self.vector_store.delete_by_document_id(document_id)

        # Step 2: MinIO のファイル削除
        if document.file_path:
            await self.blob_storage.delete(document.file_path)

        # Step 3: PostgreSQL のメタデータ削除
        return await self.document_repo.delete(document_id)
```

---

## 5. API Schemas (Pydantic)

```python
# src/presentation/schemas/documents.py

from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field

from src.domain.value_objects import DocumentStatus, ContentType


class DocumentUploadResponse(BaseModel):
    """ドキュメントアップロードレスポンス"""
    id: UUID
    filename: str
    content_type: ContentType
    size_bytes: int
    status: DocumentStatus
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentDetailResponse(BaseModel):
    """ドキュメント詳細レスポンス"""
    id: UUID
    filename: str
    content_type: ContentType
    size_bytes: int
    status: DocumentStatus
    metadata: dict
    created_at: datetime
    updated_at: datetime
    chunk_count: int
    file_path: str | None = None
    error_message: str | None = None

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    """ドキュメント一覧レスポンス"""
    items: list[DocumentDetailResponse]
    total: int
    limit: int
    offset: int


# src/presentation/schemas/query.py

class QueryRequest(BaseModel):
    """クエリリクエスト"""
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=100)
    rerank_top_n: int = Field(default=5, ge=1, le=20)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    filters: dict | None = None
    include_sources: bool = True
    stream: bool = False


class SourceResponse(BaseModel):
    """ソース情報レスポンス"""
    chunk_id: UUID
    document_id: UUID
    content: str
    score: float
    rerank_score: float | None = None
    display_score: float  # UI表示用の正規化スコア
    metadata: dict


class QueryResponse(BaseModel):
    """クエリレスポンス"""
    id: UUID
    query: str
    answer: str
    sources: list[SourceResponse]
    model: str
    usage: dict
    latency_ms: float
    created_at: datetime
```

---

## 6. Implementation Sequence

### Step 0: Storage基盤 (新規)
1. `docker-compose.yml` - MinIO追加
2. `config/settings.py` - MinIOSettings追加
3. `src/domain/interfaces.py` - BlobStorage Protocol追加
4. `src/infrastructure/storage/minio_storage.py` - MinIO実装

### Step 1: 基盤整備
1. `config/settings.py` - 全設定（完成版）
2. `src/domain/value_objects.py` - 値オブジェクト
3. `src/domain/entities.py` - エンティティ（file_path追加）
4. `src/domain/exceptions.py` - カスタム例外（StorageError追加）
5. `src/domain/interfaces.py` - インターフェース（BlobStorage含む）

### Step 2: Infrastructure
1. `src/infrastructure/storage/minio_storage.py`
2. `src/infrastructure/embeddings/openai_embedding.py`
3. `src/infrastructure/vectorstores/qdrant_vectorstore.py`
4. `src/infrastructure/repositories/postgres_document_repository.py`

### Step 3: Ingestion
1. Document Parser (Unstructured library)
2. Chunking Service (LangChain RecursiveCharacterTextSplitter)
3. Ingestion Service (補償トランザクション付き)

### Step 4: Search & Rerank
1. FastEmbed Sparse Embedding
2. Hybrid Search implementation
3. Cohere Reranker (Rerank 3.5)

### Step 5: Generation
1. OpenAI LLM Client (GPT-5/GPT-5 mini)
2. RAG Generation Service

### Step 6: API & Integration
1. FastAPI endpoints
2. Dependency injection setup
3. Error handling middleware
4. Integration tests

### Step 7: 検証
1. 日本語キーワード検索テスト（SPLADE性能検証）
2. エラーハンドリング・補償トランザクションテスト
3. End-to-End テスト

---

## 7. API Pricing & Cost Estimation (2025年11月時点)

### OpenAI

| Model | Input | Output | 用途 |
|-------|-------|--------|------|
| GPT-5 | $1.25/1M | $10/1M | Primary |
| GPT-5 mini | $0.25/1M | $2/1M | Fallback |
| GPT-5 nano | $0.05/1M | $0.40/1M | 軽量タスク |
| text-embedding-3-small | $0.02/1M | - | Embedding |

### Cohere

| Model | 単価 | 備考 |
|-------|------|------|
| Rerank 3.5 | $2.00/1K searches | 1 search = 1 query + max 100 docs |

### 1クエリあたりのコスト見積もり

```
想定:
- クエリ: 50 tokens
- 検索結果: 10件 × 500 tokens = 5,000 tokens
- Rerank: 10 documents (1 search unit)
- LLM Context: 5件 × 500 tokens = 2,500 tokens
- LLM Response: 500 tokens
```

| 処理 | コスト |
|------|--------|
| Query Embedding | ~$0.000001 |
| Cohere Rerank | ~$0.002 |
| LLM Input (GPT-5) | ~$0.003 |
| LLM Output (GPT-5) | ~$0.005 |
| **合計** | **~$0.010/query** |

### 月間コスト目安

| 利用量 | クエリ/月 | コスト |
|--------|----------|--------|
| 開発/テスト | 1,000 | ~$10 |
| 小規模運用 | 10,000 | ~$100 |
| 本番運用 | 100,000 | ~$1,000 |

---

## 8. AWS Bedrock Migration Strategy (将来対応)

### モデル置き換えマッピング

| コンポーネント | Phase 1 (OpenAI/Cohere) | AWS Bedrock 代替 |
|---------------|------------------------|------------------|
| LLM | GPT-5 | Claude Sonnet 4.5 ($3/$15 per 1M) |
| Fallback LLM | GPT-5 mini | Claude Haiku 4.5 ($1/$5 per 1M) |
| Embedding | text-embedding-3-small (1536次元) | Amazon Titan Text Embeddings v2 (1024次元) |
| Rerank | Cohere Rerank 3.5 | Cohere Rerank on Bedrock |

### 移行時の注意点

1. **Embedding次元数変更 (Critical)**
   - OpenAI: 1536次元 → Amazon Titan v2: 1024次元
   - Qdrantコレクション再作成 & 全ドキュメント再Ingestion必須

2. **プロンプトエンジニアリングの調整**
   - Claude向けに最適化（XMLタグ活用）

3. **リージョン可用性**
   - us-east-1 または us-west-2 推奨

### Provider切り替え実装例

```python
# src/presentation/api/dependencies.py

from config.settings import get_settings, Settings
from src.domain.interfaces import LLMService, EmbeddingService

def get_llm_service(settings: Settings = Depends(get_settings)) -> LLMService:
    if settings.provider == "bedrock":
        from src.infrastructure.llm.bedrock_llm import BedrockLLMService
        return BedrockLLMService(settings.bedrock)
    from src.infrastructure.llm.openai_llm import OpenAILLMService
    return OpenAILLMService(settings.openai)

def get_embedding_service(settings: Settings = Depends(get_settings)) -> EmbeddingService:
    if settings.provider == "bedrock":
        from src.infrastructure.embeddings.bedrock_embedding import BedrockEmbeddingService
        return BedrockEmbeddingService(settings.bedrock)
    from src.infrastructure.embeddings.openai_embedding import OpenAIEmbeddingService
    return OpenAIEmbeddingService(settings.openai)
```

---

## 9. Verification Checklist (Phase 1 完了前)

### 機能検証

- [ ] ドキュメントアップロード & Ingestion
- [ ] Hybrid Search (Dense + Sparse)
- [ ] Re-ranking
- [ ] RAG回答生成

### エラーハンドリング検証

- [ ] Ingestion中のエラーで補償トランザクションが発動
- [ ] Qdrant障害時のフォールバック
- [ ] API Rate Limit時のリトライ

### 日本語対応検証

- [ ] 日本語ドキュメントでのキーワード検索
  - 製品名、型番などの完全一致テスト
- [ ] SPLADE性能が不十分な場合の代替案検討
  - BM25 (Qdrant payload検索)
  - 日本語トークナイザ + TF-IDF

### パフォーマンス検証

- [ ] Ingestion: 10ページPDFで < 30秒
- [ ] Search: < 500ms (Rerank込み)
- [ ] Generation: < 3秒 (ストリーミング開始)
