# ADR-001: Phase 1 Core RAG Architecture

## Status

**Accepted**

## Date

2025-11-28

## Context

Enterprise Agentic RAG Platformの第一フェーズとして、堅牢なRAG基盤を構築する必要がある。
このフェーズでは以下を実現する：

1. ドキュメントのIngestion（取り込み）パイプライン
2. Hybrid Search（Dense Vector + Sparse Vector）
3. Re-ranking（Cross-Encoder による精度向上）

単純なベクトル検索では不十分な理由：
- キーワードマッチが重要な専門用語（製品名、型番など）を見逃す
- 意味的に近いが文脈的に不適切な結果を返す可能性がある

## Decision Drivers

- **精度優先**: エンタープライズ用途では回答の正確性が最重要
- **スケーラビリティ**: 大量のドキュメントに対応できる設計
- **拡張性**: Phase 2以降のAgentic機能を容易に追加できる構造
- **テスタビリティ**: 各コンポーネントを独立してテスト可能
- **可観測性**: 各ステップのメトリクスを収集可能
- **データ再処理可能性**: 元ファイルを保持し、将来のモデル変更に対応

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      Ingestion Pipeline                          │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────┐ │
│  │  Upload  │──▶│   Store  │──▶│  Parse   │──▶│   Chunk &    │ │
│  │  (API)   │   │  (MinIO) │   │(Unstruct)│   │    Embed     │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────────┘ │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                       Query Pipeline                             │
│  ┌──────────┐   ┌──────────────────────────┐   ┌─────────────┐ │
│  │  Query   │──▶│      Hybrid Search       │──▶│  Re-rank    │ │
│  │  (API)   │   │  (Dense + Sparse/BM25)   │   │  (Cohere)   │ │
│  └──────────┘   └──────────────────────────┘   └─────────────┘ │
│                                                       │          │
│                                                       ▼          │
│                                              ┌─────────────┐    │
│                                              │  Generate   │    │
│                                              │   (LLM)     │    │
│                                              └─────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Domain Model Design

### Core Entities

```python
# src/domain/entities.py

@dataclass
class Document:
    """アップロードされたドキュメント"""
    id: UUID
    filename: str
    content_type: ContentType
    size_bytes: int
    status: DocumentStatus  # pending, processing, completed, failed
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    error_message: str | None = None
    chunk_count: int = 0
    file_path: str | None = None  # BlobStorage上のパス

@dataclass
class Chunk:
    """ドキュメントを分割したチャンク"""
    id: UUID
    document_id: UUID
    content: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: dict[str, Any]
    dense_embedding: list[float] | None = None
    sparse_embedding: dict[int, float] | None = None  # SPLADE形式

@dataclass
class Query:
    """ユーザーからの検索クエリ"""
    id: UUID
    text: str
    top_k: int = 10
    rerank_top_n: int = 5
    alpha: float = 0.5  # Hybrid search weight (0=sparse, 1=dense)
    filters: dict[str, Any] | None = None

@dataclass
class SearchResult:
    """検索結果"""
    chunk: Chunk
    score: float                    # Hybrid search スコア
    rerank_score: float | None = None  # Rerank後のスコア
    rank: int = 0

    @property
    def display_score(self) -> float:
        """UI表示用の正規化スコア (0.0-1.0)"""
        if self.rerank_score is not None:
            return self.rerank_score
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
```

### Domain Interfaces

```python
# src/domain/interfaces.py

class DocumentRepository(Protocol):
    """ドキュメントメタデータの永続化"""
    async def save(self, document: Document) -> Document: ...
    async def get_by_id(self, document_id: UUID) -> Document | None: ...
    async def update(self, document: Document) -> Document: ...
    async def delete(self, document_id: UUID) -> bool: ...

class BlobStorage(Protocol):
    """元ファイルの永続化（MinIO/S3）"""
    async def upload(self, file: BinaryIO, filename: str, content_type: str) -> str: ...
    async def download(self, path: str) -> BinaryIO: ...
    async def delete(self, path: str) -> bool: ...
    async def exists(self, path: str) -> bool: ...

class VectorStore(Protocol):
    """ベクトルストア操作"""
    async def upsert_chunks(self, chunks: list[Chunk]) -> None: ...
    async def hybrid_search(self, query_text: str, query_dense_embedding: list[float],
                            query_sparse_embedding: dict[int, float], top_k: int,
                            alpha: float = 0.5) -> list[SearchResult]: ...
    async def delete_by_document_id(self, document_id: UUID) -> int: ...

class EmbeddingService(Protocol):
    """Dense埋め込みベクトル生成"""
    async def embed_text(self, text: str) -> list[float]: ...
    async def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

class SparseEmbeddingService(Protocol):
    """Sparse埋め込み生成（SPLADE）"""
    async def embed_text(self, text: str) -> dict[int, float]: ...
    async def embed_texts(self, texts: list[str]) -> list[dict[int, float]]: ...

class Reranker(Protocol):
    """Re-ranking サービス"""
    async def rerank(self, query: str, results: list[SearchResult], top_n: int) -> list[SearchResult]: ...

class LLMService(Protocol):
    """LLM呼び出し"""
    async def generate(self, prompt: str, context: list[str], temperature: float = 0.0) -> GenerationResult: ...
```

## Considered Options

### Blob Storage

#### Option 1: Local File System
- **概要:** ローカルディスクにファイルを保存
- **メリット:** セットアップ不要
- **デメリット:** スケールしない、本番環境と異なる

#### Option 2: MinIO (S3互換)
- **概要:** MinIOをDocker Composeで起動しS3互換ストレージを使用
- **メリット:** S3互換、本番と同じコード、マルチノード対応
- **デメリット:** 追加のコンテナが必要

**選択:** Option 2 (MinIO)
- 本番環境との差異を最小化
- 将来的なAWS S3への移行が容易

### Chunking Strategy

#### Option 1: Fixed-Size Chunking
- **概要:** 固定文字数（例：1000文字）で分割
- **メリット:** 実装がシンプル、予測可能なチャンクサイズ
- **デメリット:** 文の途中で分割される可能性

#### Option 2: Semantic Chunking
- **概要:** 意味的な境界（段落、セクション）で分割
- **メリット:** 意味的に完結したチャンク、検索精度向上
- **デメリット:** チャンクサイズのばらつき、実装が複雑

#### Option 3: Recursive Character Splitting (LangChain方式)
- **概要:** 階層的な区切り文字（\n\n → \n → .）で再帰的に分割
- **メリット:** バランスの取れた分割、実装の成熟度
- **デメリット:** ドキュメント構造に依存

**選択:** Option 3 (Recursive Character Splitting)
- LangChainの実績あるアルゴリズムを使用
- チャンクサイズ: 1000文字、オーバーラップ: 200文字

### Hybrid Search Implementation

#### Option 1: Qdrant Native Hybrid Search
- **概要:** QdrantのSparse Vector機能を使用
- **メリット:** 単一のストアで完結、クエリがシンプル
- **デメリット:** Sparse Vector生成にSPLADEモデルが必要

#### Option 2: Separate BM25 Index
- **概要:** Elasticsearchを別途用意しBM25検索
- **メリット:** 成熟したBM25実装、豊富な機能
- **デメリット:** インフラ複雑化、結果のマージが必要

#### Option 3: Qdrant + FastEmbed SPLADE
- **概要:** FastEmbedでSPLADEベクトルを生成しQdrantに格納
- **メリット:** 軽量、Qdrantのみで完結
- **デメリット:** SPLADEモデルのロードが必要

**選択:** Option 3 (Qdrant + FastEmbed SPLADE)
- 単一ストアでHybrid Searchを実現
- FastEmbedで軽量にSparse Vector生成

### Re-ranking Model

#### Option 1: Cohere Rerank API
- **概要:** Cohere社のRerank APIを使用
- **メリット:** 高精度、メンテナンス不要、多言語対応
- **デメリット:** API費用、外部依存

#### Option 2: Self-hosted Cross-Encoder
- **概要:** HuggingFaceのCross-Encoderをセルフホスト
- **メリット:** 費用なし、低レイテンシ
- **デメリット:** GPU必要、モデル管理が必要

**選択:** Option 1 (Cohere Rerank API)
- Phase 1では開発速度を優先
- 将来的にセルフホストへの移行も可能な設計

## Decision

### 1. Storage
- **Blob Storage:** MinIO (S3互換) - 元ファイルの永続化
- **Metadata:** PostgreSQL - ドキュメントメタデータ
- **Vector:** Qdrant - チャンクとベクトル

### 2. Domain層の設計
上記のエンティティとインターフェースを採用（BlobStorage追加）

### 3. Chunking
- RecursiveCharacterTextSplitter（LangChain）
- chunk_size: 1000, chunk_overlap: 200

### 4. Embedding
- Dense: OpenAI text-embedding-3-small (1536次元)
- Sparse: FastEmbed SPLADE (Qdrant Sparse Vector)

### 5. Vector Store
- Qdrant with Hybrid Search (Dense + Sparse)
- alpha パラメータで重み調整可能

### 6. Re-ranking
- Cohere Rerank 3.5
- top_n: 5 (上位5件に絞り込み)

### 7. LLM
- OpenAI GPT-5 (primary)
- GPT-5 mini (fallback)

## File Structure

```
src/
├── domain/
│   ├── __init__.py
│   ├── entities.py        # Document, Chunk, Query, SearchResult
│   ├── exceptions.py      # DomainError, DocumentNotFoundError, etc.
│   ├── interfaces.py      # Protocol classes (BlobStorage追加)
│   └── value_objects.py   # DocumentStatus, TokenUsage, etc.
├── application/
│   ├── __init__.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── ingestion_service.py   # ドキュメント取り込み（補償トランザクション付き）
│   │   ├── search_service.py      # Hybrid Search + Rerank
│   │   └── generation_service.py  # RAG生成
│   └── use_cases/
│       ├── __init__.py
│       ├── upload_document.py
│       ├── search_documents.py
│       └── generate_answer.py
├── infrastructure/
│   ├── __init__.py
│   ├── storage/
│   │   ├── __init__.py
│   │   └── minio_storage.py       # MinIO BlobStorage実装
│   ├── repositories/
│   │   ├── __init__.py
│   │   └── postgres_document_repository.py
│   ├── vectorstores/
│   │   ├── __init__.py
│   │   └── qdrant_vectorstore.py
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── openai_embedding.py
│   │   └── fastembed_sparse.py
│   ├── rerankers/
│   │   ├── __init__.py
│   │   └── cohere_reranker.py
│   └── llm/
│       ├── __init__.py
│       └── openai_llm.py
├── presentation/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   ├── health.py
│   │   │   ├── documents.py
│   │   │   └── query.py
│   │   └── dependencies.py
│   └── schemas/
│       ├── __init__.py
│       ├── documents.py
│       └── query.py
└── main.py

config/
├── __init__.py
└── settings.py   # Pydantic BaseSettings (MinIO設定追加)
```

## Error Handling: Saga Pattern (簡易版)

### Ingestion Flow with Compensating Transactions

```
1. Document作成 (status=PENDING) → PostgreSQL
2. ファイル保存 → MinIO
3. Document.file_path 更新
4. status=PROCESSING に更新
5. パース → チャンキング
6. Embedding生成
7. Qdrant Upsert
8. status=COMPLETED に更新

失敗時の補償トランザクション:
- Step 7 失敗 → status=FAILED + Qdrantから不完全なチャンク削除
- Step 2 失敗 → Document削除
```

```python
class IngestionService:
    async def ingest_document(self, document: Document, file: BinaryIO) -> Document:
        try:
            # Step 2: ファイル保存
            file_path = await self.blob_storage.upload(file, document.filename, ...)
            document.metadata["file_path"] = file_path

            # Step 4: status更新
            document.mark_processing()
            await self.document_repo.update(document)

            # Step 5-7: パース → チャンク → Embedding → Qdrant
            text = await self.parser.parse(file_path)
            chunks = self.chunker.chunk(text, document.id)
            await self._embed_and_store(chunks)

            # Step 8: 完了
            document.mark_completed(len(chunks))
            await self.document_repo.update(document)

        except Exception as e:
            await self._handle_failure(document, e)
            raise

        return document

    async def _handle_failure(self, document: Document, error: Exception) -> None:
        """補償トランザクション"""
        document.mark_failed(str(error))
        await self.document_repo.update(document)

        # Qdrantの不完全なチャンクを削除
        try:
            await self.vector_store.delete_by_document_id(document.id)
        except Exception:
            logger.warning(f"Failed to cleanup chunks for {document.id}")
```

## Consequences

### Positive
- Clean Architectureにより各層が独立してテスト可能
- Protocolベースの設計により実装の差し替えが容易（AWS Bedrock移行対応）
- Hybrid Searchにより精度と再現率のバランスを確保
- 元ファイル保持により、将来のモデル変更時に再処理可能
- 補償トランザクションによりデータ不整合を防止

### Negative
- 初期実装の複雑さが増す
- 複数の外部APIへの依存（OpenAI, Cohere）
- MinIOコンテナの追加によるリソース消費

### Risks

#### 1. SPLADE日本語性能 (Medium)
- **リスク:** FastEmbedのデフォルトSPLADEモデル（prithvida/Splade_PP_en_v1等）は英語特化
- **影響:** 日本語ドキュメントでキーワードヒット率が低下する可能性
- **対策:**
  - Phase 1完了前に日本語キーワード検索テストを実施
  - 代替案: BM25 (Qdrant payload検索) または日本語トークナイザ+TF-IDF

#### 2. 分散トランザクション (High)
- **リスク:** PostgreSQL→Qdrant間で部分的失敗が発生
- **影響:** DBは「完了」だが検索に出ない不整合状態
- **対策:** Saga Pattern（補償トランザクション）で対応（上記実装）

#### 3. 外部API依存 (Medium)
- **リスク:** OpenAI/Cohere APIの可用性・料金変更
- **影響:** サービス停止、コスト増
- **対策:** Fallback戦略、将来的なAWS Bedrock移行設計

## Implementation Order

### Step 0: Storage基盤 (新規)
- [ ] MinIO設定追加 (docker-compose.yml, settings.py)
- [ ] BlobStorage インターフェース定義
- [ ] MinIOStorage 実装

### Step 1: 基盤整備
- [ ] config/settings.py (MinIO, OpenAI, Cohere設定)
- [ ] src/domain/entities.py
- [ ] src/domain/exceptions.py
- [ ] src/domain/interfaces.py (BlobStorage含む)
- [ ] src/domain/value_objects.py

### Step 2: Infrastructure実装
- [ ] MinIO BlobStorage
- [ ] OpenAI Embedding Client
- [ ] Qdrant VectorStore (Dense only first)
- [ ] PostgreSQL Document Repository

### Step 3: Ingestion Pipeline
- [ ] Document Parser (Unstructured)
- [ ] Chunking Service
- [ ] Ingestion Service (補償トランザクション付き)

### Step 4: Search Pipeline
- [ ] FastEmbed Sparse Embedding
- [ ] Hybrid Search (Dense + Sparse)
- [ ] Cohere Reranker
- [ ] Search Service

### Step 5: Generation
- [ ] OpenAI LLM Client
- [ ] Generation Service
- [ ] RAG Use Case

### Step 6: API Endpoints
- [ ] Health endpoints
- [ ] Document upload endpoint
- [ ] Query endpoint

### Step 7: 検証
- [ ] 日本語キーワード検索テスト
- [ ] エラーハンドリングテスト

## AWS Bedrock Migration Strategy (将来対応)

Clean ArchitectureのProtocolベースインターフェースにより、Infrastructure層の実装を差し替えるだけでAWS Bedrockに移行可能。

### モデル置き換えマッピング

| コンポーネント | Phase 1 (OpenAI/Cohere) | AWS Bedrock 代替 |
|---------------|------------------------|------------------|
| LLM | GPT-5 | Claude Sonnet 4.5 |
| Fallback LLM | GPT-5 mini | Claude Haiku 4.5 |
| Embedding | text-embedding-3-small (1536次元) | Amazon Titan Text Embeddings v2 (1024次元) |
| Rerank | Cohere Rerank 3.5 | Cohere Rerank on Bedrock |

### 移行時の注意点
1. **Embedding次元数変更:** OpenAI 1536 → Titan 1024（再Ingestion必須）
2. **プロンプト調整:** Claude向けに最適化（XMLタグ活用）
3. **リージョン:** us-east-1 または us-west-2 推奨

## Related

- Issue: Phase 1 Implementation
- CLAUDE.md: Project Guidelines

## Notes

### API Pricing

> **Last verified:** 2025-11-28 15:23 JST
>
> ⚠️ 価格は変動する可能性があります。最新情報は各サービスの公式ページを確認してください。

#### OpenAI

| Model | Input | Output | 用途 |
|-------|-------|--------|------|
| GPT-5 | $1.25/1M | $10/1M | Primary |
| GPT-5 mini | $0.25/1M | $2/1M | Fallback |
| GPT-5 nano | $0.05/1M | $0.40/1M | 軽量タスク |
| text-embedding-3-small | $0.02/1M | - | Embedding |

#### Cohere

| Model | 単価 |
|-------|------|
| Rerank 3.5 | $2.00/1K searches |

#### AWS Bedrock (将来参考)

| Model | Input | Output |
|-------|-------|--------|
| Claude Sonnet 4.5 | $3/1M | $15/1M |
| Claude Haiku 4.5 | $1/1M | $5/1M |
| Titan Embeddings v2 | $0.02/1M | - |

### コスト見積もり (1クエリあたり)

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

### Qdrant Collection Schema

```python
{
    "collection_name": "documents",
    "vectors": {
        "dense": {
            "size": 1536,
            "distance": "Cosine"
        }
    },
    "sparse_vectors": {
        "sparse": {}
    },
    "payload_schema": {
        "document_id": "uuid",
        "chunk_index": "integer",
        "content": "text",
        "metadata": "json"
    }
}
```
