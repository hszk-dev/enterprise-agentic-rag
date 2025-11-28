# Enterprise Agentic RAG Platform - Project Context

## Core Philosophy

**L5 Quality:** Reliability, Observability, and Scalability over quick hacks.

## Overview

L5（シニアエンジニア）レベルの品質を目指す、Advanced RAGおよびAgentic Workflowプラットフォーム。
Python (FastAPI), LangGraph, Qdrant, Langfuse を使用。

### Project Goals
- **Phase 1:** Core RAG with Hybrid Search (Vector + Keyword)
- **Phase 2:** Agentic Workflow with Query Decomposition
- **Phase 3:** Observability & Evaluation Integration
- **Phase 4:** Reliability & Frontend

## Model Context Protocol (MCP) - AI開発支援

このプロジェクトでは以下のMCPサーバーを活用し、AIによる正確なコーディングを実現する。

### 利用可能なMCPサーバー

| Server | 役割 | 主な用途 |
|--------|------|----------|
| **Serena** | LSP + 長期記憶 | コード解析、定義ジャンプ、参照検索。ハルシネーション防止 |
| **PostgreSQL** | DB管理 | スキーマ確認、クエリ検証 |
| **Qdrant** | Vector DB管理 | コレクション操作、検索テスト |

### MCP利用ガイドライン (Claude向け)

#### 1. Serena (`mcp__serena`)
- **コード編集前に必ず使用:** `find_symbol` や `get_hover_info` で既存の実装を確認
- **関数シグネチャを推測しない:** Serenaで検証してから使用
- **新しい依存関係:** 追加前に既存の使用パターンを確認

```
# 使用例
- get_codebase_symbols: プロジェクト全体の構造把握
- find_symbol: 特定のクラス/関数の検索
- get_hover_info: 型情報・ドキュメント取得
- find_references: 参照箇所の特定
```

#### 2. PostgreSQL (`mcp__postgres`)
- SQLクエリ作成前にスキーマを確認
- マイグレーション作成時にテーブル構造を検証

#### 3. Qdrant (`mcp__qdrant`)
- ベクトル検索のテスト・デバッグ
- コレクションの状態確認

## Development Workflow (AIネイティブ開発)

### ライフサイクル概要

```
Design → Context → Implementation → Verification → Review → Merge
```

### Phase 1: Design (設計 & 合意)

1. **Issue作成**: 機能要件と非機能要件を定義
2. **ADR作成**: 重要な技術選定は `docs/adr/YYYY-MM-DD-title.md` に記録

### Phase 2: Context Loading (コンテキスト同期)

1. **Feature Branch作成**: `git checkout -b feat/<ticket-id>-<desc>`
2. **Serenaインデックス更新**: `uvx --from git+https://github.com/oraios/serena serena project index`
3. **関連コード読み込み**: Serenaで実装対象周辺のコードを解析

### Phase 3: Implementation (TDD実装)

1. **Red**: テストケースを先に作成（`tests/unit/`）
2. **Green**: テストが通る最小限の実装
3. **Refactor**: 可読性向上、エラーハンドリング追加

### Phase 4: Verification (品質保証)

```bash
make format   # コードフォーマット
make lint     # リンターチェック
make test     # 全テスト実行
```

### Phase 5: Review (PR作成 & セルフレビュー)

1. **コミット**: Conventional Commits形式
2. **AIセルフレビュー**: セキュリティ・パフォーマンス観点でチェック
3. **PR作成**: テンプレートに従い、ADRへのリンクを含める

### Phase 6: Merge (完了)

1. **Human Review**: GitHub上でApprove
2. **Squash and Merge**: 履歴をクリーンに保つ
3. **Cleanup**: ブランチ削除、mainに戻る

### Workflow Rules (Claude向け)

1. **Design First:** 重要なアーキテクチャ変更は先にADRを作成
2. **Context Aware:** 実装前にSerenaで既存コードと影響範囲を分析
3. **Test Driven:** ビジネスロジック実装前にユニットテストを書く
4. **Verification:** コミット前に必ず `make test` と `make lint` を実行
5. **PR Standard:** PRテンプレートを使用、ADR/Issueへのリンクを含める

## Architecture Rules (Strict Enforcement)

### 1. Clean Architecture (Dependency Inversion)
```
presentation → application → domain ← infrastructure
```

- `src/domain/`: ドメインモデル、インターフェース定義。外部依存は**絶対禁止**。
- `src/application/`: ユースケース、サービスロジック。domain のみに依存。
- `src/infrastructure/`: 具体的な実装（Qdrant, OpenAI, Langfuse）。domain のインターフェースを実装。
- `src/presentation/`: API Endpoints (FastAPI)。application を呼び出す。

### 2. Typing (厳格)
- すべての関数引数と戻り値に**型ヒントを必須**とする。
- `Any` の使用は**原則禁止**。やむを得ない場合はコメントで理由を明記。
- Pydantic モデルを積極的に使用し、ランタイムバリデーションを行う。

### 3. Configuration
- 設定は `os.environ` を直接読み込まず、必ず `config/settings.py` の Pydantic `BaseSettings` を経由。
- 機密情報は `.env` に配置し、絶対にコミットしない。

### 4. Error Handling
- 例外を握りつぶさない（bare `except:` 禁止）。
- カスタム例外を `src/domain/exceptions.py` に定義。
- API層で `HTTPException` に変換してハンドリング。

### 5. Async First
- I/Oバウンドな処理（DB, LLM API, 外部サービス）はすべて `async/await` で実装。
- 同期的なブロッキング呼び出しは `run_in_executor` でラップ。

## Coding Style

### Formatter & Linter
- **Ruff** を使用（`ruff format` + `ruff check`）。
- Line length: 88 characters。
- Import sorting: Ruff の isort 互換機能を使用。

### Docstrings
- **Google Style Docstrings** をパブリックメソッドに記述。
- 内部ヘルパー関数にはオプション。

```python
def search_documents(query: str, top_k: int = 10) -> list[Document]:
    """Search for relevant documents using hybrid retrieval.

    Args:
        query: The search query string.
        top_k: Maximum number of documents to return.

    Returns:
        List of Document objects sorted by relevance.

    Raises:
        SearchError: If the search operation fails.
    """
```

### Naming Conventions
- Classes: `PascalCase`
- Functions/Variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private methods: `_leading_underscore`

### Testing
- **pytest** を使用。
- Unit Tests: `tests/unit/` - 外部依存は Mock。
- Integration Tests: `tests/integration/` - 実際の外部サービスを使用。
- E2E Tests: `tests/e2e/` - API エンドポイントのテスト。

### Logging
- **f-string を使用**: このプロジェクトでは `logger.info(f"...")` 形式を使用する
- Ruff の `G004` (flake8-logging-format) は有効化しない

```python
# OK: f-string
logger.info(f"Uploaded file: {filename} ({size} bytes)")

# NO: 遅延フォーマット（原則使用しない）
logger.info("Uploaded file: %s (%d bytes)", filename, size)
```

## Development Commands

```bash
# Setup
uv sync                    # Install dependencies

# Development
make run                   # Start dev server
make test                  # Run all tests
make test-unit             # Run unit tests only
make lint                  # Run linter
make format                # Format code

# Docker
make up                    # Start all services (Qdrant, Redis, Postgres)
make down                  # Stop all services
make logs                  # View service logs
```

## Tech Stack Details

### Core
- **Backend:** FastAPI, Pydantic V2, Uvicorn
- **Agent Framework:** LangGraph (ステート管理付きのワークフロー)
- **Vector DB:** Qdrant (Hybrid Search: Dense + Sparse Vectors)
- **Database:** PostgreSQL (メタデータ、会話履歴)
- **Cache/Queue:** Redis (Semantic Cache, Task Queue)

### LLM
- **Primary:** OpenAI API (GPT-4o, text-embedding-3-small)
- **Fallback:** Azure OpenAI / Anthropic Claude
- **Re-ranking:** Cohere Rerank API

### Observability
- **Tracing:** Langfuse (OSSセルフホスト可能)
- **Evaluation:** Ragas (RAG精度評価)

## File Structure Reference

```
enterprise-agentic-rag/
├── .github/                 # CI/CD, PR templates
├── config/                  # Pydantic settings
│   ├── __init__.py
│   └── settings.py
├── docs/
│   ├── adr/                 # Architecture Decision Records
│   └── design/              # Detailed design documents
├── src/
│   ├── domain/              # 外部依存なし
│   │   ├── __init__.py
│   │   ├── entities.py      # Document, Chunk, Query, SearchResult
│   │   ├── exceptions.py    # Custom exceptions
│   │   ├── interfaces.py    # Protocol classes
│   │   └── value_objects.py # DocumentStatus, TokenUsage, etc.
│   ├── application/         # ユースケース
│   │   ├── __init__.py
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── ingestion_service.py
│   │   │   ├── search_service.py
│   │   │   └── generation_service.py
│   │   └── use_cases/
│   │       ├── __init__.py
│   │       ├── upload_document.py
│   │       ├── search_documents.py
│   │       └── generate_answer.py
│   ├── infrastructure/      # 具体実装
│   │   ├── __init__.py
│   │   ├── storage/
│   │   │   └── minio_storage.py
│   │   ├── repositories/
│   │   │   └── postgres_document_repository.py
│   │   ├── vectorstores/
│   │   │   └── qdrant_vectorstore.py
│   │   ├── embeddings/
│   │   │   ├── openai_embedding.py
│   │   │   └── fastembed_sparse.py
│   │   ├── rerankers/
│   │   │   └── cohere_reranker.py
│   │   └── llm/
│   │       └── openai_llm.py
│   ├── presentation/        # API層
│   │   ├── __init__.py
│   │   ├── api/
│   │   │   └── v1/
│   │   │       ├── health.py
│   │   │       ├── documents.py
│   │   │       └── query.py
│   │   └── schemas/
│   │       ├── documents.py
│   │       └── query.py
│   └── main.py              # Entrypoint
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── CLAUDE.md                # This file
├── docker-compose.yml
├── Makefile
├── pyproject.toml
└── README.md
```

## Git Workflow

### Branch Naming
- `main`: Production-ready code
- `feat/<ticket-id>-<short-desc>`: Feature development
- `fix/<ticket-id>-<short-desc>`: Bug fixes
- `docs/<short-desc>`: Documentation updates
- `refactor/<short-desc>`: Code refactoring

### Commit Message Convention
Format: `type(scope): subject`

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting (no code change)
- `refactor`: Code restructuring
- `test`: Adding/updating tests
- `chore`: Maintenance tasks
- `perf`: Performance improvements

Examples:
```
feat(retrieval): implement hybrid search with qdrant
fix(api): handle empty context gracefully
docs(readme): add architecture diagram
test(search): add unit tests for re-ranking
```

### Commit/PR Rules (Claude向け)
- コミットメッセージやPRに「🤖 Generated with Claude Code」や「Co-Authored-By: Claude」などのAI生成署名を**含めない**
- 人間が書いたコミットと同じ形式で記述する

### Atomic Commit Guidelines (Claude向け)

コミットは**論理的な単位（Atomic）**で分割する。1つのコミットは1つの責務を持つ。

#### コミット分割の基準

| 分類 | 説明 | 例 |
|------|------|-----|
| **infra** | Docker, CI/CD, 環境設定 | `docker-compose.yml`, `.env.example` |
| **config** | アプリケーション設定 | `config/settings.py` |
| **domain** | ドメイン層（エンティティ、インターフェース、例外） | `src/domain/interfaces.py`, `src/domain/exceptions.py` |
| **実装** | Infrastructure層の具体実装 | `src/infrastructure/storage/minio_storage.py` |
| **test** | テストコード | `tests/unit/`, `tests/integration/` |
| **deps** | 依存関係 | `pyproject.toml`, `uv.lock` |

#### 実装例: Step 0 (MinIO Storage) のコミット履歴

```
1. feat(infra): add MinIO service to docker-compose
2. feat(config): add Pydantic settings with MinIO configuration
3. feat(domain): add BlobStorage interface and storage exceptions
4. feat(storage): implement MinIO blob storage
5. test(storage): add unit and integration tests for MinIO storage
6. chore(deps): add minio dependency for S3-compatible storage
```

#### 原則
1. **依存関係順にコミット:** 下位レイヤー（domain）→ 上位レイヤー（infrastructure）→ テスト → 依存関係
2. **1コミット1責務:** 設定と実装を混ぜない、テストは実装と別コミット
3. **レビュー容易性:** 各コミットが独立してレビュー可能であること

### Pre-commit Hook対応 (Claude向け)

このプロジェクトではpre-commit hooksが設定されており、コミット時に自動チェックが実行される。

#### 1. Secret Detection (`detect-secrets`)
- テストファイル内の`secret_key`などのキーワードがfalse positiveとして検出される
- 対策: テスト用のダミー値には`# pragma: allowlist secret`コメントを付与する

```python
# 例: テストファイル内
MinIOSettings(
    endpoint="localhost:9000",
    access_key="testuser",
    secret_key="testpass",  # pragma: allowlist secret
    bucket_name="test-bucket",
)
```

#### 2. `.secrets.baseline`の更新
- hookが`.secrets.baseline`を更新した場合、そのファイルも一緒にコミットする必要がある
- エラーメッセージ: `Please git add .secrets.baseline, thank you.`

## Important Constraints

1. **No Hallucination:** 不明な点はコードを読んで確認する。推測で実装しない。
2. **Incremental Changes:** 大きな変更は小さなステップに分割する。
3. **Test Before Commit:** 変更後は必ず関連テストを実行。
4. **Observe Boundaries:** 各層の境界を厳守。domain に infrastructure を import しない。

## Common Tasks Reference

### Adding a New Endpoint
1. `src/domain/entities.py` にドメインモデルを追加
2. `src/domain/interfaces.py` にリポジトリインターフェースを追加
3. `src/application/use_cases.py` にユースケースを追加
4. `src/presentation/schemas.py` にリクエスト/レスポンススキーマを追加
5. `src/presentation/api.py` にエンドポイントを追加
6. `tests/unit/` にユニットテストを追加

### Adding a New External Integration
1. `src/domain/interfaces.py` に抽象インターフェースを定義
2. `src/infrastructure/` に具体実装を追加
3. `config/settings.py` に設定項目を追加
4. `tests/integration/` に統合テストを追加
