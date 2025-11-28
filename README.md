# Enterprise Agentic RAG Platform

L5レベルの品質を目指す、Advanced RAGおよびAgentic Workflowプラットフォーム。

## Overview

このプロジェクトは、単純な「質問→検索→回答」のフローではなく、**推論エンジン（Agent）** が検索戦略を決定し、必要に応じてクエリを分解・実行するループ構造を持つRAGシステムです。

### Key Features

- **Hybrid Search**: Dense Vector (意味検索) + Sparse Vector (キーワード検索)
- **Re-ranking**: Cross-Encoderを用いた関連度順の並べ替え
- **Query Decomposition**: 複雑な質問を複数のサブタスクに分解
- **Semantic Caching**: 意味的に近い質問のキャッシュ
- **Observability**: 全ステップのトレース、レイテンシ、トークンコスト監視

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Orchestration Layer                          │
│  ┌──────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │  Router  │───▶│   Decomposition  │───▶│ Parallel Executor│  │
│  │  Agent   │    │      Agent       │    │                  │  │
│  └──────────┘    └──────────────────┘    └──────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Retrieval Layer                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │ Dense Search │    │ Sparse Search│    │    Re-ranking    │  │
│  │   (Vector)   │    │    (BM25)    │    │  (Cross-Encoder) │  │
│  └──────────────┘    └──────────────┘    └──────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Generation Layer                             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                   LLM (GPT-4o / Claude)                  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI, Python 3.11+ |
| Agent Framework | LangGraph |
| Vector DB | Qdrant |
| Database | PostgreSQL |
| Cache | Redis |
| LLM | OpenAI GPT-4o, Azure OpenAI |
| Observability | Langfuse, OpenTelemetry |
| Evaluation | Ragas |

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- [uv](https://github.com/astral-sh/uv) (recommended) or Poetry

### Setup

1. **Clone and setup environment**

```bash
# Clone the repository
git clone https://github.com/yourusername/enterprise-agentic-rag.git
cd enterprise-agentic-rag

# Copy environment file
cp .env.example .env
# Edit .env and add your API keys

# Install dependencies
uv sync --all-extras

# Install pre-commit hooks
uv run pre-commit install
```

2. **Setup MCP for Claude Code**

```bash
# Copy MCP configuration template
cp .mcp.json.example .mcp.json

# Edit .mcp.json and update PostgreSQL credentials if needed
# Default: postgresql://rag_user:rag_password@localhost:5432/rag_db
```

3. **Start infrastructure services**

```bash
# Start Qdrant, Redis, PostgreSQL, Langfuse
make up

# Verify services are running
docker-compose ps
```

4. **Run the development server**

```bash
make run
```

5. **Access the API**

- API Docs: http://localhost:8000/docs
- Langfuse UI: http://localhost:3000

## Development

### Common Commands

```bash
# Install dependencies
make install-dev

# Run development server
make run

# Run tests
make test           # All tests
make test-unit      # Unit tests only
make test-cov       # With coverage

# Code quality
make format         # Format code
make lint           # Run linter
make type-check     # Type checking

# Docker
make up             # Start services
make down           # Stop services
make logs           # View logs
```

### Project Structure

```
enterprise-agentic-rag/
├── .github/                 # CI/CD, PR templates
├── config/                  # Pydantic settings
├── docs/                    # ADRs, Design docs
├── src/
│   ├── domain/              # Entities, Interfaces (no external deps)
│   ├── application/         # Use cases, Services
│   ├── infrastructure/      # External integrations
│   ├── presentation/        # API endpoints
│   └── main.py              # Entry point
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── CLAUDE.md                # AI assistant context
├── docker-compose.yml
├── Makefile
└── pyproject.toml
```

## Implementation Roadmap

### Phase 1: Core RAG with Hybrid Search
- [x] Project structure setup
- [x] Configuration management
- [ ] Document ingestion pipeline
- [ ] Hybrid search implementation
- [ ] Re-ranking integration

### Phase 2: Agentic Workflow
- [ ] LangGraph state machine
- [ ] Router agent
- [ ] Decomposition agent
- [ ] Parallel execution engine

### Phase 3: Observability & Evaluation
- [ ] Langfuse integration
- [ ] Ragas evaluation pipeline
- [ ] LLM-as-a-Judge implementation

### Phase 4: Reliability & Frontend
- [ ] Semantic cache
- [ ] Fallback strategy
- [ ] Next.js chat UI
- [ ] Kubernetes deployment

## API Reference

### Health Check

```bash
# Basic health check
curl http://localhost:8000/api/v1/health

# Detailed health check
curl http://localhost:8000/api/v1/health/detailed
```

### Query

```bash
# Submit a query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?", "top_k": 5}'
```

### Document Upload

```bash
# Upload a document
curl -X POST http://localhost:8000/api/v1/documents \
  -F "file=@document.pdf"
```

## Contributing

1. Create a feature branch: `git checkout -b feat/your-feature`
2. Make changes following the coding standards in `CLAUDE.md`
3. Run tests: `make test`
4. Run linting: `make lint`
5. Create a PR using the template

## License

MIT License - see LICENSE file for details.
