# Enterprise Agentic RAG Platform

## Overview

This platform provides advanced RAG capabilities with agentic workflows.

## Features

### Hybrid Search

- **Dense Vectors**: OpenAI text-embedding-3-small (1536 dimensions)
- **Sparse Vectors**: FastEmbed SPLADE for keyword matching
- **Fusion**: Reciprocal Rank Fusion with configurable alpha

### Document Processing

Supported formats:
- PDF documents
- Microsoft Word (DOCX)
- Plain text (TXT)
- Markdown (MD)
- HTML pages

### Agentic Workflows

The platform supports multi-step reasoning:

1. Query decomposition into sub-questions
2. Parallel retrieval for each sub-question
3. Evidence synthesis and ranking
4. Final answer generation with citations

## Quick Start

```python
from src.application.services import IngestionService

# Upload a document
document = await ingestion_service.ingest_document(doc, file)

# Search for relevant chunks
results = await search_service.search(query)
```

## Configuration

All settings are managed through environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| QDRANT_HOST | Qdrant server host | localhost |
| OPENAI_API_KEY | OpenAI API key | required |
| MINIO_ENDPOINT | MinIO endpoint | localhost:9000 |

## License

MIT License - See LICENSE file for details.
