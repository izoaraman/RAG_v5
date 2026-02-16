# RAG_v5 - Agentic RAG System

An agentic RAG (Retrieval Augmented Generation) system with LangGraph routing, Azure OpenAI, and PostgreSQL/pgvector.

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              STREAMLIT UI (app.py)                          │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │  Ask Mode   │  │ URL Crawler  │  │ Doc Upload   │  │ Session Memory  │  │
│  └──────┬──────┘  └──────┬───────┘  └──────┬───────┘  └─────────────────┘  │
└─────────┼────────────────┼─────────────────┼────────────────────────────────┘
          │                │                 │
          ▼                ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DOCUMENT INGESTION PIPELINE                          │
│                                                                              │
│  ┌──────────────────┐    ┌──────────────────┐    ┌────────────────────┐    │
│  │   RAGCrawler     │    │ DoclingExtractor │    │  EntityExtractor   │    │
│  │   (Crawl4AI)     │───▶│  (PDF/DOCX/PPTX) │───▶│  DocumentSummarizer│    │
│  └──────────────────┘    └────────┬─────────┘    └──────────┬─────────┘    │
│                                   │                         │               │
│                                   ▼                         │               │
│                          ┌──────────────────┐               │               │
│                          │ DoclingHybrid    │               │               │
│                          │ Chunker          │◀──────────────┘               │
│                          └────────┬─────────┘                               │
│                                   │                                          │
│                                   ▼                                          │
│                          ┌──────────────────┐                               │
│                          │ EmbeddingGenerator│                               │
│                          │ (Azure OpenAI)   │                               │
│                          └────────┬─────────┘                               │
│                                   │                                          │
│                                   ▼                                          │
│                          ┌──────────────────┐                               │
│                          │ PostgreSQL       │                               │
│                          │ + pgvector       │                               │
│                          └──────────────────┘                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                     AGENTIC RAG PIPELINE (LangGraph)                         │
│                                                                              │
│  User Query ──▶ ┌───────────────────────┐                                   │
│                 │   Query Classifier     │ (Heuristic + LLM)                │
│                 └───────────┬───────────┘                                   │
│                             │                                                │
│            ┌────────────────┼────────────────┐                              │
│            ▼                ▼                ▼                              │
│    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                      │
│    │ QuickFact    │ │  InDepth     │ │  LogicRAG    │                      │
│    │ Agent        │ │  Agent       │ │  Agent       │                      │
│    │ (top_k=3)    │ │ (top_k=10)   │ │ (Multi-hop)  │                      │
│    └──────┬───────┘ └──────┬───────┘ └──────┬───────┘                      │
│           │                │                │                               │
│           ▼                ▼                ▼                               │
│    ┌─────────────────────────────────────────────────────┐                 │
│    │              RETRIEVAL SYSTEM                        │                 │
│    │  ┌───────────────┐  ┌────────────────────────────┐  │                 │
│    │  │VectorRetriever│─▶│ Reranker (FlashRank/Cross- │  │                 │
│    │  │ (pgvector)    │  │ Encoder/Hybrid)            │  │                 │
│    │  └───────────────┘  └────────────────────────────┘  │                 │
│    └─────────────────────────────────────────────────────┘                 │
│           │                │                │                               │
│           └────────────────┼────────────────┘                               │
│                            ▼                                                │
│                 ┌───────────────────────┐                                   │
│                 │  Response Generator   │ (Azure OpenAI + Citations)        │
│                 └───────────┬───────────┘                                   │
│                             │                                                │
│                             ▼                                                │
│                 ┌───────────────────────┐                                   │
│                 │  ConversationMemory   │                                   │
│                 │  + RollingMemory      │                                   │
│                 └───────────────────────┘                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Query Classification & Routing

| Intent | Description | Agent | Retrieval Strategy |
|--------|-------------|-------|-------------------|
| **QUICK_FACT** | Simple factual questions | QuickFactAgent | top_k=3, direct answer |
| **IN_DEPTH** | Complex analysis/comparison | InDepthAgent | top_k=10 + reranking |
| **MULTI_HOP** | Multi-step reasoning | LogicRAGAgent | Query decomposition + rolling memory |

### Technology Stack

| Layer | Technologies |
|-------|--------------|
| **Frontend** | Streamlit 1.40+ |
| **Orchestration** | LangGraph 0.2+, LangChain 0.3+ |
| **LLM & Embeddings** | Azure OpenAI (GPT-4.1, text-embedding-3-small) |
| **Document Processing** | Docling 2.55+, Crawl4AI 0.4+ |
| **Vector Database** | PostgreSQL + pgvector, asyncpg |
| **Reranking** | FlashRank 0.2+, sentence-transformers 3.0+ |
| **Package Management** | UV (Astral) |

## Features

### Ingestion Pipeline
- **Web Crawling**: Extract website content using Crawl4AI
- **Document Processing**: PDF, DOCX, PPTX, XLSX with Docling + OCR
- **Semantic Chunking**: Structure-aware chunking with HybridChunker
- **Embedding Generation**: Azure OpenAI text-embedding-3-small
- **Vector Storage**: PostgreSQL with pgvector extension

### Agentic RAG Pipeline
- **LangGraph Router**: StateGraph-based agent orchestration
- **Query Classification**: LLM-based intent detection
- **QuickFactAgent**: Fast retrieval (top_k=3), direct answers
- **InDepthAgent**: Extended retrieval (top_k=10) with reranking
- **Conversation Memory**: Session-based context management
- **FlashRank Reranking**: Fast neural reranking for precision

## Installation

```bash
# Install dependencies with uv
uv sync

# Or with pip
pip install -e .
```

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Required environment variables:
- `DATABASE_URL`: PostgreSQL connection string
- `AZURE_OPENAI_API_KEY`: Azure OpenAI API key
- `AZURE_OPENAI_ENDPOINT`: Azure OpenAI endpoint URL
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`: Embedding model deployment name

## Usage

### Web Crawling

```bash
# Crawl a website
uv run rag-crawler --url https://example.com

# With options
uv run rag-crawler --url https://example.com --depth 3 --max-pages 100
```

### Document Ingestion

```bash
# Ingest documents from a folder
uv run rag-ingest --documents ./documents

# Ingest from crawler report
uv run rag-ingest --crawl-report ./output/crawl_report.json

# Force re-ingest existing documents
uv run rag-ingest --documents ./documents --force-update
```

## CLI Options

### rag-crawler
- `--url, -u`: Target website URL (required)
- `--depth, -d`: Maximum crawl depth (default: 3)
- `--max-pages, -m`: Maximum pages to crawl (default: 200)
- `--output-dir, -o`: Output directory (default: ./output)

### rag-ingest
- `--documents, -d`: Documents folder path (default: documents)
- `--crawl-report, -c`: Path to crawler JSON report
- `--force-update, -f`: Re-ingest existing documents
- `--clean`: Delete ALL existing data before ingestion
- `--chunk-size`: Chunk size for splitting (default: 1000)
- `--chunk-overlap`: Chunk overlap size (default: 200)
- `--verbose, -v`: Enable verbose logging

## Project Structure

```
src/rag_crawler/
├── main.py                    # CLI entry point
├── crawler.py                 # Web crawling (Crawl4AI)
├── config.py                  # Configuration settings
├── agents/                    # Agentic components
│   ├── query_classifier.py    # Intent classification
│   ├── quick_fact_agent.py    # Fast retrieval agent
│   ├── in_depth_agent.py      # Extended analysis agent
│   ├── logic_rag_agent.py     # Multi-hop reasoning agent
│   └── response_generator.py  # Response synthesis
├── router/                    # LangGraph router
│   ├── state.py               # RAGState definition
│   └── router_graph.py        # StateGraph workflow
├── retrieval/                 # Retrieval layer
│   ├── vector_retriever.py    # pgvector search
│   ├── reranker.py            # FlashRank/CrossEncoder
│   ├── memory.py              # Conversation memory
│   └── rolling_memory.py      # Multi-hop context
├── extractors/                # Document extraction
│   ├── docling_extractor.py   # Multi-format extraction
│   └── document.py            # Document models
├── ingestion/                 # Ingestion pipeline
│   ├── ingest.py              # Pipeline orchestrator
│   ├── chunker.py             # Structure-aware chunking
│   ├── embedder.py            # Azure OpenAI embeddings
│   ├── entity_extractor.py    # Entity/topic extraction
│   └── summarizer.py          # LLM summarization
├── output/                    # Output formatters
│   └── markdown.py            # Markdown generation
└── utils/                     # Utilities
    ├── db_utils.py            # PostgreSQL operations
    ├── models.py              # Pydantic data models
    ├── providers.py           # LLM client factory
    └── azure_providers.py     # Azure configuration
```

## Requirements

- Python 3.11+
- PostgreSQL with PGVector extension
- Azure OpenAI API access

## Quick Start

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### 2. Configure Environment

```bash
# Copy template
cp .env.example .env

# Edit .env with your credentials:
# - AZURE_OPENAI_API_KEY
# - AZURE_OPENAI_ENDPOINT
# - AZURE_OPENAI_DEPLOYMENT_NAME (e.g., gpt-4o)
# - AZURE_OPENAI_EMBEDDING_DEPLOYMENT (e.g., text-embedding-3-small)
# - DATABASE_URL (PostgreSQL connection string)
```

### 3. Set Up Database

```sql
-- Connect to PostgreSQL and create database
CREATE DATABASE rag_database;

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
```

### 4. Ingest Documents

```bash
# Ingest from a folder
uv run rag-ingest --documents ./documents

# Or crawl a website
uv run rag-crawler --url https://example.com
```

### 5. Run the Chat Interface

```bash
# Start Streamlit app
uv run streamlit run app.py

# Or with python directly
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## Testing the Pipeline

### Test Query Classification

```python
import asyncio
from rag_crawler.agents.query_classifier import classify_query

async def test():
    # Simple query -> should route to QuickFactAgent
    result = await classify_query("What is ACCC?")
    print(f"Intent: {result.intent}, Confidence: {result.confidence}")
    
    # Complex query -> should route to InDepthAgent
    result = await classify_query("Explain the regulatory framework and enforcement mechanisms")
    print(f"Intent: {result.intent}, Confidence: {result.confidence}")

asyncio.run(test())
```

### Test Full Pipeline

```python
from rag_crawler.router.router_graph import RAGRouter

router = RAGRouter()
result = router.process_query_sync("What does ACCC do?")

print(f"Response: {result['response']}")
print(f"Agent Used: {result['agent_used']}")
print(f"Sources: {len(result['sources'])}")
```

### Verify Azure Configuration

```python
from rag_crawler.utils.azure_providers import validate_azure_configuration

status = validate_azure_configuration()
print(status)
```

## Full Project Structure

```
RAG_v5/
├── app.py                         # Streamlit chat interface
├── pyproject.toml                 # Project configuration (UV)
├── .env.example                   # Environment template
├── data/                          # Runtime data
│   ├── documents/                 # Document storage
│   └── new_uploads/               # Uploaded files
├── src/rag_crawler/
│   ├── main.py                    # CLI entry point
│   ├── crawler.py                 # Web crawling (Crawl4AI)
│   ├── config.py                  # Settings & configuration
│   ├── agents/                    # Agentic components
│   │   ├── query_classifier.py    # Heuristic + LLM classification
│   │   ├── quick_fact_agent.py    # Fast retrieval (top_k=3)
│   │   ├── in_depth_agent.py      # Extended analysis (top_k=10)
│   │   ├── logic_rag_agent.py     # Multi-hop reasoning
│   │   └── response_generator.py  # LLM response synthesis
│   ├── router/                    # LangGraph router
│   │   ├── state.py               # RAGState TypedDict
│   │   └── router_graph.py        # StateGraph workflow
│   ├── retrieval/                 # Retrieval layer
│   │   ├── vector_retriever.py    # pgvector similarity search
│   │   ├── reranker.py            # FlashRank/CrossEncoder/Hybrid
│   │   ├── memory.py              # Conversation & session memory
│   │   └── rolling_memory.py      # Multi-hop context compression
│   ├── ingestion/                 # Ingestion pipeline
│   │   ├── ingest.py              # Pipeline orchestrator
│   │   ├── chunker.py             # DoclingHybridChunker
│   │   ├── embedder.py            # Azure OpenAI embeddings
│   │   ├── entity_extractor.py    # Regex entity extraction
│   │   └── summarizer.py          # LLM document summarization
│   ├── extractors/                # Document extraction
│   │   ├── docling_extractor.py   # PDF/DOCX/PPTX/XLSX/HTML/OCR
│   │   └── document.py            # Document data models
│   ├── output/                    # Output formatters
│   │   └── markdown.py            # Crawl report generation
│   └── utils/                     # Utilities
│       ├── db_utils.py            # PostgreSQL/pgvector operations
│       ├── models.py              # Pydantic configuration models
│       ├── providers.py           # LLM client factory (singleton)
│       └── azure_providers.py     # Azure OpenAI validation
```

## Troubleshooting

### "AZURE_OPENAI_API_KEY is not set"
- Ensure `.env` file exists in the project root
- Check that the key is correct and not expired
- Verify the key has access to the specified deployments

### "DATABASE_URL is not configured"
- Create a PostgreSQL database with pgvector extension
- Format: `postgresql://user:pass@host:5432/dbname?sslmode=require`

### "Router initialization failed"
- Check all environment variables are set
- Verify Azure OpenAI endpoint is accessible
- Ensure database is running and accessible

## License

MIT
