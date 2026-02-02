# RAG Data Ingestion Crawler

A Python-based RAG (Retrieval Augmented Generation) pipeline for crawling websites and ingesting documents into a PostgreSQL vector database with PGVector.

## Features

- **Web Crawling**: Extract website content using Crawl4AI into RAG-optimized markdown
- **Document Processing**: Convert PDF, DOCX, PPTX, XLSX files using Docling with OCR support
- **Audio Transcription**: Transcribe audio files (MP3, WAV, M4A, FLAC) using Whisper ASR
- **Semantic Chunking**: Structure-aware document chunking with Docling's HybridChunker
- **Embedding Generation**: Generate embeddings using Azure OpenAI
- **Vector Storage**: Store documents and embeddings in PostgreSQL with PGVector
- **Deduplication**: Skip existing documents, only add new content

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
├── main.py              # CLI entry point for crawler
├── crawler.py           # Web crawling logic
├── config.py            # Configuration settings
├── extractors/          # Document extraction
│   ├── docling_extractor.py
│   └── document.py
├── ingestion/           # Ingestion pipeline
│   ├── ingest.py        # Main ingestion orchestrator
│   ├── chunker.py       # Document chunking
│   └── embedder.py      # Embedding generation
├── output/              # Output formatters
│   └── markdown.py
└── utils/               # Utilities
    ├── db_utils.py      # PostgreSQL operations
    ├── models.py        # Data models
    └── providers.py     # Azure OpenAI client
```

## Requirements

- Python 3.11+
- PostgreSQL with PGVector extension
- Azure OpenAI API access

## License

MIT
