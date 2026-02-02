"""
Main ingestion pipeline for processing documents into vector DB.

Supports:
- Local files (PDF, DOCX, PPTX, XLSX, MD, TXT, HTML)
- Audio files (MP3, WAV, M4A, FLAC) via Whisper ASR
- Crawled content from RAG crawler output
"""

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

from rag_crawler.ingestion.chunker import ChunkingConfig, DocumentChunk, create_chunker
from rag_crawler.ingestion.embedder import create_embedder
from rag_crawler.utils.db_utils import (
    close_database,
    db_pool,
    delete_document,
    document_exists,
    get_document_by_source,
    get_existing_documents,
    initialize_database,
)
from rag_crawler.utils.models import IngestionConfig, IngestionResult

load_dotenv()

logger = logging.getLogger(__name__)


class DocumentIngestionPipeline:
    """Pipeline for ingesting documents into vector DB."""

    def __init__(
        self,
        config: IngestionConfig,
        documents_folder: str = "documents",
        clean_before_ingest: bool = False,
        force_update: bool = False,
    ):
        """
        Initialize ingestion pipeline.

        Args:
            config: Ingestion configuration.
            documents_folder: Folder containing documents.
            clean_before_ingest: Whether to clean ALL existing data (use with caution).
            force_update: Whether to re-ingest documents that already exist.
        """
        self.config = config
        self.documents_folder = documents_folder
        self.clean_before_ingest = clean_before_ingest
        self.force_update = force_update

        # Initialize components
        self.chunker_config = ChunkingConfig(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            max_chunk_size=config.max_chunk_size,
            use_semantic_splitting=config.use_semantic_chunking,
            max_tokens=config.max_tokens,
        )

        self.chunker = create_chunker(self.chunker_config)
        self.embedder = create_embedder()

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize database connections."""
        if self._initialized:
            return

        logger.info("Initializing ingestion pipeline...")
        await initialize_database()
        self._initialized = True
        logger.info("Ingestion pipeline initialized")

    async def close(self) -> None:
        """Close database connections."""
        if self._initialized:
            await close_database()
            self._initialized = False

    async def ingest_documents(
        self,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[IngestionResult]:
        """
        Ingest all documents from the documents folder.

        Only adds new documents. Existing documents are skipped unless force_update=True.

        Args:
            progress_callback: Optional callback for progress updates.

        Returns:
            List of ingestion results.
        """
        if not self._initialized:
            await self.initialize()

        if self.clean_before_ingest:
            logger.warning("clean_before_ingest=True: Deleting ALL existing data!")
            await self._clean_databases()

        document_files = self._find_document_files()

        if not document_files:
            logger.warning(f"No supported document files found in {self.documents_folder}")
            return []

        logger.info(f"Found {len(document_files)} document files to process")

        # Get existing documents for deduplication
        existing_docs = await get_existing_documents()
        logger.info(f"Found {len(existing_docs)} existing documents in database")

        results = []
        skipped = 0
        new_docs = 0

        for i, file_path in enumerate(document_files):
            try:
                document_source = os.path.relpath(file_path, self.documents_folder)

                # Check if document already exists
                if document_source in existing_docs and not self.force_update:
                    existing = existing_docs[document_source]
                    logger.info(f"[SKIP] Already exists: {existing['title']}")
                    results.append(
                        IngestionResult(
                            document_id=existing["id"],
                            title=existing["title"],
                            chunks_created=0,
                            processing_time_ms=0,
                            errors=["Already exists (skipped)"],
                        )
                    )
                    skipped += 1

                    if progress_callback:
                        progress_callback(i + 1, len(document_files))
                    continue

                # If force_update, delete existing document first
                if document_source in existing_docs and self.force_update:
                    existing = existing_docs[document_source]
                    logger.info(f"[UPDATE] Replacing existing: {existing['title']}")
                    await delete_document(existing["id"])

                logger.info(f"Processing file {i+1}/{len(document_files)}: {file_path}")
                result = await self._ingest_single_document(file_path)
                results.append(result)
                new_docs += 1

                if progress_callback:
                    progress_callback(i + 1, len(document_files))

            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                results.append(
                    IngestionResult(
                        document_id="",
                        title=os.path.basename(file_path),
                        chunks_created=0,
                        processing_time_ms=0,
                        errors=[str(e)],
                    )
                )

        total_chunks = sum(r.chunks_created for r in results)
        total_errors = sum(len(r.errors) for r in results if "Already exists" not in str(r.errors))

        logger.info(
            f"Ingestion complete: {new_docs} new documents, "
            f"{skipped} skipped, {total_chunks} chunks, {total_errors} errors"
        )

        return results

    async def ingest_crawl_report(
        self,
        report_path: str,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[IngestionResult]:
        """
        Ingest documents from a RAG crawler report.

        Only adds new pages. Existing pages are skipped unless force_update=True.

        Args:
            report_path: Path to crawler JSON report.
            progress_callback: Optional callback for progress updates.

        Returns:
            List of ingestion results.
        """
        if not self._initialized:
            await self.initialize()

        if self.clean_before_ingest:
            logger.warning("clean_before_ingest=True: Deleting ALL existing data!")
            await self._clean_databases()

        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        results_data = report.get("results", [])

        if not results_data:
            logger.warning("No results found in crawl report")
            return []

        logger.info(f"Ingesting {len(results_data)} pages from crawl report")

        # Get existing documents for deduplication
        existing_docs = await get_existing_documents()
        logger.info(f"Found {len(existing_docs)} existing documents in database")

        results = []
        skipped = 0
        new_docs = 0

        for i, page_data in enumerate(results_data):
            try:
                url = page_data.get("url", "")
                title = page_data.get("title", "Untitled")

                # Check if document already exists (by URL as source)
                if url in existing_docs and not self.force_update:
                    existing = existing_docs[url]
                    logger.info(f"[SKIP] Already exists: {existing['title']}")
                    results.append(
                        IngestionResult(
                            document_id=existing["id"],
                            title=existing["title"],
                            chunks_created=0,
                            processing_time_ms=0,
                            errors=["Already exists (skipped)"],
                        )
                    )
                    skipped += 1

                    if progress_callback:
                        progress_callback(i + 1, len(results_data))
                    continue

                # If force_update, delete existing document first
                if url in existing_docs and self.force_update:
                    existing = existing_docs[url]
                    logger.info(f"[UPDATE] Replacing existing: {existing['title']}")
                    await delete_document(existing["id"])

                result = await self._ingest_crawl_page(page_data)
                results.append(result)
                new_docs += 1

                if progress_callback:
                    progress_callback(i + 1, len(results_data))

            except Exception as e:
                logger.error(f"Failed to process page {page_data.get('url', 'unknown')}: {e}")
                results.append(
                    IngestionResult(
                        document_id="",
                        title=page_data.get("title", "Unknown"),
                        chunks_created=0,
                        processing_time_ms=0,
                        errors=[str(e)],
                    )
                )

        total_chunks = sum(r.chunks_created for r in results)
        total_errors = sum(len(r.errors) for r in results if "Already exists" not in str(r.errors))

        logger.info(
            f"Ingestion complete: {new_docs} new documents, "
            f"{skipped} skipped, {total_chunks} chunks, {total_errors} errors"
        )

        return results

    async def _ingest_single_document(self, file_path: str) -> IngestionResult:
        """Ingest a single document file."""
        start_time = datetime.now()

        document_content, docling_doc = self._read_document(file_path)
        document_title = self._extract_title(document_content, file_path)
        document_source = os.path.relpath(file_path, self.documents_folder)
        document_metadata = self._extract_document_metadata(document_content, file_path)

        logger.info(f"Processing document: {document_title}")

        # Chunk the document
        chunks = await self.chunker.chunk_document(
            content=document_content,
            title=document_title,
            source=document_source,
            metadata=document_metadata,
            docling_doc=docling_doc,
        )

        if not chunks:
            logger.warning(f"No chunks created for {document_title}")
            return IngestionResult(
                document_id="",
                title=document_title,
                chunks_created=0,
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                errors=["No chunks created"],
            )

        logger.info(f"Created {len(chunks)} chunks")

        # Generate embeddings
        embedded_chunks = await self.embedder.embed_chunks(chunks)
        logger.info(f"Generated embeddings for {len(embedded_chunks)} chunks")

        # Save to PostgreSQL
        document_id = await self._save_to_postgres(
            document_title,
            document_source,
            document_content,
            embedded_chunks,
            document_metadata,
        )

        logger.info(f"Saved document to PostgreSQL with ID: {document_id}")

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return IngestionResult(
            document_id=document_id,
            title=document_title,
            chunks_created=len(chunks),
            processing_time_ms=processing_time,
            errors=[],
        )

    async def _ingest_crawl_page(self, page_data: dict[str, Any]) -> IngestionResult:
        """Ingest a single page from crawl report."""
        start_time = datetime.now()

        url = page_data.get("url", "")
        title = page_data.get("title", "Untitled")
        content = page_data.get("markdown", "")
        content_type = page_data.get("content_type", "page")

        metadata = {
            "url": url,
            "content_type": content_type,
            "retrieved_at": page_data.get("retrieved_at", ""),
            "extraction_quality": page_data.get("extraction_quality", "unknown"),
            "links_found": page_data.get("links_found", 0),
        }

        logger.info(f"Processing crawled page: {title}")

        # Chunk the content (no docling_doc for crawled content)
        chunks = await self.chunker.chunk_document(
            content=content,
            title=title,
            source=url,
            metadata=metadata,
            docling_doc=None,
        )

        if not chunks:
            return IngestionResult(
                document_id="",
                title=title,
                chunks_created=0,
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                errors=["No chunks created"],
            )

        # Generate embeddings
        embedded_chunks = await self.embedder.embed_chunks(chunks)

        # Save to database
        document_id = await self._save_to_postgres(
            title,
            url,
            content,
            embedded_chunks,
            metadata,
        )

        return IngestionResult(
            document_id=document_id,
            title=title,
            chunks_created=len(chunks),
            processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            errors=[],
        )

    def _find_document_files(self) -> list[str]:
        """Find all supported document files."""
        if not os.path.exists(self.documents_folder):
            logger.error(f"Documents folder not found: {self.documents_folder}")
            return []

        patterns = [
            "**/*.md",
            "**/*.markdown",
            "**/*.txt",
            "**/*.pdf",
            "**/*.docx",
            "**/*.doc",
            "**/*.pptx",
            "**/*.ppt",
            "**/*.xlsx",
            "**/*.xls",
            "**/*.html",
            "**/*.htm",
            "**/*.mp3",
            "**/*.wav",
            "**/*.m4a",
            "**/*.flac",
        ]

        files = []
        folder_path = Path(self.documents_folder)

        for pattern in patterns:
            files.extend(str(f) for f in folder_path.glob(pattern))

        return sorted(files)

    def _read_document(self, file_path: str) -> tuple[str, Any]:
        """
        Read document content from file.

        Returns:
            Tuple of (markdown_content, docling_document).
        """
        file_ext = os.path.splitext(file_path)[1].lower()

        # Audio formats - transcribe with Whisper ASR
        audio_formats = [".mp3", ".wav", ".m4a", ".flac"]
        if file_ext in audio_formats:
            content = self._transcribe_audio(file_path)
            return (content, None)

        # Docling-supported formats
        docling_formats = [
            ".pdf",
            ".docx",
            ".doc",
            ".pptx",
            ".ppt",
            ".xlsx",
            ".xls",
            ".html",
            ".htm",
        ]

        if file_ext in docling_formats:
            try:
                from docling.document_converter import DocumentConverter

                logger.info(f"Converting {file_ext} file using Docling: {os.path.basename(file_path)}")

                converter = DocumentConverter()
                result = converter.convert(file_path)

                markdown_content = result.document.export_to_markdown()
                logger.info(f"Successfully converted {os.path.basename(file_path)} to markdown")

                return (markdown_content, result.document)

            except Exception as e:
                logger.error(f"Failed to convert {file_path} with Docling: {e}")
                logger.warning(f"Falling back to raw text extraction for {file_path}")
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        return (f.read(), None)
                except Exception:
                    return (f"[Error: Could not read file {os.path.basename(file_path)}]", None)

        # Text-based formats
        else:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return (f.read(), None)
            except UnicodeDecodeError:
                with open(file_path, "r", encoding="latin-1") as f:
                    return (f.read(), None)

    def _transcribe_audio(self, file_path: str) -> str:
        """Transcribe audio file using Whisper ASR via Docling."""
        try:
            from docling.datamodel import asr_model_specs
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import AsrPipelineOptions
            from docling.document_converter import AudioFormatOption, DocumentConverter
            from docling.pipeline.asr_pipeline import AsrPipeline

            audio_path = Path(file_path).resolve()
            logger.info(f"Transcribing audio file using Whisper Turbo: {audio_path.name}")

            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            pipeline_options = AsrPipelineOptions()
            pipeline_options.asr_options = asr_model_specs.WHISPER_TURBO

            converter = DocumentConverter(
                format_options={
                    InputFormat.AUDIO: AudioFormatOption(
                        pipeline_cls=AsrPipeline,
                        pipeline_options=pipeline_options,
                    )
                }
            )

            result = converter.convert(audio_path)
            markdown_content = result.document.export_to_markdown()
            logger.info(f"Successfully transcribed {os.path.basename(file_path)}")
            return markdown_content

        except Exception as e:
            logger.error(f"Failed to transcribe {file_path} with Whisper ASR: {e}")
            return f"[Error: Could not transcribe audio file {os.path.basename(file_path)}]"

    def _extract_title(self, content: str, file_path: str) -> str:
        """Extract title from document content or filename."""
        lines = content.split("\n")
        for line in lines[:10]:
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()

        return os.path.splitext(os.path.basename(file_path))[0]

    def _extract_document_metadata(self, content: str, file_path: str) -> dict[str, Any]:
        """Extract metadata from document content."""
        metadata: dict[str, Any] = {
            "file_path": file_path,
            "file_size": len(content),
            "ingestion_date": datetime.now().isoformat(),
        }

        # Try to extract YAML frontmatter
        if content.startswith("---"):
            try:
                import yaml

                end_marker = content.find("\n---\n", 4)
                if end_marker != -1:
                    frontmatter = content[4:end_marker]
                    yaml_metadata = yaml.safe_load(frontmatter)
                    if isinstance(yaml_metadata, dict):
                        metadata.update(yaml_metadata)
            except ImportError:
                logger.warning("PyYAML not installed, skipping frontmatter extraction")
            except Exception as e:
                logger.warning(f"Failed to parse frontmatter: {e}")

        lines = content.split("\n")
        metadata["line_count"] = len(lines)
        metadata["word_count"] = len(content.split())

        return metadata

    async def _save_to_postgres(
        self,
        title: str,
        source: str,
        content: str,
        chunks: list[DocumentChunk],
        metadata: dict[str, Any],
    ) -> str:
        """Save document and chunks to PostgreSQL."""
        async with db_pool.acquire() as conn:
            async with conn.transaction():
                # Insert document
                document_result = await conn.fetchrow(
                    """
                    INSERT INTO documents (title, source, content, metadata)
                    VALUES ($1, $2, $3, $4)
                    RETURNING id::text
                    """,
                    title,
                    source,
                    content,
                    json.dumps(metadata),
                )

                document_id = document_result["id"]

                # Insert chunks
                for chunk in chunks:
                    embedding_data = None
                    if chunk.embedding:
                        embedding_data = "[" + ",".join(map(str, chunk.embedding)) + "]"

                    await conn.execute(
                        """
                        INSERT INTO chunks (document_id, content, embedding, chunk_index, metadata, token_count)
                        VALUES ($1::uuid, $2, $3::vector, $4, $5, $6)
                        """,
                        document_id,
                        chunk.content,
                        embedding_data,
                        chunk.index,
                        json.dumps(chunk.metadata),
                        chunk.token_count,
                    )

                return document_id

    async def _clean_databases(self) -> None:
        """Clean existing data from databases."""
        logger.warning("Cleaning existing data from databases...")

        async with db_pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("DELETE FROM chunks")
                await conn.execute("DELETE FROM documents")

        logger.info("Cleaned PostgreSQL database")


def main() -> None:
    """Main function for running ingestion."""
    parser = argparse.ArgumentParser(description="Ingest documents into vector DB")
    parser.add_argument("--documents", "-d", default="documents", help="Documents folder path")
    parser.add_argument(
        "--crawl-report",
        "-c",
        help="Path to crawler JSON report (alternative to documents folder)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete ALL existing data before ingestion (use with caution)",
    )
    parser.add_argument(
        "--force-update",
        "-f",
        action="store_true",
        help="Re-ingest documents that already exist (replaces existing)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size for splitting documents",
    )
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap size")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens per chunk")
    parser.add_argument("--no-semantic", action="store_true", help="Disable semantic chunking")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    config = IngestionConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_tokens=args.max_tokens,
        use_semantic_chunking=not args.no_semantic,
    )

    pipeline = DocumentIngestionPipeline(
        config=config,
        documents_folder=args.documents,
        clean_before_ingest=args.clean,
        force_update=args.force_update,
    )

    def progress_callback(current: int, total: int) -> None:
        print(f"Progress: {current}/{total} documents processed")

    async def run_ingestion() -> None:
        try:
            start_time = datetime.now()

            if args.crawl_report:
                results = await pipeline.ingest_crawl_report(
                    args.crawl_report,
                    progress_callback,
                )
            else:
                results = await pipeline.ingest_documents(progress_callback)

            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()

            # Calculate statistics
            new_docs = [r for r in results if r.chunks_created > 0]
            skipped_docs = [r for r in results if "Already exists" in str(r.errors)]
            error_docs = [
                r for r in results
                if r.errors and "Already exists" not in str(r.errors)
            ]

            print("\n" + "=" * 50)
            print("INGESTION SUMMARY")
            print("=" * 50)
            print(f"Documents processed: {len(results)}")
            print(f"  - New/Updated: {len(new_docs)}")
            print(f"  - Skipped (existing): {len(skipped_docs)}")
            print(f"  - Errors: {len(error_docs)}")
            print(f"Total chunks created: {sum(r.chunks_created for r in results)}")
            print(f"Total processing time: {total_time:.2f} seconds")
            print()

            for result in results:
                if "Already exists" in str(result.errors):
                    print(f"[SKIP] {result.title}")
                elif result.errors:
                    print(f"[ERR]  {result.title}")
                    for error in result.errors:
                        print(f"       Error: {error}")
                else:
                    print(f"[OK]   {result.title}: {result.chunks_created} chunks")

        except KeyboardInterrupt:
            print("\nIngestion interrupted by user")
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            raise
        finally:
            await pipeline.close()

    asyncio.run(run_ingestion())


if __name__ == "__main__":
    main()
