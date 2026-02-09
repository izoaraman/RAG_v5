"""
Docling HybridChunker implementation for intelligent document splitting.

Uses Docling's HybridChunker which combines:
- Token-aware chunking (uses actual tokenizer)
- Document structure preservation (headings, sections, tables)
- Semantic boundary respect (paragraphs, code blocks)
- Contextualized output (chunks include heading hierarchy)
"""

import logging
import re
import hashlib
import importlib
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class ChunkingConfig:
    """Configuration for chunking."""

    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_chunk_size: int = 2000
    min_chunk_size: int = 100
    use_semantic_splitting: bool = True
    max_tokens: int = 512

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        if self.min_chunk_size <= 0:
            raise ValueError("Minimum chunk size must be positive")


@dataclass
class DocumentChunk:
    """Represents a document chunk with optional embedding."""

    content: str
    index: int
    start_char: int
    end_char: int
    metadata: dict[str, Any] = field(default_factory=dict)
    token_count: int | None = None
    embedding: list[float] | None = None
    content_hash: str | None = None

    def __post_init__(self) -> None:
        """Calculate token count if not provided."""
        if self.token_count is None:
            self.token_count = len(self.content) // 4
        if self.content_hash is None and self.content:
            self.content_hash = hashlib.sha256(self.content.strip().lower().encode()).hexdigest()


class DoclingHybridChunker:
    """
    Docling HybridChunker wrapper for intelligent document splitting.

    Uses Docling's HybridChunker for structure-aware, token-precise chunking.
    """

    def __init__(self, config: ChunkingConfig):
        """
        Initialize chunker.

        Args:
            config: Chunking configuration.
        """
        self.config = config
        self._tokenizer = None
        self._chunker = None

    @property
    def tokenizer(self):
        """Lazy-load tokenizer."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            model_id = "sentence-transformers/all-MiniLM-L6-v2"
            logger.info(f"Initializing tokenizer: {model_id}")
            self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        return self._tokenizer

    @property
    def chunker(self):
        """Lazy-load HybridChunker."""
        if self._chunker is None:
            chunking_module = importlib.import_module("docling.chunking")
            hybrid_chunker_cls = getattr(chunking_module, "HybridChunker")

            self._chunker = hybrid_chunker_cls(
                tokenizer=self.tokenizer,
                max_tokens=self.config.max_tokens,
                merge_peers=True,
            )
            logger.info(f"HybridChunker initialized (max_tokens={self.config.max_tokens})")
        return self._chunker

    async def chunk_document(
        self,
        content: str,
        title: str,
        source: str,
        metadata: dict[str, Any] | None = None,
        docling_doc=None,
    ) -> list[DocumentChunk]:
        """
        Chunk a document using Docling's HybridChunker.

        Args:
            content: Document content (markdown format).
            title: Document title.
            source: Document source.
            metadata: Additional metadata.
            docling_doc: Pre-converted DoclingDocument for efficiency.

        Returns:
            List of document chunks with contextualized content.
        """
        if not content.strip():
            return []

        base_metadata = {
            "title": title,
            "source": source,
            "chunk_method": "hybrid",
            **(metadata or {}),
        }

        # If no DoclingDocument provided, use fallback
        if docling_doc is None:
            logger.warning("No DoclingDocument provided, using simple chunking fallback")
            return self._simple_fallback_chunk(content, base_metadata)

        try:
            # Use HybridChunker to chunk the DoclingDocument
            chunk_iter = self.chunker.chunk(dl_doc=docling_doc)
            chunks = list(chunk_iter)

            document_chunks = []
            current_pos = 0

            for i, chunk in enumerate(chunks):
                # Get contextualized text (includes heading hierarchy)
                contextualized_text = self.chunker.contextualize(chunk=chunk)
                chunk_content = contextualized_text.strip()
                token_count = len(self.tokenizer.encode(contextualized_text))
                chunk_hash = hashlib.sha256(chunk_content.lower().encode()).hexdigest()

                chunk_metadata = {
                    **base_metadata,
                    "total_chunks": len(chunks),
                    "token_count": token_count,
                    "has_context": True,
                    "content_hash": chunk_hash,
                }

                start_char = current_pos
                end_char = start_char + len(contextualized_text)

                document_chunks.append(
                    DocumentChunk(
                        content=chunk_content,
                        index=i,
                        start_char=start_char,
                        end_char=end_char,
                        metadata=chunk_metadata,
                        token_count=token_count,
                        content_hash=chunk_hash,
                    )
                )

                current_pos = end_char

            logger.info(f"Created {len(document_chunks)} chunks using HybridChunker")
            return document_chunks

        except Exception as e:
            logger.error(f"HybridChunker failed: {e}, falling back to simple chunking")
            return self._simple_fallback_chunk(content, base_metadata)

    def _simple_fallback_chunk(
        self,
        content: str,
        base_metadata: dict[str, Any],
    ) -> list[DocumentChunk]:
        """Simple fallback chunking when HybridChunker unavailable."""
        chunks = []
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        start = 0
        chunk_index = 0

        while start < len(content):
            end = start + chunk_size

            if end >= len(content):
                chunk_text = content[start:]
            else:
                # Try to end at sentence boundary
                chunk_end = end
                for i in range(end, max(start + self.config.min_chunk_size, end - 200), -1):
                    if i < len(content) and content[i] in ".!?\n":
                        chunk_end = i + 1
                        break
                chunk_text = content[start:chunk_end]
                end = chunk_end

            if chunk_text.strip():
                chunk_content = chunk_text.strip()
                token_count = len(self.tokenizer.encode(chunk_text))
                chunk_hash = hashlib.sha256(chunk_content.lower().encode()).hexdigest()

                chunks.append(
                    DocumentChunk(
                        content=chunk_content,
                        index=chunk_index,
                        start_char=start,
                        end_char=end,
                        metadata={
                            **base_metadata,
                            "chunk_method": "simple_fallback",
                            "total_chunks": -1,
                            "content_hash": chunk_hash,
                        },
                        token_count=token_count,
                        content_hash=chunk_hash,
                    )
                )

                chunk_index += 1

            start = end - overlap

        # Update total chunks
        for chunk in chunks:
            chunk.metadata["total_chunks"] = len(chunks)

        logger.info(f"Created {len(chunks)} chunks using simple fallback")
        return chunks


class SimpleChunker:
    """Simple non-semantic chunker for faster processing."""

    def __init__(self, config: ChunkingConfig):
        """Initialize simple chunker."""
        self.config = config

    async def chunk_document(
        self,
        content: str,
        title: str,
        source: str,
        metadata: dict[str, Any] | None = None,
        **kwargs,
    ) -> list[DocumentChunk]:
        """
        Chunk document using simple paragraph-based rules.

        Args:
            content: Document content.
            title: Document title.
            source: Document source.
            metadata: Additional metadata.

        Returns:
            List of document chunks.
        """
        if not content.strip():
            return []

        base_metadata = {
            "title": title,
            "source": source,
            "chunk_method": "simple",
            **(metadata or {}),
        }

        # Split on double newlines (paragraphs)
        paragraphs = re.split(r"\n\s*\n", content)

        chunks = []
        current_chunk = ""
        current_pos = 0
        chunk_index = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            potential_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph

            if len(potential_chunk) <= self.config.chunk_size:
                current_chunk = potential_chunk
            else:
                if current_chunk:
                    chunks.append(
                        self._create_chunk(
                            current_chunk,
                            chunk_index,
                            current_pos,
                            current_pos + len(current_chunk),
                            base_metadata.copy(),
                        )
                    )

                    current_pos += len(current_chunk)
                    chunk_index += 1

                current_chunk = paragraph

        # Add final chunk
        if current_chunk:
            chunks.append(
                self._create_chunk(
                    current_chunk,
                    chunk_index,
                    current_pos,
                    current_pos + len(current_chunk),
                    base_metadata.copy(),
                )
            )

        # Update total chunks
        for chunk in chunks:
            chunk.metadata["total_chunks"] = len(chunks)

        return chunks

    def _create_chunk(
        self,
        content: str,
        index: int,
        start_pos: int,
        end_pos: int,
        metadata: dict[str, Any],
    ) -> DocumentChunk:
        """Create a DocumentChunk object."""
        chunk_content = content.strip()
        chunk_hash = hashlib.sha256(chunk_content.lower().encode()).hexdigest()
        return DocumentChunk(
            content=chunk_content,
            index=index,
            start_char=start_pos,
            end_char=end_pos,
            metadata={**metadata, "content_hash": chunk_hash},
            content_hash=chunk_hash,
        )


def create_chunker(config: ChunkingConfig) -> DoclingHybridChunker | SimpleChunker:
    """
    Create appropriate chunker based on configuration.

    Args:
        config: Chunking configuration.

    Returns:
        Chunker instance.
    """
    if config.use_semantic_splitting:
        return DoclingHybridChunker(config)
    else:
        return SimpleChunker(config)
