"""Document embedding generation using Azure OpenAI."""

import asyncio
import hashlib
import logging
from datetime import datetime
from typing import Any, Callable

from dotenv import load_dotenv
from openai import APIError, RateLimitError

from rag_crawler.ingestion.chunker import DocumentChunk
from rag_crawler.utils.providers import get_embedding_client, get_embedding_model

load_dotenv()

logger = logging.getLogger(__name__)

# Lazy-loaded client
_embedding_client = None
_embedding_model: str | None = None


def _get_client():
    """Get or create embedding client (lazy initialization)."""
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = get_embedding_client()
    return _embedding_client


def _get_model() -> str:
    """Get embedding model name."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = get_embedding_model()
    return _embedding_model


class EmbeddingGenerator:
    """Generates embeddings for document chunks using Azure OpenAI."""

    def __init__(
        self,
        model: str | None = None,
        batch_size: int = 100,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize embedding generator.

        Args:
            model: Azure OpenAI deployment name for embeddings.
            batch_size: Number of texts to process in parallel.
            max_retries: Maximum retry attempts.
            retry_delay: Delay between retries in seconds.
        """
        self.model = model or _get_model()
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Model-specific configurations
        self.model_configs = {
            "text-embedding-3-small": {"dimensions": 1536, "max_tokens": 8191},
            "text-embedding-3-large": {"dimensions": 3072, "max_tokens": 8191},
            "text-embedding-ada-002": {"dimensions": 1536, "max_tokens": 8191},
        }

        if self.model not in self.model_configs:
            logger.warning(f"Unknown model {self.model}, using default config")
            self.config = {"dimensions": 1536, "max_tokens": 8191}
        else:
            self.config = self.model_configs[self.model]

    async def generate_embedding(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        if len(text) > self.config["max_tokens"] * 4:
            text = text[: self.config["max_tokens"] * 4]

        for attempt in range(self.max_retries):
            try:
                client = _get_client()
                response = await client.embeddings.create(
                    model=self.model,
                    input=text,
                )
                return response.data[0].embedding

            except RateLimitError:
                if attempt == self.max_retries - 1:
                    raise

                delay = self.retry_delay * (2**attempt)
                logger.warning(f"Rate limit hit, retrying in {delay}s")
                await asyncio.sleep(delay)

            except APIError as e:
                logger.error(f"Azure OpenAI API error: {e}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(self.retry_delay)

            except Exception as e:
                logger.error(f"Unexpected error generating embedding: {e}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(self.retry_delay)

        return [0.0] * self.config["dimensions"]

    async def generate_embeddings_batch(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        processed_texts = []
        for text in texts:
            if not text or not text.strip():
                processed_texts.append("")
                continue

            if len(text) > self.config["max_tokens"] * 4:
                text = text[: self.config["max_tokens"] * 4]

            processed_texts.append(text)

        for attempt in range(self.max_retries):
            try:
                client = _get_client()
                response = await client.embeddings.create(
                    model=self.model,
                    input=processed_texts,
                )
                return [data.embedding for data in response.data]

            except RateLimitError:
                if attempt == self.max_retries - 1:
                    raise

                delay = self.retry_delay * (2**attempt)
                logger.warning(f"Rate limit hit, retrying batch in {delay}s")
                await asyncio.sleep(delay)

            except APIError as e:
                logger.error(f"Azure OpenAI API error in batch: {e}")
                if attempt == self.max_retries - 1:
                    return await self._process_individually(processed_texts)
                await asyncio.sleep(self.retry_delay)

            except Exception as e:
                logger.error(f"Unexpected error in batch embedding: {e}")
                if attempt == self.max_retries - 1:
                    return await self._process_individually(processed_texts)
                await asyncio.sleep(self.retry_delay)

        return [[0.0] * self.config["dimensions"]] * len(texts)

    async def _process_individually(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Process texts individually as fallback."""
        embeddings = []

        for text in texts:
            try:
                if not text or not text.strip():
                    embeddings.append([0.0] * self.config["dimensions"])
                    continue

                embedding = await self.generate_embedding(text)
                embeddings.append(embedding)

                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Failed to embed text: {e}")
                embeddings.append([0.0] * self.config["dimensions"])

        return embeddings

    async def embed_chunks(
        self,
        chunks: list[DocumentChunk],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[DocumentChunk]:
        """
        Generate embeddings for document chunks.

        Args:
            chunks: List of document chunks.
            progress_callback: Optional callback for progress updates.

        Returns:
            Chunks with embeddings added.
        """
        if not chunks:
            return chunks

        logger.info(f"Generating embeddings for {len(chunks)} chunks")

        embedded_chunks = []
        total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[i : i + self.batch_size]
            batch_texts = [chunk.content for chunk in batch_chunks]

            try:
                embeddings = await self.generate_embeddings_batch(batch_texts)

                for chunk, embedding in zip(batch_chunks, embeddings):
                    embedded_chunk = DocumentChunk(
                        content=chunk.content,
                        index=chunk.index,
                        start_char=chunk.start_char,
                        end_char=chunk.end_char,
                        metadata={
                            **chunk.metadata,
                            "embedding_model": self.model,
                            "embedding_generated_at": datetime.now().isoformat(),
                        },
                        token_count=chunk.token_count,
                    )

                    embedded_chunk.embedding = embedding
                    embedded_chunks.append(embedded_chunk)

                current_batch = (i // self.batch_size) + 1
                if progress_callback:
                    progress_callback(current_batch, total_batches)

                logger.info(f"Processed batch {current_batch}/{total_batches}")

            except Exception as e:
                logger.error(f"Failed to process batch {i//self.batch_size + 1}: {e}")

                for chunk in batch_chunks:
                    chunk.metadata.update(
                        {
                            "embedding_error": str(e),
                            "embedding_generated_at": datetime.now().isoformat(),
                        }
                    )
                    chunk.embedding = [0.0] * self.config["dimensions"]
                    embedded_chunks.append(chunk)

        logger.info(f"Generated embeddings for {len(embedded_chunks)} chunks")
        return embedded_chunks

    async def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a search query."""
        return await self.generate_embedding(query)

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings for this model."""
        return self.config["dimensions"]


class EmbeddingCache:
    """Simple in-memory cache for embeddings."""

    def __init__(self, max_size: int = 1000):
        """Initialize cache."""
        self.cache: dict[str, list[float]] = {}
        self.access_times: dict[str, datetime] = {}
        self.max_size = max_size

    def get(self, text: str) -> list[float] | None:
        """Get embedding from cache."""
        text_hash = self._hash_text(text)
        if text_hash in self.cache:
            self.access_times[text_hash] = datetime.now()
            return self.cache[text_hash]
        return None

    def put(self, text: str, embedding: list[float]) -> None:
        """Store embedding in cache."""
        text_hash = self._hash_text(text)

        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]

        self.cache[text_hash] = embedding
        self.access_times[text_hash] = datetime.now()

    def _hash_text(self, text: str) -> str:
        """Generate hash for text."""
        return hashlib.md5(text.encode()).hexdigest()


def create_embedder(
    model: str | None = None,
    use_cache: bool = True,
    **kwargs: Any,
) -> EmbeddingGenerator:
    """
    Create embedding generator with optional caching.

    Args:
        model: Embedding model to use.
        use_cache: Whether to use caching.
        **kwargs: Additional arguments for EmbeddingGenerator.

    Returns:
        EmbeddingGenerator instance.
    """
    embedder = EmbeddingGenerator(model=model, **kwargs)

    if use_cache:
        cache = EmbeddingCache()
        original_generate = embedder.generate_embedding

        async def cached_generate(text: str) -> list[float]:
            cached = cache.get(text)
            if cached is not None:
                return cached

            embedding = await original_generate(text)
            cache.put(text, embedding)
            return embedding

        embedder.generate_embedding = cached_generate

    return embedder
