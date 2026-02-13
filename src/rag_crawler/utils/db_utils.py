"""Database utilities for PostgreSQL with PGVector."""

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import asyncpg
from asyncpg.pool import Pool
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class DatabasePool:
    """Manages PostgreSQL connection pool."""

    def __init__(self, database_url: str | None = None):
        """
        Initialize database pool.

        Args:
            database_url: PostgreSQL connection URL.
        """
        self._database_url = database_url
        self._pool: Pool | None = None

    @property
    def database_url(self) -> str:
        """Get database URL, loading from env if not set."""
        if self._database_url is None:
            self._database_url = os.getenv("DATABASE_URL")
        if not self._database_url:
            raise ValueError("DATABASE_URL environment variable not set")
        return self._database_url

    async def initialize(self) -> None:
        """Create connection pool with retry logic."""
        if not self._pool:
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    self._pool = await asyncpg.create_pool(
                        self.database_url,
                        min_size=1,
                        max_size=10,
                        max_inactive_connection_lifetime=300,
                        command_timeout=60,
                        timeout=10,  # Connection timeout (reduced from 30s)
                    )
                    logger.info("Database connection pool initialized")
                    return
                except Exception as e:
                    logger.warning(
                        f"Database connection attempt {attempt + 1}/{max_retries} failed: {e}"
                    )
                    if attempt < max_retries - 1:
                        import asyncio

                        await asyncio.sleep(1)  # Fast retry
                    else:
                        raise

    async def close(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Database connection pool closed")

    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool."""
        if not self._pool:
            await self.initialize()

        assert self._pool is not None

        async with self._pool.acquire() as connection:
            yield connection


# Global database pool instance
db_pool = DatabasePool()


async def initialize_database() -> None:
    """Initialize database connection pool and create tables."""
    await db_pool.initialize()

    # Create tables if they don't exist
    async with db_pool.acquire() as conn:
        # Enable pgvector extension
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

        # Create documents table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                title TEXT NOT NULL,
                source TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        # Create chunks table with vector embedding
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                content TEXT NOT NULL,
                embedding vector(1536),
                chunk_index INTEGER NOT NULL,
                metadata JSONB DEFAULT '{}',
                token_count INTEGER,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        # Create index for vector similarity search
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS chunks_embedding_idx
            ON chunks USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
        """)

        # Create document_summaries table for LogicRAG-style warm-up retrieval
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS document_summaries (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                summary TEXT NOT NULL,
                key_entities TEXT[] DEFAULT '{}',
                key_topics TEXT[] DEFAULT '{}',
                embedding vector(1536),
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        # Create index for summary vector search
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS document_summaries_embedding_idx
            ON document_summaries USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
        """)

        # Add content_hash column to chunks table for deduplication
        await conn.execute("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'chunks' AND column_name = 'content_hash'
                ) THEN
                    ALTER TABLE chunks ADD COLUMN content_hash TEXT;
                END IF;
            END $$;
        """)

        # Create unique index on content_hash for dedup lookups
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS chunks_content_hash_idx
            ON chunks (content_hash)
            WHERE content_hash IS NOT NULL
        """)

        logger.info("Database tables initialized")


async def close_database() -> None:
    """Close database connection pool."""
    await db_pool.close()


async def get_document(document_id: str) -> dict[str, Any] | None:
    """
    Get document by ID.

    Args:
        document_id: Document UUID.

    Returns:
        Document data or None if not found.
    """
    async with db_pool.acquire() as conn:
        result = await conn.fetchrow(
            """
            SELECT
                id::text,
                title,
                source,
                content,
                metadata,
                created_at,
                updated_at
            FROM documents
            WHERE id = $1::uuid
            """,
            document_id,
        )

        if result:
            return {
                "id": result["id"],
                "title": result["title"],
                "source": result["source"],
                "content": result["content"],
                "metadata": json.loads(result["metadata"]),
                "created_at": result["created_at"].isoformat(),
                "updated_at": result["updated_at"].isoformat(),
            }

        return None


async def list_documents(
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """
    List documents with pagination.

    Args:
        limit: Maximum number of documents to return.
        offset: Number of documents to skip.

    Returns:
        List of documents.
    """
    async with db_pool.acquire() as conn:
        results = await conn.fetch(
            """
            SELECT
                d.id::text,
                d.title,
                d.source,
                d.metadata,
                d.created_at,
                d.updated_at,
                COUNT(c.id) AS chunk_count
            FROM documents d
            LEFT JOIN chunks c ON d.id = c.document_id
            GROUP BY d.id, d.title, d.source, d.metadata, d.created_at, d.updated_at
            ORDER BY d.created_at DESC
            LIMIT $1 OFFSET $2
            """,
            limit,
            offset,
        )

        return [
            {
                "id": row["id"],
                "title": row["title"],
                "source": row["source"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                "created_at": row["created_at"].isoformat(),
                "updated_at": row["updated_at"].isoformat(),
                "chunk_count": row["chunk_count"],
            }
            for row in results
        ]


async def search_chunks(
    query_embedding: list[float],
    limit: int = 10,
    threshold: float = 0.7,
) -> list[dict[str, Any]]:
    """
    Search chunks by vector similarity.

    Args:
        query_embedding: Query embedding vector.
        limit: Maximum number of results.
        threshold: Minimum similarity threshold.

    Returns:
        List of matching chunks with scores.
    """
    embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

    async with db_pool.acquire() as conn:
        results = await conn.fetch(
            """
            SELECT
                c.id::text AS chunk_id,
                c.document_id::text,
                c.content,
                c.metadata,
                c.chunk_index,
                d.title AS document_title,
                d.source AS document_source,
                1 - (c.embedding <=> $1::vector) AS score
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE 1 - (c.embedding <=> $1::vector) >= $2
            ORDER BY c.embedding <=> $1::vector
            LIMIT $3
            """,
            embedding_str,
            threshold,
            limit,
        )

        return [
            {
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "content": row["content"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                "chunk_index": row["chunk_index"],
                "document_title": row["document_title"],
                "document_source": row["document_source"],
                "score": float(row["score"]),
            }
            for row in results
        ]


async def get_existing_documents() -> dict[str, dict[str, Any]]:
    """
    Get all existing documents indexed by source.

    Returns:
        Dictionary mapping source to document info.
    """
    async with db_pool.acquire() as conn:
        results = await conn.fetch(
            """
            SELECT
                id::text,
                title,
                source,
                metadata,
                created_at,
                updated_at
            FROM documents
            """
        )

        return {
            row["source"]: {
                "id": row["id"],
                "title": row["title"],
                "source": row["source"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                "created_at": row["created_at"].isoformat(),
                "updated_at": row["updated_at"].isoformat(),
            }
            for row in results
        }


async def document_exists(source: str) -> bool:
    """
    Check if a document with the given source already exists.

    Args:
        source: Document source (URL or file path).

    Returns:
        True if document exists.
    """
    async with db_pool.acquire() as conn:
        result = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM documents WHERE source = $1)",
            source,
        )
        return result


async def get_document_by_source(source: str) -> dict[str, Any] | None:
    """
    Get document by source.

    Args:
        source: Document source (URL or file path).

    Returns:
        Document data or None if not found.
    """
    async with db_pool.acquire() as conn:
        result = await conn.fetchrow(
            """
            SELECT
                id::text,
                title,
                source,
                content,
                metadata,
                created_at,
                updated_at
            FROM documents
            WHERE source = $1
            """,
            source,
        )

        if result:
            return {
                "id": result["id"],
                "title": result["title"],
                "source": result["source"],
                "content": result["content"],
                "metadata": json.loads(result["metadata"]) if result["metadata"] else {},
                "created_at": result["created_at"].isoformat(),
                "updated_at": result["updated_at"].isoformat(),
            }

        return None


async def delete_document(document_id: str) -> bool:
    """
    Delete a document and its chunks.

    Args:
        document_id: Document UUID.

    Returns:
        True if document was deleted.
    """
    async with db_pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM documents WHERE id = $1::uuid",
            document_id,
        )
        return result == "DELETE 1"


async def test_connection() -> bool:
    """
    Test database connection.

    Returns:
        True if connection successful.
    """
    try:
        async with db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False


async def chunk_hash_exists(content_hash: str) -> bool:
    """Check if a chunk with this content hash already exists."""
    async with db_pool.acquire() as conn:
        result = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM chunks WHERE content_hash = $1)",
            content_hash,
        )
        return result


async def store_document_summary(
    document_id: str,
    summary: str,
    key_entities: list[str],
    key_topics: list[str],
    embedding: list[float] | None = None,
) -> str:
    """Store a document summary in the database."""
    embedding_str = None
    if embedding:
        embedding_str = "[" + ",".join(map(str, embedding)) + "]"

    async with db_pool.acquire() as conn:
        result = await conn.fetchrow(
            """
            INSERT INTO document_summaries (document_id, summary, key_entities, key_topics, embedding)
            VALUES ($1::uuid, $2, $3, $4, $5::vector)
            RETURNING id::text
            """,
            document_id,
            summary,
            key_entities,
            key_topics,
            embedding_str,
        )
        return result["id"]


async def get_document_summary(document_id: str) -> dict[str, Any] | None:
    """Get summary for a document."""
    async with db_pool.acquire() as conn:
        result = await conn.fetchrow(
            """
            SELECT id::text, document_id::text, summary, key_entities, key_topics, created_at
            FROM document_summaries
            WHERE document_id = $1::uuid
            """,
            document_id,
        )
        if result:
            return {
                "id": result["id"],
                "document_id": result["document_id"],
                "summary": result["summary"],
                "key_entities": list(result["key_entities"]) if result["key_entities"] else [],
                "key_topics": list(result["key_topics"]) if result["key_topics"] else [],
                "created_at": result["created_at"].isoformat(),
            }
        return None
