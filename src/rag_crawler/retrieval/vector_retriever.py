"""
Vector Retriever for RAG_v5 using Azure PostgreSQL with pgvector.

Provides a clean interface for vector similarity search that can be used
by the agent architecture.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from rag_crawler.ingestion.embedder import create_embedder
from rag_crawler.utils.db_utils import db_pool, initialize_database

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDocument:
    """A document chunk retrieved from vector search."""

    chunk_id: str
    document_id: str
    content: str
    score: float
    document_title: str
    document_source: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def page_content(self) -> str:
        """Alias for content to match LangChain Document interface."""
        return self.content


class VectorRetriever:
    """
    Vector retriever using Azure PostgreSQL with pgvector extension.

    Provides async vector similarity search with configurable parameters.
    """

    def __init__(
        self,
        embedding_model: str | None = None,
        default_top_k: int = 5,
        default_threshold: float = 0.3,
    ):
        """
        Initialize vector retriever.

        Args:
            embedding_model: Azure OpenAI embedding deployment name.
            default_top_k: Default number of results to return.
            default_threshold: Default similarity threshold (0-1).
        """
        self.embedder = create_embedder(model=embedding_model)
        self.default_top_k = default_top_k
        self.default_threshold = default_threshold
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize database connection if not already done."""
        if not self._initialized:
            await initialize_database()
            self._initialized = True

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        threshold: float | None = None,
        filter_metadata: dict[str, Any] | None = None,
        source_filter: str | None = None,
    ) -> list[RetrievedDocument]:
        """
        Retrieve documents similar to the query.

        Args:
            query: Search query text.
            top_k: Number of results to return (default: self.default_top_k).
            threshold: Minimum similarity threshold (default: self.default_threshold).
            filter_metadata: Optional metadata filters.
            source_filter: Optional source path prefix to filter documents (e.g., "new_uploads/").

        Returns:
            List of RetrievedDocument objects sorted by relevance.
        """
        await self.initialize()

        top_k = top_k or self.default_top_k
        threshold = threshold or self.default_threshold

        # Generate query embedding
        query_embedding = await self.embedder.embed_query(query)

        # Search using pgvector
        results = await self._search_vectors(
            query_embedding=query_embedding,
            limit=top_k,
            threshold=threshold,
            filter_metadata=filter_metadata,
            source_filter=source_filter,
        )

        return results

    async def retrieve_with_scores(
        self,
        query: str,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> list[tuple[RetrievedDocument, float]]:
        """
        Retrieve documents with their similarity scores.

        Args:
            query: Search query text.
            top_k: Number of results to return.
            threshold: Minimum similarity threshold.

        Returns:
            List of (document, score) tuples.
        """
        docs = await self.retrieve(query, top_k, threshold)
        return [(doc, doc.score) for doc in docs]

    async def retrieve_with_exclusion(
        self,
        query: str,
        exclude_chunk_ids: set[str] | None = None,
        top_k: int | None = None,
        threshold: float | None = None,
        source_filter: str | None = None,
    ) -> list[RetrievedDocument]:
        """
        Retrieve documents excluding already-seen chunks.

        Used by LogicRAG agent for multi-hop retrieval to avoid
        retrieving the same chunks across rounds.
        """
        await self.initialize()

        top_k = top_k or self.default_top_k
        threshold = threshold or self.default_threshold

        if not exclude_chunk_ids:
            return await self.retrieve(query, top_k, threshold, source_filter=source_filter)

        extended_k = top_k + len(exclude_chunk_ids)
        results = await self.retrieve(query, extended_k, threshold, source_filter=source_filter)
        filtered = [doc for doc in results if doc.chunk_id not in exclude_chunk_ids]
        return filtered[:top_k]

    async def retrieve_summaries(
        self,
        query: str,
        top_k: int = 3,
        threshold: float = 0.3,
    ) -> list[dict[str, Any]]:
        """
        Retrieve document-level summaries for warm-up retrieval.

        Used by LogicRAG agent for broad document-level context
        before drilling into specific chunks.
        """
        await self.initialize()

        query_embedding = await self.embedder.embed_query(query)
        embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

        async with db_pool.acquire() as conn:
            results = await conn.fetch(
                """
                SELECT
                    ds.id::text AS summary_id,
                    ds.document_id::text,
                    ds.summary,
                    ds.key_entities,
                    ds.key_topics,
                    d.title AS document_title,
                    d.source AS document_source,
                    1 - (ds.embedding <=> $1::vector) AS score
                FROM document_summaries ds
                JOIN documents d ON ds.document_id = d.id
                WHERE ds.embedding IS NOT NULL
                AND 1 - (ds.embedding <=> $1::vector) >= $2
                ORDER BY ds.embedding <=> $1::vector
                LIMIT $3
                """,
                embedding_str,
                threshold,
                top_k,
            )

            return [
                {
                    "summary_id": row["summary_id"],
                    "document_id": row["document_id"],
                    "summary": row["summary"],
                    "key_entities": list(row["key_entities"]) if row["key_entities"] else [],
                    "key_topics": list(row["key_topics"]) if row["key_topics"] else [],
                    "document_title": row["document_title"],
                    "document_source": row["document_source"],
                    "score": float(row["score"]),
                }
                for row in results
            ]

    async def _search_vectors(
        self,
        query_embedding: list[float],
        limit: int,
        threshold: float,
        filter_metadata: dict[str, Any] | None = None,
        source_filter: str | None = None,
    ) -> list[RetrievedDocument]:
        """
        Execute vector similarity search against PostgreSQL.

        Args:
            query_embedding: Query embedding vector.
            limit: Maximum number of results.
            threshold: Minimum similarity threshold.
            filter_metadata: Optional metadata filters.
            source_filter: Optional source path prefix to filter documents.

        Returns:
            List of RetrievedDocument objects.
        """
        embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

        # Build query with optional metadata filters
        base_query = """
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
        """

        # Add metadata filters if provided
        params = [embedding_str, threshold]

        # Add source path filter if provided (for "New document" mode)
        if source_filter:
            param_idx = len(params) + 1
            base_query += f" AND d.source LIKE ${param_idx}"
            params.append(f"{source_filter}%")

        if filter_metadata:
            for key, value in filter_metadata.items():
                param_idx = len(params) + 1
                base_query += f" AND c.metadata->>'{key}' = ${param_idx}"
                params.append(str(value))

        base_query += """
            ORDER BY c.embedding <=> $1::vector
            LIMIT $""" + str(len(params) + 1)
        params.append(limit)

        async with db_pool.acquire() as conn:
            results = await conn.fetch(base_query, *params)

            documents = []
            for row in results:
                import json

                metadata = {}
                if row["metadata"]:
                    try:
                        metadata = (
                            json.loads(row["metadata"])
                            if isinstance(row["metadata"], str)
                            else row["metadata"]
                        )
                    except (json.JSONDecodeError, TypeError):
                        metadata = {}

                doc = RetrievedDocument(
                    chunk_id=row["chunk_id"],
                    document_id=row["document_id"],
                    content=row["content"],
                    score=float(row["score"]),
                    document_title=row["document_title"],
                    document_source=row["document_source"],
                    chunk_index=row["chunk_index"],
                    metadata=metadata,
                )
                documents.append(doc)

            return documents

    async def get_document_count(self) -> int:
        """Get total number of documents in the database."""
        await self.initialize()

        async with db_pool.acquire() as conn:
            result = await conn.fetchval("SELECT COUNT(*) FROM documents")
            return result or 0

    async def get_chunk_count(self) -> int:
        """Get total number of chunks in the database."""
        await self.initialize()

        async with db_pool.acquire() as conn:
            result = await conn.fetchval("SELECT COUNT(*) FROM chunks")
            return result or 0

    async def similarity_search(
        self,
        query: str,
        k: int = 5,
        source_filter: str | None = None,
    ) -> list[RetrievedDocument]:
        """
        Simple similarity search interface (LangChain-compatible).

        Args:
            query: Search query.
            k: Number of results.
            source_filter: Optional source path prefix to filter documents.

        Returns:
            List of retrieved documents.
        """
        return await self.retrieve(query, top_k=k, source_filter=source_filter)


# Factory function for easy instantiation
def create_retriever(
    embedding_model: str | None = None,
    default_top_k: int = 5,
    default_threshold: float = 0.3,
) -> VectorRetriever:
    """
    Create a VectorRetriever instance.

    Args:
        embedding_model: Azure OpenAI embedding deployment name.
        default_top_k: Default number of results.
        default_threshold: Default similarity threshold.

    Returns:
        Configured VectorRetriever instance.
    """
    return VectorRetriever(
        embedding_model=embedding_model,
        default_top_k=default_top_k,
        default_threshold=default_threshold,
    )
