"""
Quick Fact Agent for RAG_v5.

Handles simple, direct questions with fast vector retrieval
and concise response generation.
"""

import logging
from typing import Any

from rag_crawler.retrieval.vector_retriever import VectorRetriever, create_retriever
from rag_crawler.router.state import RAGState, documents_to_state_format

logger = logging.getLogger(__name__)


class QuickFactAgent:
    """
    Agent for quick, factual questions.

    Uses simple vector retrieval with a small top_k for fast responses.
    Generates concise, direct answers.
    """

    def __init__(
        self,
        retriever: VectorRetriever | None = None,
        top_k: int = 3,
        threshold: float = 0.3,
    ):
        """
        Initialize Quick Fact Agent.

        Args:
            retriever: VectorRetriever instance (created if not provided).
            top_k: Number of documents to retrieve.
            threshold: Minimum similarity threshold.
        """
        self.retriever = retriever or create_retriever(
            default_top_k=top_k,
            default_threshold=threshold,
        )
        self.top_k = top_k
        self.threshold = threshold

        logger.info(f"QuickFactAgent initialized: top_k={top_k}, threshold={threshold}")

    async def process(self, state: RAGState) -> RAGState:
        """
        Process query through Quick Fact pipeline.

        Args:
            state: Current RAG state with query.

        Returns:
            Updated state with retrieved documents.
        """
        query = state.get("query", "")

        if not query:
            logger.warning("QuickFactAgent received empty query")
            state["error"] = "Empty query"
            return state

        logger.info(f"QuickFactAgent processing: {query[:50]}...")

        try:
            # Get source filter from state (for "New document" mode)
            source_filter = state.get("source_filter")

            # Simple vector retrieval
            docs = await self.retriever.retrieve(
                query=query,
                top_k=self.top_k,
                threshold=self.threshold,
                source_filter=source_filter,
            )

            # Convert to state format
            state["retrieved_documents"] = documents_to_state_format(docs)
            state["reranked_documents"] = state["retrieved_documents"]  # No reranking
            state["current_agent"] = "quick_fact"
            state["step_count"] = state.get("step_count", 0) + 1

            logger.info(f"QuickFactAgent retrieved {len(docs)} documents")

        except Exception as e:
            logger.error(f"QuickFactAgent retrieval failed: {e}")
            state["error"] = str(e)
            state["retrieved_documents"] = []
            state["reranked_documents"] = []

        return state

    async def __call__(self, state: RAGState) -> RAGState:
        """Allow agent to be called directly."""
        return await self.process(state)


# Response prompt for quick facts
QUICK_FACT_PROMPT = """You are a helpful assistant answering a simple factual question.

Based on the provided context, give a direct, concise answer.

Guidelines:
- Be brief and to the point (1-3 sentences typically)
- If the answer is a specific fact, date, or number, state it clearly
- If the context doesn't contain the answer, say so briefly
- Include a citation number [1] if referencing a specific source
- Do not provide unnecessary elaboration

Context:
{context}

Question: {query}

Answer:"""


def get_quick_fact_prompt(query: str, documents: list[dict[str, Any]]) -> str:
    """
    Generate prompt for quick fact response.

    Args:
        query: User's question.
        documents: Retrieved documents.

    Returns:
        Formatted prompt string.
    """
    if not documents:
        context = "No relevant documents found."
    else:
        context_parts = []
        for i, doc in enumerate(documents, 1):
            content = doc.get("content", "")
            title = doc.get("document_title", "Unknown")
            context_parts.append(f"[{i}] {title}:\n{content[:500]}")
        context = "\n\n".join(context_parts)

    return QUICK_FACT_PROMPT.format(context=context, query=query)
