"""
State definitions for RAG_v5 LangGraph router.

Defines the state that flows through the agentic RAG pipeline.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypedDict

from rag_crawler.retrieval.vector_retriever import RetrievedDocument


class QueryIntent(str, Enum):
    """Query intent types for agent routing."""

    QUICK_FACT = "quick_fact"  # Simple factual questions
    MULTI_HOP = "multi_hop"  # Multi-step reasoning across sources
    IN_DEPTH = "in_depth"  # Complex analysis, explanation, comparison
    UNKNOWN = "unknown"  # When classification is uncertain


@dataclass
class QueryAnalysis:
    """Analysis result from query classifier."""

    intent: QueryIntent
    confidence: float  # 0.0 to 1.0
    reasoning: str  # LLM's reasoning for the classification
    keywords: list[str] = field(default_factory=list)
    complexity_score: float = 0.5  # 0.0 (simple) to 1.0 (complex)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "intent": self.intent.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "keywords": self.keywords,
            "complexity_score": self.complexity_score,
        }


class RAGState(TypedDict, total=False):
    """
    State for the RAG LangGraph workflow.

    This state flows through all nodes in the graph.
    Uses TypedDict for LangGraph compatibility.
    """

    # Session info
    session_id: str

    # Input
    query: str
    conversation_history: list[dict[str, str]]

    # Classification
    query_analysis: dict[str, Any] | None

    # Retrieval
    retrieved_documents: list[dict[str, Any]]
    reranked_documents: list[dict[str, Any]]
    source_filter: str | None  # Filter for "New document" mode (e.g., "new_uploads/")

    # LogicRAG fields
    subproblems: list[str]
    dependency_pairs: list[list[int]]
    sorted_subproblems: list[str]
    info_summary: str
    round_count: int
    max_rounds: int

    # Response
    response: str
    sources: list[dict[str, Any]]

    # Generation settings
    temperature: float

    # Metadata
    current_agent: str
    step_count: int
    error: str | None
    metadata: dict[str, Any]


def create_initial_state(
    query: str,
    session_id: str = "",
    conversation_history: list[dict[str, str]] | None = None,
    source_filter: str | None = None,
    temperature: float = 0.0,
) -> RAGState:
    """
    Create an initial RAGState for a new query.

    Args:
        query: User's query.
        session_id: Session identifier.
        conversation_history: Previous conversation turns.
        source_filter: Optional source path prefix for filtering (e.g., "new_uploads/").
        temperature: LLM temperature for response generation (0 = strict/factual).

    Returns:
        Initialized RAGState.
    """
    return RAGState(
        session_id=session_id,
        query=query,
        conversation_history=conversation_history or [],
        query_analysis=None,
        retrieved_documents=[],
        reranked_documents=[],
        source_filter=source_filter,
        subproblems=[],
        dependency_pairs=[],
        sorted_subproblems=[],
        info_summary="",
        round_count=0,
        max_rounds=5,
        response="",
        sources=[],
        temperature=temperature,
        current_agent="classifier",
        step_count=0,
        error=None,
        metadata={},
    )


def documents_to_state_format(docs: list[RetrievedDocument]) -> list[dict[str, Any]]:
    """
    Convert RetrievedDocument objects to state-friendly dict format.

    Args:
        docs: List of RetrievedDocument objects.

    Returns:
        List of dictionaries suitable for state.
    """
    return [
        {
            "chunk_id": doc.chunk_id,
            "document_id": doc.document_id,
            "content": doc.content,
            "score": doc.score,
            "document_title": doc.document_title,
            "document_source": doc.document_source,
            "chunk_index": doc.chunk_index,
            "metadata": doc.metadata,
        }
        for doc in docs
    ]


def format_sources_for_response(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Format documents as sources for the final response.

    Args:
        docs: Retrieved/reranked documents.

    Returns:
        List of source references.
    """
    sources = []
    for i, doc in enumerate(docs, 1):
        sources.append(
            {
                "number": i,
                "title": doc.get("document_title", "Unknown"),
                "source": doc.get("document_source", ""),
                "chunk_index": doc.get("chunk_index", 0),
                "page": doc.get("metadata", {}).get("page", "N/A"),
                "snippet": doc.get("content", "")[:200] + "..."
                if len(doc.get("content", "")) > 200
                else doc.get("content", ""),
                "score": doc.get("score", 0.0),
                "content": doc.get("content", ""),
            }
        )
    return sources
