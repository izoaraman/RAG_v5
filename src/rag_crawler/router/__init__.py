"""
Router module for RAG_v5 agentic architecture.

Components:
- RAGState: State dataclass for LangGraph workflow
- RAGRouter: Main LangGraph-based query router
"""

from rag_crawler.router.state import RAGState, QueryIntent, QueryAnalysis
from rag_crawler.router.router_graph import RAGRouter, create_router

__all__ = [
    "RAGState",
    "QueryIntent",
    "QueryAnalysis",
    "RAGRouter",
    "create_router",
]
