"""
LangGraph Router for RAG_v5 Agentic Architecture.

Orchestrates query classification, agent routing, retrieval, and response generation
using LangGraph StateGraph.
"""

import logging
import uuid
from typing import Any, Literal

from langgraph.graph import StateGraph, END

from rag_crawler.router.state import RAGState, QueryIntent, create_initial_state
from rag_crawler.agents.query_classifier import QueryClassifier, classify_query
from rag_crawler.agents.quick_fact_agent import QuickFactAgent
from rag_crawler.agents.in_depth_agent import InDepthAgent
from rag_crawler.agents.logic_rag_agent import LogicRAGAgent
from rag_crawler.agents.response_generator import ResponseGenerator
from rag_crawler.retrieval.memory import ConversationMemory, SessionMemoryManager

logger = logging.getLogger(__name__)


class RAGRouter:
    """
    Main LangGraph-based router for RAG_v5.

    Orchestrates the complete RAG pipeline:
    1. Query Classification (LLM-based)
    2. Agent Routing (Quick Fact or In-Depth)
    3. Document Retrieval
    4. Response Generation
    """

    def __init__(self):
        """Initialize RAG Router with all components."""
        # Initialize agents
        self.classifier = QueryClassifier()
        self.quick_fact_agent = QuickFactAgent()
        self.in_depth_agent = InDepthAgent()
        self.logic_rag_agent = LogicRAGAgent()
        self.response_generator = ResponseGenerator()

        # Session management
        self.session_manager = SessionMemoryManager()

        # Build the graph
        self.graph = self._build_graph()
        self.compiled_graph = self.graph.compile()

        logger.info("RAGRouter initialized and compiled")

    def _build_graph(self) -> Any:
        """
        Build the LangGraph workflow.

        Flow:
        START -> classify_query -> route_to_agent -> [quick_fact | in_depth] -> generate_response -> END
        """
        # Create graph with RAGState
        graph = StateGraph(RAGState)

        # Add nodes
        graph.add_node("classify_query", self._classify_node)
        graph.add_node("quick_fact", self._quick_fact_node)
        graph.add_node("in_depth", self._in_depth_node)
        graph.add_node("logic_rag", self._logic_rag_node)
        graph.add_node("generate_response", self._generate_node)

        # Set entry point
        graph.set_entry_point("classify_query")

        # Add conditional routing after classification
        graph.add_conditional_edges(
            "classify_query",
            self._route_to_agent,
            {
                "quick_fact": "quick_fact",
                "in_depth": "in_depth",
                "logic_rag": "logic_rag",
            },
        )

        # Both agents lead to response generation
        graph.add_edge("quick_fact", "generate_response")
        graph.add_edge("in_depth", "generate_response")
        graph.add_edge("logic_rag", "generate_response")

        # Response generation leads to END
        graph.add_edge("generate_response", END)

        return graph

    async def _classify_node(self, state: RAGState) -> RAGState:
        """Node: Classify the query."""
        logger.info(f"Classifying query: {state.get('query', '')[:50]}...")

        try:
            query = state.get("query", "")
            history = state.get("conversation_history", [])

            analysis = await classify_query(query, history)

            state["query_analysis"] = analysis.to_dict()
            state["step_count"] = state.get("step_count", 0) + 1

            logger.info(
                f"Classification: {analysis.intent.value} (confidence: {analysis.confidence})"
            )

        except Exception as e:
            logger.error(f"Classification failed: {e}")
            # Default to quick_fact on error
            state["query_analysis"] = {
                "intent": QueryIntent.QUICK_FACT.value,
                "confidence": 0.5,
                "reasoning": f"Fallback due to error: {e}",
                "keywords": [],
                "complexity_score": 0.5,
            }

        return state

    def _route_to_agent(self, state: RAGState) -> Literal["quick_fact", "in_depth", "logic_rag"]:
        """Conditional edge: Route to appropriate agent based on classification."""
        query_analysis = state.get("query_analysis") or {}
        intent = query_analysis.get("intent", QueryIntent.QUICK_FACT.value)

        if intent == QueryIntent.MULTI_HOP.value:
            return "logic_rag"
        if intent == QueryIntent.IN_DEPTH.value:
            return "in_depth"
        return "quick_fact"

    async def _quick_fact_node(self, state: RAGState) -> RAGState:
        """Node: Process with Quick Fact agent."""
        logger.info("Processing with QuickFactAgent")
        return await self.quick_fact_agent.process(state)

    async def _in_depth_node(self, state: RAGState) -> RAGState:
        """Node: Process with In-Depth agent."""
        logger.info("Processing with InDepthAgent")
        return await self.in_depth_agent.process(state)

    async def _logic_rag_node(self, state: RAGState) -> RAGState:
        """Node: Process with LogicRAG agent."""
        logger.info("Processing with LogicRAGAgent")
        return await self.logic_rag_agent.process(state)

    async def _generate_node(self, state: RAGState) -> RAGState:
        """Node: Generate final response."""
        logger.info("Generating response")
        return await self.response_generator.generate(state)

    async def process_query(
        self,
        query: str,
        session_id: str | None = None,
        source_filter: str | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """
        Process a query through the complete RAG pipeline.

        Args:
            query: User's question.
            session_id: Optional session ID for conversation continuity.
            source_filter: Optional source path prefix for filtering (e.g., "new_uploads/").
            temperature: LLM temperature (0 = strict/factual, 1 = creative).

        Returns:
            Dictionary with response, sources, and metadata.
        """
        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())

        # Get conversation memory
        memory = self.session_manager.get_session(session_id)

        # Create initial state
        state = create_initial_state(
            query=query,
            session_id=session_id,
            conversation_history=memory.get_langchain_messages(),
            source_filter=source_filter,
            temperature=temperature,
        )

        logger.info(f"Processing query (session: {session_id[:8]}...): {query[:50]}...")

        try:
            # Run the graph
            result = await self.compiled_graph.ainvoke(state)

            # Update conversation memory
            memory.add_user_message(query)
            memory.add_assistant_message(result.get("response", ""))

            # Prepare response
            response = {
                "session_id": session_id,
                "query": query,
                "response": result.get("response", ""),
                "sources": result.get("sources", []),
                "query_analysis": result.get("query_analysis", {}),
                "agent_used": result.get("current_agent", "unknown"),
                "documents_retrieved": len(result.get("retrieved_documents", [])),
                "documents_reranked": len(result.get("reranked_documents", [])),
                "steps": result.get("step_count", 0),
                "error": result.get("error"),
                "metadata": result.get("metadata", {}),
            }

            response_text = str(response.get("response", ""))
            logger.info(f"Query processed successfully: {len(response_text)} chars")
            return response

        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "session_id": session_id,
                "query": query,
                "response": "I apologize, but I encountered an error processing your query. Please try again.",
                "sources": [],
                "error": str(e),
            }

    def process_query_sync(
        self,
        query: str,
        session_id: str | None = None,
        source_filter: str | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """
        Synchronous version of process_query.

        Args:
            query: User's question.
            session_id: Optional session ID.
            source_filter: Optional source path prefix for filtering.
            temperature: LLM temperature (0 = strict/factual, 1 = creative).

        Returns:
            Response dictionary.
        """
        import asyncio

        from rag_crawler.utils.db_utils import db_pool

        # Reset DB pool to avoid "attached to a different loop" errors
        # when Streamlit re-runs the script with a new event loop.
        if db_pool._pool is not None:
            try:
                db_pool._pool.close()
            except Exception:
                pass
            db_pool._pool = None

        # Reset retriever initialization flags so they re-init on new loop
        self.quick_fact_agent.retriever._initialized = False
        self.in_depth_agent.retriever._initialized = False
        self.logic_rag_agent.retriever._initialized = False

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            return loop.run_until_complete(
                self.process_query(query, session_id, source_filter, temperature)
            )
        finally:
            loop.close()

    def clear_session(self, session_id: str) -> bool:
        """
        Clear a conversation session.

        Args:
            session_id: Session to clear.

        Returns:
            True if session was cleared.
        """
        return self.session_manager.delete_session(session_id)

    def get_session_history(self, session_id: str) -> list[dict[str, str]]:
        """
        Get conversation history for a session.

        Args:
            session_id: Session ID.

        Returns:
            List of message dictionaries.
        """
        memory = self.session_manager.get_session(session_id)
        return memory.get_langchain_messages()


# Singleton instance
_router: RAGRouter | None = None


def get_router() -> RAGRouter:
    """Get or create the global router instance."""
    global _router
    if _router is None:
        _router = RAGRouter()
    return _router


def create_router() -> RAGRouter:
    """Create a new router instance."""
    return RAGRouter()


async def query(
    query: str,
    session_id: str | None = None,
    source_filter: str | None = None,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """
    Convenience function to process a query.

    Args:
        query: User's question.
        session_id: Optional session ID.
        source_filter: Optional source path prefix for filtering.
        temperature: LLM temperature (0 = strict/factual, 1 = creative).

    Returns:
        Response dictionary.
    """
    router = get_router()
    return await router.process_query(query, session_id, source_filter, temperature)
