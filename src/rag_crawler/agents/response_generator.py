"""
Response Generator for RAG_v5.

Handles final response generation using Azure OpenAI based on
retrieved documents and agent type.
"""

import logging
import os
from typing import Any

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from rag_crawler.router.state import RAGState, QueryIntent, format_sources_for_response
from rag_crawler.agents.quick_fact_agent import get_quick_fact_prompt
from rag_crawler.agents.in_depth_agent import get_in_depth_prompt
from rag_crawler.utils.providers import get_shared_llm

logger = logging.getLogger(__name__)


# System prompt for RAG responses (used at all temperature levels)
RAG_SYSTEM_PROMPT = """You are a knowledgeable assistant that provides accurate, well-sourced answers based on the provided context.

Key guidelines:
- Base your answer on the provided context documents
- Do NOT add information from your own knowledge or training data
- Use citation numbers [1], [2], etc. when referencing specific sources
- Answer as thoroughly as the context allows — extract and synthesise relevant details even if the context is indirect
- If the context only partially addresses the question, answer what you can and briefly note what is not covered
- Only say you cannot answer if the context is truly unrelated to the question
- Match your response length and style to the question type
- Be helpful, accurate, and professional
"""


class ResponseGenerator:
    """
    Generates final responses using Azure OpenAI.

    Adapts response style based on query intent and agent type.
    """

    def __init__(
        self,
        model_name: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ):
        """
        Initialize Response Generator.

        Args:
            model_name: Azure OpenAI deployment name.
            temperature: LLM temperature for response generation.
            max_tokens: Maximum tokens in response.
        """
        self.model_name = model_name or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1")
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Use shared LLM instance
        self.llm = get_shared_llm(temperature=temperature)

        logger.info(f"ResponseGenerator initialized with model: {self.model_name}")

    async def generate(self, state: RAGState) -> RAGState:
        """
        Generate final response based on state.

        Args:
            state: Current RAG state with query and retrieved documents.

        Returns:
            Updated state with response and sources.
        """
        query = state.get("query", "")
        documents = state.get("reranked_documents") or state.get("retrieved_documents", [])
        current_agent = state.get("current_agent", "unknown")
        query_analysis = state.get("query_analysis", {})

        if not query:
            state["response"] = "I need a question to answer."
            state["sources"] = []
            return state

        logger.info(f"Generating response for: {query[:50]}... (agent: {current_agent})")

        try:
            # Determine intent
            intent = QueryIntent.QUICK_FACT
            if query_analysis:
                intent_str = query_analysis.get("intent", "quick_fact")
                intent = (
                    QueryIntent(intent_str)
                    if intent_str in [e.value for e in QueryIntent]
                    else QueryIntent.QUICK_FACT
                )

            # Use temperature from state (set by user's Answer Style slider)
            user_temperature = state.get("temperature", 0.2)

            # Generate appropriate prompt
            if current_agent == "quick_fact" or intent == QueryIntent.QUICK_FACT:
                prompt = get_quick_fact_prompt(query, documents)
            else:
                prompt = get_in_depth_prompt(query, documents, use_framework=True)

            temperature = user_temperature

            # Enhance prompt with LogicRAG rolling memory context
            info_summary = state.get("info_summary", "")
            if current_agent == "logic_rag" and info_summary:
                prompt = (
                    f"## Multi-Hop Reasoning Context\n"
                    f"The following summary was built by iteratively retrieving and analyzing "
                    f"information across multiple reasoning steps:\n\n"
                    f"{info_summary}\n\n"
                    f"---\n\n{prompt}"
                )

            # Set temperature on LLM
            self.llm.temperature = temperature

            # Use balanced system prompt at all temperature levels
            system_prompt = RAG_SYSTEM_PROMPT

            # Include conversation history if available
            conversation_history = state.get("conversation_history", [])
            history_context = ""
            if conversation_history:
                recent = conversation_history[-4:]  # Last 2 exchanges
                history_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent])
                prompt = f"Previous conversation:\n{history_context}\n\n{prompt}"

            # Generate response
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt),
            ]

            response = await self.llm.ainvoke(messages)
            response_text = response.content.strip()

            # Reset temperature to default
            self.llm.temperature = self.temperature

            # Format sources
            sources = format_sources_for_response(documents)

            state["response"] = response_text
            state["sources"] = sources
            state["step_count"] = state.get("step_count", 0) + 1

            # Include LogicRAG metadata in state for UI display
            if current_agent == "logic_rag":
                state["metadata"] = {
                    **state.get("metadata", {}),
                    "subproblems": state.get("subproblems", []),
                    "round_count": state.get("round_count", 0),
                    "max_rounds": state.get("max_rounds", 5),
                }

            logger.info(f"Generated response: {len(response_text)} chars, {len(sources)} sources")

        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            state["response"] = (
                "I apologize, but I encountered an error generating a response. Please try again."
            )
            state["error"] = str(e)
            state["sources"] = []

        return state

    def generate_sync(self, state: RAGState) -> RAGState:
        """
        Synchronous version of generate.

        Args:
            state: Current RAG state.

        Returns:
            Updated state with response.
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.generate(state))

    async def __call__(self, state: RAGState) -> RAGState:
        """Allow generator to be called directly."""
        return await self.generate(state)


# Singleton instance
_generator: ResponseGenerator | None = None


def get_generator() -> ResponseGenerator:
    """Get or create the global generator instance."""
    global _generator
    if _generator is None:
        _generator = ResponseGenerator()
    return _generator


async def generate_response(state: RAGState) -> RAGState:
    """
    Convenience function to generate response.

    Args:
        state: Current RAG state.

    Returns:
        Updated state with response.
    """
    generator = get_generator()
    return await generator.generate(state)
