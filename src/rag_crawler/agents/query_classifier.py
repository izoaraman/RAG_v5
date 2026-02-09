"""
LLM-based Query Classifier for RAG_v5.

Uses Azure OpenAI to classify queries into intent categories
for intelligent agent routing.
"""

import json
import logging
import os
import re
from typing import Any

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import SecretStr

from rag_crawler.router.state import QueryIntent, QueryAnalysis

logger = logging.getLogger(__name__)

# Classification prompt
CLASSIFICATION_SYSTEM_PROMPT = """You are a query classification expert. Your task is to analyze user questions and classify them into one of three categories:

1. QUICK_FACT: Simple, direct questions that can be answered with a brief, factual response.
   Examples:
   - "What is the definition of X?"
   - "When was Y established?"
   - "Who is the CEO of Z?"
   - "How much does X cost?"
   - "What are the main features of Y?"

2. IN_DEPTH: Complex questions requiring detailed explanation, analysis, comparison, or comprehensive coverage.
   Examples:
   - "Explain how X works and its implications"
   - "Compare A and B in terms of efficiency"
   - "Analyze the impact of X on Y"
   - "What are the pros and cons of X?"
   - "Provide a comprehensive overview of X"
    - "How does X affect Y in different scenarios?"

3. MULTI_HOP: Questions that require connecting information from multiple sources, multi-step reasoning, or answering through intermediate sub-questions.
   Examples:
   - "What is the relationship between X's policy on Y and Z's regulation of W?"
   - "How did the decision about X lead to changes in Y?"
   - "Compare the enforcement actions taken by ACCC and AER in cases involving Z"
   - "What are the implications of X for Y, considering Z?"
   - Questions with multiple entities/concepts that must be connected
   - Questions requiring causal chains or temporal reasoning

Respond with a JSON object containing:
{
    "intent": "QUICK_FACT" or "IN_DEPTH" or "MULTI_HOP",
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation of why this classification",
    "keywords": ["key", "words", "from", "query"],
    "complexity_score": 0.0 to 1.0
}

Only respond with the JSON object, no other text."""


class QueryClassifier:
    """
    LLM-based query classifier using Azure OpenAI.

    Analyzes user queries and classifies them for optimal agent routing.
    """

    def __init__(
        self,
        model_name: str | None = None,
        temperature: float = 0.0,
    ):
        """
        Initialize the query classifier.

        Args:
            model_name: Azure OpenAI deployment name.
            temperature: LLM temperature (lower = more deterministic).
        """
        self.model_name = model_name or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
        self.temperature = temperature

        # Initialize Azure OpenAI
        self.llm = AzureChatOpenAI(
            azure_deployment=self.model_name,
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=SecretStr(os.getenv("AZURE_OPENAI_API_KEY", "")),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            temperature=temperature,
        )

        logger.info(f"QueryClassifier initialized with model: {self.model_name}")

    async def classify(
        self,
        query: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> QueryAnalysis:
        """
        Classify a user query.

        Args:
            query: The user's question.
            conversation_history: Previous conversation for context.

        Returns:
            QueryAnalysis with classification result.
        """
        try:
            # Build context from conversation history
            context = ""
            if conversation_history and len(conversation_history) > 0:
                recent = conversation_history[-4:]  # Last 2 exchanges
                context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent])

            # Build the classification prompt
            user_prompt = f"""Classify the following query:

Query: {query}

{f"Recent conversation context:{chr(10)}{context}" if context else ""}

Respond with JSON only."""

            # Call LLM
            messages = [
                SystemMessage(content=CLASSIFICATION_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]

            response = await self.llm.ainvoke(messages)
            response_content = response.content
            if isinstance(response_content, str):
                response_text = response_content.strip()
            else:
                response_text = str(response_content).strip()

            # Parse JSON response
            try:
                # Handle potential markdown code blocks
                if response_text.startswith("```"):
                    response_text = response_text.split("```")[1]
                    if response_text.startswith("json"):
                        response_text = response_text[4:]
                    response_text = response_text.strip()

                result = json.loads(response_text)

                intent_str = result.get("intent", "UNKNOWN").upper()
                if intent_str == "QUICK_FACT":
                    intent = QueryIntent.QUICK_FACT
                elif intent_str == "MULTI_HOP":
                    intent = QueryIntent.MULTI_HOP
                else:
                    intent = QueryIntent.IN_DEPTH

                return QueryAnalysis(
                    intent=intent,
                    confidence=float(result.get("confidence", 0.8)),
                    reasoning=result.get("reasoning", "LLM classification"),
                    keywords=result.get("keywords", []),
                    complexity_score=float(result.get("complexity_score", 0.5)),
                )

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse LLM response as JSON: {e}")
                return self._fallback_classification(query)

        except Exception as e:
            logger.error(f"Query classification failed: {e}")
            return self._fallback_classification(query)

    def classify_sync(
        self,
        query: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> QueryAnalysis:
        """
        Synchronous version of classify.

        Args:
            query: The user's question.
            conversation_history: Previous conversation for context.

        Returns:
            QueryAnalysis with classification result.
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.classify(query, conversation_history))

    def _fallback_classification(self, query: str) -> QueryAnalysis:
        """
        Pattern-based fallback classification when LLM fails.

        Args:
            query: The user's question.

        Returns:
            QueryAnalysis based on heuristics.
        """
        query_lower = query.lower()

        # Quick fact patterns
        quick_patterns = [
            "what is",
            "what are",
            "when",
            "who",
            "where",
            "how much",
            "how many",
            "define",
            "meaning of",
            "list",
            "name",
            "give me",
            "show me",
            "tell me",
        ]

        # In-depth patterns
        depth_patterns = [
            "explain",
            "analyze",
            "analyse",
            "compare",
            "contrast",
            "evaluate",
            "assess",
            "describe how",
            "why does",
            "what are the implications",
            "pros and cons",
            "comprehensive",
            "detailed",
            "in-depth",
            "thorough",
            "how does",
            "impact of",
            "effect of",
        ]

        multi_hop_patterns = [
            r"relationship between",
            r"how did.*lead to",
            r"how does.*affect",
            r"compare.*and.*in terms of",
            r"implications of.*for",
            r"what caused.*to",
            r"connection between",
            r"link between",
        ]

        # Check patterns
        quick_score = sum(1 for p in quick_patterns if p in query_lower)
        depth_score = sum(1 for p in depth_patterns if p in query_lower)
        multi_hop_score = sum(1 for p in multi_hop_patterns if re.search(p, query_lower))

        # Also consider query length
        word_count = len(query.split())
        if word_count > 20:
            depth_score += 1

        if multi_hop_score > 0:
            return QueryAnalysis(
                intent=QueryIntent.MULTI_HOP,
                confidence=0.6,
                reasoning="Pattern-based fallback: detected multi-hop reasoning indicators",
                keywords=query.split()[:5],
                complexity_score=0.8,
            )

        if depth_score > quick_score:
            return QueryAnalysis(
                intent=QueryIntent.IN_DEPTH,
                confidence=0.6,
                reasoning="Pattern-based fallback: detected complexity indicators",
                keywords=query.split()[:5],
                complexity_score=0.7,
            )
        else:
            return QueryAnalysis(
                intent=QueryIntent.QUICK_FACT,
                confidence=0.6,
                reasoning="Pattern-based fallback: simple question structure",
                keywords=query.split()[:5],
                complexity_score=0.3,
            )


# Singleton instance for convenience
_classifier: QueryClassifier | None = None


def get_classifier() -> QueryClassifier:
    """Get or create the global classifier instance."""
    global _classifier
    if _classifier is None:
        _classifier = QueryClassifier()
    return _classifier


async def classify_query(
    query: str,
    conversation_history: list[dict[str, str]] | None = None,
) -> QueryAnalysis:
    """
    Convenience function to classify a query.

    Args:
        query: The user's question.
        conversation_history: Previous conversation for context.

    Returns:
        QueryAnalysis with classification result.
    """
    classifier = get_classifier()
    return await classifier.classify(query, conversation_history)
