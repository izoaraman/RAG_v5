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
    Fast query classifier with heuristic-first approach.

    Uses regex/pattern matching for instant classification (< 1ms).
    LLM-based classification available as optional mode for ambiguous queries.
    """

    def __init__(
        self,
        model_name: str | None = None,
        temperature: float = 0.0,
        use_llm: bool = False,
    ):
        """
        Initialize the query classifier.

        Args:
            model_name: Azure OpenAI deployment name (only used if use_llm=True).
            temperature: LLM temperature (lower = more deterministic).
            use_llm: Whether to use LLM for classification (default: False for speed).
        """
        self.model_name = model_name or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1")
        self.temperature = temperature
        self.use_llm = use_llm

        # Lazy-initialize LLM only when needed
        self._llm: AzureChatOpenAI | None = None

        logger.info(f"QueryClassifier initialized (mode={'llm' if use_llm else 'heuristic'})")

    @property
    def llm(self) -> AzureChatOpenAI:
        """Lazy-initialize Azure OpenAI client."""
        if self._llm is None:
            self._llm = AzureChatOpenAI(
                azure_deployment=self.model_name,
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=SecretStr(os.getenv("AZURE_OPENAI_API_KEY", "")),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
                temperature=self.temperature,
            )
        return self._llm

    async def classify(
        self,
        query: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> QueryAnalysis:
        """
        Classify a user query.

        Uses fast heuristic classification by default.
        Falls back to LLM only when use_llm=True.

        Args:
            query: The user's question.
            conversation_history: Previous conversation for context.

        Returns:
            QueryAnalysis with classification result.
        """
        # Fast path: heuristic classification (< 1ms)
        if not self.use_llm:
            return self._heuristic_classification(query)

        # Slow path: LLM classification (2-5s)
        return await self._llm_classification(query, conversation_history)

    async def _llm_classification(
        self,
        query: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> QueryAnalysis:
        """LLM-based classification for when high accuracy is needed."""
        try:
            context = ""
            if conversation_history and len(conversation_history) > 0:
                recent = conversation_history[-4:]
                context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent])

            user_prompt = f"""Classify the following query:

Query: {query}

{f"Recent conversation context:{chr(10)}{context}" if context else ""}

Respond with JSON only."""

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

            try:
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
                return self._heuristic_classification(query)

        except Exception as e:
            logger.error(f"LLM classification failed, using heuristic: {e}")
            return self._heuristic_classification(query)

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

    def _heuristic_classification(self, query: str) -> QueryAnalysis:
        """
        Fast pattern-based classification (< 1ms).

        Uses regex patterns, query structure analysis, and word count
        heuristics for reliable zero-latency classification.

        Args:
            query: The user's question.

        Returns:
            QueryAnalysis based on heuristics.
        """
        query_lower = query.lower().strip()
        word_count = len(query.split())

        # --- Multi-hop patterns (check first — most specific) ---
        multi_hop_patterns = [
            r"relationship between",
            r"how did.*lead to",
            r"how does.*affect.*(?:and|while|considering)",
            r"compare.*and.*in terms of",
            r"compare.*between",
            r"implications of.*for",
            r"what caused.*to",
            r"connection between",
            r"link between",
            r"what is the.*between.*and",
            r"how are.*and.*related",
            r"trace the.*from.*to",
            r"what role does.*play in",
            r"how.*interact with",
            r"combine.*with",
            r"across.*(?:different|multiple|various)",
        ]
        multi_hop_score = sum(1 for p in multi_hop_patterns if re.search(p, query_lower))

        # Multi-entity detection: queries mentioning 2+ distinct concepts with connectors
        connector_words = [" and ", " vs ", " versus ", " or ", " while ", " whereas "]
        connector_count = sum(1 for c in connector_words if c in query_lower)
        if connector_count >= 1 and word_count > 15:
            multi_hop_score += 1

        # Question marks + sub-clauses suggest multi-step
        if query.count("?") > 1:
            multi_hop_score += 1

        # Require stronger evidence for multi-hop on short queries
        # Short questions like "what is the relationship between X and Y?"
        # are usually in-depth, not multi-hop
        multi_hop_threshold = 1 if word_count >= 15 else 2

        if multi_hop_score >= multi_hop_threshold:
            return QueryAnalysis(
                intent=QueryIntent.MULTI_HOP,
                confidence=0.75,
                reasoning=f"Heuristic: {multi_hop_score} multi-hop indicator(s), threshold={multi_hop_threshold}",
                keywords=query.split()[:5],
                complexity_score=0.8,
            )

        # --- In-depth patterns ---
        depth_patterns = [
            "explain",
            "analyze",
            "analyse",
            "compare",
            "contrast",
            "evaluate",
            "assess",
            "describe how",
            "describe the",
            "why does",
            "why is",
            "why are",
            "what are the implications",
            "pros and cons",
            "comprehensive",
            "detail",
            "detailed",
            "in-depth",
            "in depth",
            "thorough",
            "how does",
            "how do",
            "impact of",
            "effect of",
            "advantages and disadvantages",
            "summarize",
            "summarise",
            "summary of",
            "overview of",
            "discuss",
            "elaborate",
            "what factors",
            "what challenges",
            "how can",
            "how should",
            "what steps",
            "walk me through",
            "break down",
            "breakdown",
            "tell me about",
            "tell me more",
            "go through",
            "deep dive",
        ]
        depth_score = sum(1 for p in depth_patterns if p in query_lower)

        # Document-specific queries: naming a specific document/file/report
        # signals the user wants substantive analysis, not a quick fact
        doc_specific_patterns = [
            r"\.pdf",
            r"\.docx?",
            r"\.xlsx?",
            r"annual.report",
            r"enterprise.agreement",
            r"compliance.*enforcement",
            r"policies.*priorities",
            r"in\s+(?:the\s+)?(?:document|report|file|paper|policy)",
            r"about\s+(?:the\s+)?(?:document|report|file|paper|policy)",
            r"from\s+(?:the\s+)?(?:document|report|file|paper|policy)",
            r"(?:accc|aer)[\s\-][\w\-]+[\s\-][\w\-]+",  # e.g. "accc-compliance-enforcement..."
        ]
        if any(re.search(p, query_lower) for p in doc_specific_patterns):
            depth_score += 2

        # Longer queries tend to be more complex
        if word_count > 20:
            depth_score += 2
        elif word_count > 12:
            depth_score += 1

        # --- Quick fact patterns ---
        quick_patterns = [
            "what is",
            "what are",
            "when was",
            "when did",
            "when is",
            "who is",
            "who was",
            "where is",
            "where was",
            "how much",
            "how many",
            "define",
            "meaning of",
            "list the",
            "name the",
            "give me",
            "show me",
            "is there",
            "does it",
            "do they",
            "what does",
        ]
        quick_score = sum(1 for p in quick_patterns if p in query_lower)

        # Very short queries are almost always quick facts
        # But only if no depth signals were detected (e.g. "explain X" is short but in-depth)
        if word_count <= 6 and depth_score == 0:
            quick_score += 2

        # Demote quick_score when depth signals are present:
        # "can you", "tell me" are conversational fluff, not intent signals.
        # They shouldn't push toward quick_fact when depth patterns already matched.
        if depth_score > 0:
            fluff_patterns = ["can you", "could you", "please"]
            fluff_count = sum(1 for p in fluff_patterns if p in query_lower)
            quick_score = max(0, quick_score - fluff_count)

        # Decision — tie favors in-depth (better to over-explain than under-explain)
        if depth_score >= quick_score:
            return QueryAnalysis(
                intent=QueryIntent.IN_DEPTH,
                confidence=0.7,
                reasoning=f"Heuristic: depth_score={depth_score} > quick_score={quick_score}",
                keywords=query.split()[:5],
                complexity_score=min(0.5 + depth_score * 0.1, 0.9),
            )

        return QueryAnalysis(
            intent=QueryIntent.QUICK_FACT,
            confidence=0.75,
            reasoning=f"Heuristic: quick_score={quick_score} >= depth_score={depth_score}",
            keywords=query.split()[:5],
            complexity_score=max(0.3, min(0.1 + word_count * 0.02, 0.5)),
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
