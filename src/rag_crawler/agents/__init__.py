"""
Agents module for RAG_v5 agentic architecture.

Components:
- QueryClassifier: LLM-based query classification
- QuickFactAgent: Fast retrieval for simple questions
- InDepthAgent: Comprehensive analysis with reranking
- ResponseGenerator: Azure OpenAI response generation
"""

from rag_crawler.agents.query_classifier import QueryClassifier, classify_query
from rag_crawler.agents.quick_fact_agent import QuickFactAgent
from rag_crawler.agents.in_depth_agent import InDepthAgent
from rag_crawler.agents.logic_rag_agent import LogicRAGAgent
from rag_crawler.agents.response_generator import ResponseGenerator, generate_response

__all__ = [
    "QueryClassifier",
    "classify_query",
    "QuickFactAgent",
    "InDepthAgent",
    "LogicRAGAgent",
    "ResponseGenerator",
    "generate_response",
]
