"""
In-Depth Agent for RAG_v5.

Handles complex questions requiring comprehensive analysis,
using extended retrieval with reranking.
"""

import logging
from typing import Any

from rag_crawler.retrieval.vector_retriever import VectorRetriever, create_retriever
from rag_crawler.retrieval.reranker import BaseReranker, create_reranker
from rag_crawler.router.state import RAGState, documents_to_state_format

logger = logging.getLogger(__name__)


class InDepthAgent:
    """
    Agent for complex, in-depth questions.

    Uses extended vector retrieval with reranking for comprehensive responses.
    Generates detailed, analytical answers.
    """

    def __init__(
        self,
        retriever: VectorRetriever | None = None,
        reranker: BaseReranker | None = None,
        retrieval_top_k: int = 10,
        final_top_k: int = 5,
        threshold: float = 0.3,
    ):
        """
        Initialize In-Depth Agent.

        Args:
            retriever: VectorRetriever instance.
            reranker: Reranker instance (created if not provided).
            retrieval_top_k: Number of documents to retrieve initially.
            final_top_k: Number of documents after reranking.
            threshold: Minimum similarity threshold.
        """
        self.retriever = retriever or create_retriever(
            default_top_k=retrieval_top_k,
            default_threshold=threshold,
        )
        self.reranker = reranker or self._create_reranker()
        self.retrieval_top_k = retrieval_top_k
        self.final_top_k = final_top_k
        self.threshold = threshold

        logger.info(
            f"InDepthAgent initialized: retrieval_k={retrieval_top_k}, final_k={final_top_k}"
        )

    def _create_reranker(self) -> BaseReranker:
        """Create reranker with fallback."""
        try:
            return create_reranker("hybrid")
        except Exception as e:
            logger.warning(f"Hybrid reranker unavailable, using basic: {e}")
            return create_reranker("basic")

    async def process(self, state: RAGState) -> RAGState:
        """
        Process query through In-Depth pipeline.

        Args:
            state: Current RAG state with query.

        Returns:
            Updated state with retrieved and reranked documents.
        """
        query = state.get("query", "")

        if not query:
            logger.warning("InDepthAgent received empty query")
            state["error"] = "Empty query"
            return state

        logger.info(f"InDepthAgent processing: {query[:50]}...")

        try:
            # Get source filter from state (for "New document" mode)
            source_filter = state.get("source_filter")

            # Extended vector retrieval
            docs = await self.retriever.retrieve(
                query=query,
                top_k=self.retrieval_top_k,
                threshold=self.threshold,
                source_filter=source_filter,
            )

            state["retrieved_documents"] = documents_to_state_format(docs)
            logger.info(f"InDepthAgent retrieved {len(docs)} documents")

            # Apply reranking
            if docs and self.reranker:
                doc_dicts = [
                    {
                        "page_content": doc.content,
                        "content": doc.content,
                        "metadata": {
                            **doc.metadata,
                            "document_title": doc.document_title,
                            "document_source": doc.document_source,
                            "chunk_id": doc.chunk_id,
                            "document_id": doc.document_id,
                            "chunk_index": doc.chunk_index,
                            "original_score": doc.score,
                        },
                    }
                    for doc in docs
                ]

                reranked = self.reranker.rerank(query, doc_dicts, self.final_top_k)

                # Convert back to state format
                reranked_docs = []
                for doc in reranked:
                    metadata = doc.get("metadata", {})
                    reranked_docs.append(
                        {
                            "chunk_id": metadata.get("chunk_id", ""),
                            "document_id": metadata.get("document_id", ""),
                            "content": doc.get("content", doc.get("page_content", "")),
                            "score": metadata.get(
                                "relevance_score", metadata.get("original_score", 0.0)
                            ),
                            "document_title": metadata.get("document_title", ""),
                            "document_source": metadata.get("document_source", ""),
                            "chunk_index": metadata.get("chunk_index", 0),
                            "metadata": metadata,
                        }
                    )

                state["reranked_documents"] = reranked_docs
                logger.info(f"InDepthAgent reranked to {len(reranked_docs)} documents")
            else:
                state["reranked_documents"] = state["retrieved_documents"][: self.final_top_k]

            state["current_agent"] = "in_depth"
            state["step_count"] = state.get("step_count", 0) + 1

        except Exception as e:
            logger.error(f"InDepthAgent processing failed: {e}")
            state["error"] = str(e)
            state["retrieved_documents"] = []
            state["reranked_documents"] = []

        return state

    async def __call__(self, state: RAGState) -> RAGState:
        """Allow agent to be called directly."""
        return await self.process(state)


# Response prompt for in-depth analysis
IN_DEPTH_PROMPT = """You are a knowledgeable assistant providing a comprehensive, analytical response.

Based on the provided context, give a thorough answer that covers multiple aspects of the question.

Guidelines:
- Provide a well-structured response (300-600 words)
- Cover multiple perspectives or aspects when relevant
- Use citations [1], [2], etc. to reference sources
- Include analysis, not just facts
- If comparing, use clear structure
- If explaining a process, be thorough but clear
- Acknowledge limitations in the available information

{framework}

Context:
{context}

Question: {query}

Comprehensive Answer:"""

ANALYSIS_FRAMEWORK = """Use this structure for your analysis:
1. **Overview**: Brief summary of the key points
2. **Details**: In-depth explanation with evidence
3. **Analysis**: Critical examination and implications
4. **Conclusion**: Summary and key takeaways
"""


def get_in_depth_prompt(
    query: str,
    documents: list[dict[str, Any]],
    use_framework: bool = True,
) -> str:
    """
    Generate prompt for in-depth response.

    Args:
        query: User's question.
        documents: Retrieved/reranked documents.
        use_framework: Whether to include analysis framework.

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
            source = doc.get("document_source", "")
            # Include more content for in-depth analysis
            context_parts.append(f"[{i}] {title} ({source}):\n{content[:1000]}")
        context = "\n\n".join(context_parts)

    framework = ANALYSIS_FRAMEWORK if use_framework else ""

    return IN_DEPTH_PROMPT.format(framework=framework, context=context, query=query)
