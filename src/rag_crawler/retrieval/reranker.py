"""
Enhanced Reranker Module for RAG_v5.

Provides FlashRank, Cross-Encoder, and hybrid reranking capabilities for
improving retrieval accuracy. Rerankers work locally without external API calls.

Ported from RAG_v4 and adapted for RAG_v5 architecture.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    from flashrank import Ranker, RerankRequest

    FLASHRANK_AVAILABLE = True
except ImportError:
    FLASHRANK_AVAILABLE = False
    logger.warning("FlashRank not available. Install with: pip install flashrank")

try:
    from sentence_transformers import CrossEncoder

    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    logger.warning(
        "sentence-transformers not available. Install with: pip install sentence-transformers"
    )


class BaseReranker(ABC):
    """Abstract base class for rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Rerank documents based on query relevance.

        Args:
            query: The search query.
            documents: List of documents with 'page_content' or 'content' and 'metadata'.
            top_k: Number of top documents to return.

        Returns:
            List of reranked documents with relevance scores.
        """
        pass

    def _get_content(self, doc: dict[str, Any] | Any) -> str:
        """Extract content from various document formats."""
        if isinstance(doc, dict):
            return doc.get("page_content", doc.get("content", ""))
        elif hasattr(doc, "page_content"):
            return doc.page_content
        elif hasattr(doc, "content"):
            return doc.content
        return str(doc)

    def _get_metadata(self, doc: dict[str, Any] | Any) -> dict[str, Any]:
        """Extract metadata from various document formats."""
        if isinstance(doc, dict):
            return doc.get("metadata", {})
        elif hasattr(doc, "metadata"):
            return doc.metadata if isinstance(doc.metadata, dict) else {}
        return {}


class FlashRankReranker(BaseReranker):
    """
    FlashRank-based reranker for fast local reranking.

    Uses lightweight models for rapid inference without external API calls.
    """

    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        """
        Initialize FlashRank reranker.

        Args:
            model_name: Model to use for reranking.
                Options: "ms-marco-MiniLM-L-12-v2" (default, balanced),
                        "ms-marco-MultiBERT-L-12" (higher accuracy),
                        "ms-marco-TinyBERT-L-2-v2" (fastest)
        """
        if not FLASHRANK_AVAILABLE:
            raise ImportError("FlashRank not available. Install with: pip install flashrank")

        self.model_name = model_name

        try:
            self.ranker = Ranker(model_name=model_name, cache_dir="./.flashrank_cache")
            logger.info(f"FlashRankReranker initialized with model: {model_name}")
        except TypeError:
            try:
                self.ranker = Ranker(model=model_name)
                logger.info(f"FlashRankReranker initialized (legacy API): {model_name}")
            except Exception as e:
                self.ranker = Ranker()
                logger.warning(f"Using default FlashRank model due to: {e}")

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Rerank documents using FlashRank."""
        if not documents:
            return []

        # Prepare passages for FlashRank
        passages = []
        for i, doc in enumerate(documents):
            passages.append(
                {
                    "id": i,
                    "text": self._get_content(doc),
                    "meta": self._get_metadata(doc),
                }
            )

        # Create rerank request
        rerank_request = RerankRequest(query=query, passages=passages)

        # Get reranked results
        results = self.ranker.rerank(rerank_request)

        # Format results with scores
        reranked_docs = []
        for result in results[:top_k]:
            original_doc = documents[result["id"]]
            metadata = self._get_metadata(original_doc)

            enhanced_doc = {
                "page_content": result["text"],
                "content": result["text"],
                "metadata": {
                    **metadata,
                    "relevance_score": result["score"],
                    "rerank_position": len(reranked_docs) + 1,
                },
            }
            reranked_docs.append(enhanced_doc)

        logger.info(f"FlashRank reranked {len(documents)} docs -> top {len(reranked_docs)}")
        return reranked_docs


class CrossEncoderReranker(BaseReranker):
    """
    Cross-Encoder based reranker using sentence-transformers.

    Provides high accuracy reranking with detailed similarity scoring.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        """
        Initialize Cross-Encoder reranker.

        Args:
            model_name: Model to use for reranking.
                Options: "BAAI/bge-reranker-v2-m3" (multilingual),
                        "ms-marco-MiniLM-L-6-v2", "cross-encoder/ms-marco-MiniLM-L-12-v2"
        """
        if not CROSS_ENCODER_AVAILABLE:
            raise ImportError("sentence-transformers not available")

        self.model_name = model_name
        self.model = CrossEncoder(model_name)
        logger.info(f"CrossEncoderReranker initialized with model: {model_name}")

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Rerank documents using Cross-Encoder."""
        if not documents:
            return []

        # Prepare query-document pairs
        pairs = []
        for doc in documents:
            pairs.append([query, self._get_content(doc)])

        # Get similarity scores
        scores = self.model.predict(pairs)

        # Create scored documents
        scored_docs = []
        for i, (doc, score) in enumerate(zip(documents, scores)):
            content = self._get_content(doc)
            metadata = self._get_metadata(doc)

            enhanced_doc = {
                "page_content": content,
                "content": content,
                "metadata": {
                    **metadata,
                    "relevance_score": float(score),
                    "original_position": i,
                },
            }
            scored_docs.append(enhanced_doc)

        # Sort by score (descending) and return top_k
        reranked_docs = sorted(
            scored_docs,
            key=lambda x: x["metadata"]["relevance_score"],
            reverse=True,
        )[:top_k]

        # Add rerank position
        for i, doc in enumerate(reranked_docs):
            doc["metadata"]["rerank_position"] = i + 1

        logger.info(f"CrossEncoder reranked {len(documents)} docs -> top {len(reranked_docs)}")
        return reranked_docs


class HybridReranker(BaseReranker):
    """
    Hybrid reranker that combines multiple reranking strategies.

    Uses both FlashRank and Cross-Encoder for optimal results.
    """

    def __init__(
        self,
        primary_model: str = "flashrank",
        flashrank_model: str = "ms-marco-MiniLM-L-12-v2",
        cross_encoder_model: str = "BAAI/bge-reranker-v2-m3",
        hybrid_weight: float = 0.7,
    ):
        """
        Initialize Hybrid reranker.

        Args:
            primary_model: Primary reranking model ("flashrank" or "cross_encoder").
            flashrank_model: FlashRank model name.
            cross_encoder_model: Cross-Encoder model name.
            hybrid_weight: Weight for primary model (0.0-1.0).
        """
        self.primary_model = primary_model
        self.hybrid_weight = hybrid_weight

        # Initialize available rerankers
        self.rerankers: dict[str, BaseReranker] = {}

        if FLASHRANK_AVAILABLE:
            try:
                self.rerankers["flashrank"] = FlashRankReranker(flashrank_model)
            except Exception as e:
                logger.warning(f"FlashRank initialization failed: {e}")

        if CROSS_ENCODER_AVAILABLE:
            try:
                self.rerankers["cross_encoder"] = CrossEncoderReranker(cross_encoder_model)
            except Exception as e:
                logger.warning(f"CrossEncoder initialization failed: {e}")

        if not self.rerankers:
            raise ImportError("No reranking models available")

        logger.info(
            f"HybridReranker initialized: primary={primary_model}, available={list(self.rerankers.keys())}"
        )

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Rerank documents using hybrid approach."""
        if not documents:
            return []

        # If only one reranker available, use it
        if len(self.rerankers) == 1:
            reranker = list(self.rerankers.values())[0]
            return reranker.rerank(query, documents, top_k)

        # Use hybrid approach if both available
        if "flashrank" in self.rerankers and "cross_encoder" in self.rerankers:
            extended_k = min(len(documents), top_k * 2)

            flashrank_results = self.rerankers["flashrank"].rerank(query, documents, extended_k)
            cross_encoder_results = self.rerankers["cross_encoder"].rerank(
                query, documents, extended_k
            )

            # Combine scores using weighted average
            combined_docs: dict[str, dict] = {}

            for doc in flashrank_results:
                doc_id = doc["page_content"][:200]  # Use first 200 chars as ID
                combined_docs[doc_id] = {
                    "doc": doc,
                    "flashrank_score": doc["metadata"]["relevance_score"],
                    "cross_encoder_score": 0.0,
                }

            for doc in cross_encoder_results:
                doc_id = doc["page_content"][:200]
                if doc_id in combined_docs:
                    combined_docs[doc_id]["cross_encoder_score"] = doc["metadata"][
                        "relevance_score"
                    ]
                else:
                    combined_docs[doc_id] = {
                        "doc": doc,
                        "flashrank_score": 0.0,
                        "cross_encoder_score": doc["metadata"]["relevance_score"],
                    }

            # Calculate combined scores
            final_docs = []
            for doc_data in combined_docs.values():
                primary_score = (
                    doc_data["flashrank_score"]
                    if self.primary_model == "flashrank"
                    else doc_data["cross_encoder_score"]
                )
                secondary_score = (
                    doc_data["cross_encoder_score"]
                    if self.primary_model == "flashrank"
                    else doc_data["flashrank_score"]
                )

                combined_score = (
                    self.hybrid_weight * primary_score + (1 - self.hybrid_weight) * secondary_score
                )

                doc = doc_data["doc"].copy()
                doc["metadata"] = {
                    **doc["metadata"],
                    "combined_relevance_score": combined_score,
                    "flashrank_score": doc_data["flashrank_score"],
                    "cross_encoder_score": doc_data["cross_encoder_score"],
                }
                final_docs.append(doc)

            # Sort by combined score and return top_k
            final_docs.sort(key=lambda x: x["metadata"]["combined_relevance_score"], reverse=True)
            result_docs = final_docs[:top_k]

            for i, doc in enumerate(result_docs):
                doc["metadata"]["rerank_position"] = i + 1

            logger.info(f"Hybrid reranked {len(documents)} docs -> top {len(result_docs)}")
            return result_docs

        # Fallback to primary model
        if self.primary_model in self.rerankers:
            return self.rerankers[self.primary_model].rerank(query, documents, top_k)

        reranker = list(self.rerankers.values())[0]
        return reranker.rerank(query, documents, top_k)


class BasicReranker(BaseReranker):
    """
    Basic reranker using simple text similarity.

    Fallback option when other rerankers are unavailable.
    """

    def __init__(self):
        """Initialize basic reranker."""
        logger.info("BasicReranker initialized (no external dependencies)")

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Rerank documents using basic text similarity."""
        if not documents:
            return []

        scored_docs = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for doc in documents:
            try:
                content = self._get_content(doc)
                content_lower = content.lower()
                content_words = set(content_lower.split())

                # Simple word overlap score
                overlap = len(query_words.intersection(content_words))
                total_query_words = len(query_words)
                score = overlap / total_query_words if total_query_words > 0 else 0.0

                # Boost score if query appears as substring
                if query_lower in content_lower:
                    score += 0.5

                metadata = self._get_metadata(doc)
                scored_doc = {
                    "page_content": content,
                    "content": content,
                    "metadata": {
                        **metadata,
                        "relevance_score": score,
                    },
                }
                scored_docs.append(scored_doc)

            except Exception as e:
                logger.warning(f"Error scoring document: {e}")
                continue

        # Sort by score descending
        scored_docs.sort(key=lambda x: x["metadata"].get("relevance_score", 0), reverse=True)

        result = scored_docs[:top_k]
        for i, doc in enumerate(result):
            doc["metadata"]["rerank_position"] = i + 1

        return result


def create_reranker(reranker_type: str = "hybrid", **kwargs: Any) -> BaseReranker:
    """
    Factory function to create appropriate reranker.

    Args:
        reranker_type: Type of reranker ("flashrank", "cross_encoder", "hybrid", "basic").
        **kwargs: Additional arguments for reranker initialization.

    Returns:
        Initialized reranker instance.
    """
    if reranker_type == "flashrank":
        if not FLASHRANK_AVAILABLE:
            logger.warning("FlashRank not available, falling back to CrossEncoder")
            reranker_type = "cross_encoder"
        else:
            return FlashRankReranker(**kwargs)

    if reranker_type == "cross_encoder":
        if not CROSS_ENCODER_AVAILABLE:
            logger.warning("CrossEncoder not available, falling back to FlashRank")
            reranker_type = "flashrank"
        else:
            return CrossEncoderReranker(**kwargs)

    if reranker_type == "hybrid":
        try:
            return HybridReranker(**kwargs)
        except Exception as e:
            logger.warning(f"Hybrid reranker failed, falling back to basic: {e}")
            return BasicReranker()

    if reranker_type == "basic":
        return BasicReranker()

    # Fallback chain
    try:
        if FLASHRANK_AVAILABLE:
            return FlashRankReranker()
        elif CROSS_ENCODER_AVAILABLE:
            return CrossEncoderReranker()
        else:
            logger.warning("No advanced reranking libraries available, using basic")
            return BasicReranker()
    except Exception as e:
        logger.warning(f"All advanced rerankers failed: {e}")
        return BasicReranker()
