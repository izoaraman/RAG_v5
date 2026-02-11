"""
LogicRAG-inspired multi-hop reasoning agent for RAG_v5.
"""

import asyncio
import json
import logging
from collections import defaultdict, deque
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from rag_crawler.retrieval.vector_retriever import VectorRetriever, create_retriever
from rag_crawler.retrieval.reranker import BaseReranker, create_reranker
from rag_crawler.retrieval.rolling_memory import RollingMemory
from rag_crawler.router.state import RAGState, documents_to_state_format
from rag_crawler.utils.providers import get_shared_llm

logger = logging.getLogger(__name__)


class LogicRAGAgent:
    """
    Multi-hop reasoning agent inspired by LogicRAG (AAAI'26).

    Decomposes complex queries into subproblems, builds a DAG of dependencies,
    performs topological sort, then iteratively retrieves and reasons with
    rolling memory and early termination.

    Enhancements over base LogicRAG:
    - Summary-level warm-up retrieval for broad document context
    - Rank-based parallel retrieval for independent subproblems
    - Graph pruning: merges same-rank subproblems into unified queries
    - Document-diversity sampling to avoid over-retrieving from one source
    """

    def __init__(
        self,
        retriever: VectorRetriever | None = None,
        reranker: BaseReranker | None = None,
        rolling_memory: RollingMemory | None = None,
        max_rounds: int = 5,
        retrieval_top_k: int = 5,
        rerank_top_k: int = 3,
        threshold: float = 0.3,
    ):
        self.retriever = retriever or create_retriever(
            default_top_k=retrieval_top_k,
            default_threshold=threshold,
        )
        self.reranker = reranker or self._create_reranker()
        self.rolling_memory = rolling_memory or RollingMemory()

        self.max_rounds = max_rounds
        self.retrieval_top_k = retrieval_top_k
        self.rerank_top_k = rerank_top_k
        self.threshold = threshold

        self.llm = get_shared_llm(temperature=0.0)

        logger.info(
            "LogicRAGAgent initialized: "
            f"max_rounds={max_rounds}, retrieval_k={retrieval_top_k}, rerank_k={rerank_top_k}"
        )

    def _create_reranker(self) -> BaseReranker:
        """Create reranker with fallback."""
        try:
            return create_reranker("hybrid")
        except Exception as e:
            logger.warning(f"Hybrid reranker unavailable, using basic: {e}")
            return create_reranker("basic")

    async def _decompose_query(self, question: str, info_summary: str) -> dict[str, Any]:
        """
        Decompose question into subproblems and identify dependencies.

        Returns:
            {"subproblems": list[str], "dependency_pairs": list[list[int]]}
        """
        messages = [
            SystemMessage(
                content=(
                    "You decompose complex questions into minimal logical subproblems and identify "
                    "dependencies. Return strict JSON only."
                )
            ),
            HumanMessage(
                content=(
                    "Decompose this question into clear subproblems and dependency pairs.\n\n"
                    f"Question: {question}\n\n"
                    "Current information summary (may be partial):\n"
                    f"{info_summary}\n\n"
                    "Return JSON exactly like:\n"
                    "{\n"
                    '  "subproblems": ["...", "..."],\n'
                    '  "dependency_pairs": [[0,1], [1,2]]\n'
                    "}\n"
                    "Where [a,b] means subproblem b depends on subproblem a."
                )
            ),
        ]

        fallback = {"subproblems": [question], "dependency_pairs": []}

        try:
            response = await self.llm.ainvoke(messages)
            response_content = response.content
            response_text = (
                response_content.strip()
                if isinstance(response_content, str)
                else str(response_content).strip()
            )
            if response_text.startswith("```"):
                response_text = response_text.split("```", maxsplit=2)[1].strip()
                if response_text.startswith("json"):
                    response_text = response_text[4:].strip()

            parsed = json.loads(response_text)
            subproblems = parsed.get("subproblems", [])
            dependency_pairs = parsed.get("dependency_pairs", [])

            if not isinstance(subproblems, list) or not subproblems:
                return fallback

            return {
                "subproblems": [str(s) for s in subproblems],
                "dependency_pairs": dependency_pairs if isinstance(dependency_pairs, list) else [],
            }
        except Exception as e:
            logger.warning(f"Query decomposition failed, using fallback: {e}")
            return fallback

    @staticmethod
    def _topological_sort(subproblems: list[str], dependency_pairs: list[list[int]]) -> list[str]:
        """DFS-based topological sort. Returns subproblems in dependency order."""
        n = len(subproblems)
        if n <= 1:
            return subproblems

        graph: dict[int, list[int]] = defaultdict(list)
        for pair in dependency_pairs:
            if not isinstance(pair, list) or len(pair) != 2:
                continue
            src, dst = pair
            if not isinstance(src, int) or not isinstance(dst, int):
                continue
            if src < 0 or dst < 0 or src >= n or dst >= n:
                continue
            graph[src].append(dst)

        visited = [0] * n
        order: list[int] = []
        has_cycle = False

        def dfs(node: int) -> None:
            nonlocal has_cycle
            if has_cycle:
                return
            if visited[node] == 1:
                has_cycle = True
                return
            if visited[node] == 2:
                return

            visited[node] = 1
            for nei in graph.get(node, []):
                dfs(nei)
            visited[node] = 2
            order.append(node)

        for i in range(n):
            if visited[i] == 0:
                dfs(i)

        if has_cycle:
            logger.warning("Dependency cycle detected; using original subproblem order")
            return subproblems

        order.reverse()
        return [subproblems[i] for i in order]

    @staticmethod
    def _compute_ranks(
        subproblems: list[str], dependency_pairs: list[list[int]]
    ) -> list[list[str]]:
        """
        Compute rank-based groups from DAG for parallel execution.

        Uses Kahn's algorithm (BFS) with rank tracking. Returns list of lists,
        where each inner list contains subproblems at the same rank
        (can be executed in parallel).
        """
        n = len(subproblems)
        if n <= 1:
            return [subproblems] if subproblems else []

        # Build adjacency and in-degree
        graph: dict[int, list[int]] = defaultdict(list)
        in_degree = [0] * n
        for pair in dependency_pairs:
            if not isinstance(pair, list) or len(pair) != 2:
                continue
            src, dst = pair
            if not isinstance(src, int) or not isinstance(dst, int):
                continue
            if src < 0 or dst < 0 or src >= n or dst >= n:
                continue
            graph[src].append(dst)
            in_degree[dst] += 1

        # BFS level assignment (Kahn's algorithm with rank tracking)
        queue: deque[int] = deque()
        for i in range(n):
            if in_degree[i] == 0:
                queue.append(i)

        ranks: list[list[str]] = []
        visited_count = 0

        while queue:
            rank_group: list[str] = []
            for _ in range(len(queue)):
                node = queue.popleft()
                rank_group.append(subproblems[node])
                visited_count += 1
                for neighbor in graph.get(node, []):
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
            ranks.append(rank_group)

        # If cycle detected (not all visited), append remaining
        if visited_count < n:
            remaining = [subproblems[i] for i in range(n) if in_degree[i] > 0]
            if remaining:
                ranks.append(remaining)

        return ranks

    async def _merge_subproblems(self, subproblems: list[str]) -> str:
        """Merge multiple independent subproblems into a single query for efficiency."""
        if len(subproblems) == 1:
            return subproblems[0]

        if len(subproblems) == 2:
            return f"{subproblems[0]} AND {subproblems[1]}"

        # For 3+, use LLM to create a concise unified query
        messages = [
            SystemMessage(
                content=(
                    "Merge multiple related questions into one concise query. "
                    "Return only the merged query, nothing else."
                )
            ),
            HumanMessage(
                content=(
                    "Merge these questions into one query:\n"
                    + "\n".join(f"- {sp}" for sp in subproblems)
                )
            ),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            content = response.content
            merged = content.strip() if isinstance(content, str) else str(content).strip()
            logger.info(f"Merged {len(subproblems)} subproblems into: {merged[:80]}...")
            return merged
        except Exception as e:
            logger.warning(f"Subproblem merge failed, using concatenation: {e}")
            return " AND ".join(subproblems)

    @staticmethod
    def _apply_diversity_filter(
        docs: list[dict[str, Any]],
        seen_doc_counts: dict[str, int],
        max_per_doc: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Filter docs to ensure diversity across documents.

        Limits chunks from any single document to max_per_doc per round,
        considering previously seen counts.
        """
        if not docs:
            return docs

        filtered: list[dict[str, Any]] = []
        round_doc_counts: dict[str, int] = {}

        for doc in docs:
            doc_id = doc.get("document_id", "")
            if not doc_id:
                filtered.append(doc)
                continue

            total_count = seen_doc_counts.get(doc_id, 0) + round_doc_counts.get(doc_id, 0)
            if total_count < max_per_doc:
                filtered.append(doc)
                round_doc_counts[doc_id] = round_doc_counts.get(doc_id, 0) + 1

        if not filtered and docs:
            # If filtering removed everything, keep top doc by score
            filtered = [docs[0]]

        return filtered

    async def _retrieve_and_rerank(
        self,
        query: str,
        source_filter: str | None = None,
        exclude_chunks: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve and rerank for a single subproblem. Filters already-seen chunks."""
        docs = await self.retriever.retrieve(
            query=query,
            top_k=self.retrieval_top_k,
            threshold=self.threshold,
            source_filter=source_filter,
        )

        excluded = exclude_chunks or set()
        filtered_docs = [doc for doc in docs if doc.chunk_id not in excluded]
        if not filtered_docs:
            return []

        if self.reranker:
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
                for doc in filtered_docs
            ]
            reranked = self.reranker.rerank(query, doc_dicts, self.rerank_top_k)

            reranked_docs: list[dict[str, Any]] = []
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
            return reranked_docs

        return documents_to_state_format(filtered_docs)[: self.rerank_top_k]

    async def process(self, state: RAGState) -> RAGState:
        """Full LogicRAG pipeline with parallel rank-based retrieval."""
        query = state.get("query", "")

        if not query:
            logger.warning("LogicRAGAgent received empty query")
            state["error"] = "Empty query"
            return state

        logger.info(f"LogicRAGAgent processing: {query[:50]}...")

        try:
            source_filter = state.get("source_filter")

            # Stage 1a: Summary-level warm-up for broad document context
            summary_contexts: list[str] = []
            try:
                summaries = await self.retriever.retrieve_summaries(
                    query=query,
                    top_k=3,
                    threshold=self.threshold,
                    source_filter=source_filter,
                )
                summary_contexts = [
                    f"[Document: {s['document_title']}] {s['summary']}" for s in summaries
                ]
                logger.info(f"Retrieved {len(summaries)} document summaries for warm-up")
            except Exception as e:
                logger.warning(f"Summary warm-up failed, using chunk-only: {e}")

            # Stage 1b: Chunk-level warm-up retrieval
            warm_docs = await self.retriever.retrieve(
                query=query,
                top_k=self.retrieval_top_k,
                threshold=self.threshold,
                source_filter=source_filter,
            )
            all_docs_state = documents_to_state_format(warm_docs)
            seen_chunk_ids: set[str] = {doc.chunk_id for doc in warm_docs}
            seen_doc_counts: dict[str, int] = {}
            for doc in warm_docs:
                seen_doc_counts[doc.document_id] = seen_doc_counts.get(doc.document_id, 0) + 1

            # Combine summary + chunk contexts for initial rolling memory
            warm_contexts = summary_contexts + [doc.content for doc in warm_docs]
            info_summary = await self.rolling_memory.create_initial_summary(query, warm_contexts)

            sufficiency = await self.rolling_memory.check_sufficiency(query, info_summary)
            round_count = 0

            subproblems: list[str] = []
            dependency_pairs: list[list[int]] = []
            sorted_subproblems: list[str] = []
            last_round_docs = documents_to_state_format(warm_docs)[: self.rerank_top_k]

            if not sufficiency.get("can_answer", False):
                # Stage 2: Decomposition
                decomposition = await self._decompose_query(query, info_summary)
                subproblems = decomposition.get("subproblems", [])
                dependency_pairs = decomposition.get("dependency_pairs", [])
                sorted_subproblems = self._topological_sort(subproblems, dependency_pairs)

                # Compute rank-based groups for parallel execution
                ranks = self._compute_ranks(subproblems, dependency_pairs)
                logger.info(
                    f"Decomposed into {len(subproblems)} subproblems across {len(ranks)} ranks"
                )

                # Stage 3: Rank-based iterative retrieval (parallel within ranks)
                for rank_group in ranks:
                    if round_count >= self.max_rounds:
                        break

                    # Graph pruning: merge if rank has 3+ subproblems
                    if len(rank_group) >= 3:
                        merged_query = await self._merge_subproblems(rank_group)
                        queries_to_retrieve = [merged_query]
                    else:
                        queries_to_retrieve = rank_group

                    # Parallel retrieval for all queries in this rank
                    retrieval_tasks = [
                        self._retrieve_and_rerank(
                            query=q,
                            source_filter=source_filter,
                            exclude_chunks=seen_chunk_ids,
                        )
                        for q in queries_to_retrieve
                    ]
                    rank_results = await asyncio.gather(*retrieval_tasks)

                    # Collect results from all parallel retrievals
                    rank_docs: list[dict[str, Any]] = []
                    for round_docs in rank_results:
                        if round_docs:
                            rank_docs.extend(round_docs)

                    round_count += 1

                    if not rank_docs:
                        continue

                    # Apply document diversity filtering
                    rank_docs = self._apply_diversity_filter(rank_docs, seen_doc_counts)

                    last_round_docs = rank_docs
                    for doc in rank_docs:
                        chunk_id = doc.get("chunk_id", "")
                        doc_id = doc.get("document_id", "")
                        if chunk_id:
                            seen_chunk_ids.add(chunk_id)
                        if doc_id:
                            seen_doc_counts[doc_id] = seen_doc_counts.get(doc_id, 0) + 1

                    all_docs_state.extend(rank_docs)

                    new_contexts = [
                        doc.get("content", "") for doc in rank_docs if doc.get("content")
                    ]
                    info_summary = await self.rolling_memory.refine_summary(
                        query,
                        info_summary,
                        new_contexts,
                    )

                    sufficiency = await self.rolling_memory.check_sufficiency(
                        query,
                        info_summary,
                        subproblems=subproblems,
                    )

                    if sufficiency.get("can_answer", False):
                        logger.info("LogicRAG early stop: sufficient information gathered")
                        break

            state["subproblems"] = subproblems
            state["dependency_pairs"] = dependency_pairs
            state["sorted_subproblems"] = sorted_subproblems
            state["info_summary"] = info_summary
            state["round_count"] = round_count
            state["max_rounds"] = self.max_rounds
            state["retrieved_documents"] = all_docs_state
            state["reranked_documents"] = last_round_docs
            state["current_agent"] = "logic_rag"
            state["step_count"] = state.get("step_count", 0) + 1

        except Exception as e:
            logger.error(f"LogicRAGAgent processing failed: {e}")
            state["error"] = str(e)
            state["retrieved_documents"] = []
            state["reranked_documents"] = []

        return state

    async def __call__(self, state: RAGState) -> RAGState:
        """Allow agent to be called directly."""
        return await self.process(state)
