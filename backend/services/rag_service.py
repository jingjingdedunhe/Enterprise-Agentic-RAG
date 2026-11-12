"""RAG 检索服务。

提供纯向量检索、「向量 + BM25 + RRF」混合检索，以及混合检索 + Reranker
重排序；生成答案留到生成层。
"""

from __future__ import annotations

from backend.retrieval.embedding import embed_query
from backend.retrieval.hybrid_retriever import search_hybrid
from backend.retrieval.reranker import rerank
from backend.retrieval.vector_store import search_similar_chunks


class RAGService:
    """面向业务层的检索入口。"""

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """检索与问题最相关的 Top-K 文本块。

        Args:
            query: 用户问题。
            top_k: 返回条数。

        Returns:
            ``[{"text": str, "metadata": dict, "distance": float}]``，
            按相似度从高到低排列（distance 为余弦距离，越小越相似）。
        """
        query_embedding = embed_query(query)
        return search_similar_chunks(query_embedding, top_k=top_k)

    def retrieve_hybrid(self, query: str, top_k: int = 5) -> list[dict]:
        """向量 + BM25 混合检索（RRF 融合）。

        两路各取 Top-K 候选，使用 RRF（k=60）融合后返回 Top-K。

        Returns:
            ``[{"text", "metadata", "distance", "rrf_score", ...}]``，
            ``distance = 1 - rrf_score``，仅为字段兼容，不表示余弦距离。
        """
        return search_hybrid(query, top_k=top_k)

    def retrieve_hybrid_rerank(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
        score_threshold: float = 0.3,
    ) -> list[dict]:
        """混合检索 + Reranker：RRF 取 Top-20，重排后返回 Top-K。

        Reranker 分数低于 ``score_threshold`` 的候选会被丢弃，因此实际返回
        条数可能少于 ``top_k``。相比 ``retrieve_hybrid`` 多了 CrossEncoder
        推理，延迟会明显增加；该方法适合对精度要求更高的问答场景。
        """
        candidate_k = max(candidate_k, top_k)
        candidates = search_hybrid(
            query, top_k=candidate_k, candidate_k=candidate_k
        )
        return rerank(
            query,
            candidates,
            top_k=top_k,
            score_threshold=score_threshold,
        )
