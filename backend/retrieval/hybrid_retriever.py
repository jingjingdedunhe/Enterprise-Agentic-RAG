"""混合检索：向量检索 + BM25，使用 Reciprocal Rank Fusion（RRF）融合。

RRF 公式：``score(d) = Σ 1 / (k + rank_i(d))``，常数 k 默认 60。
两路检索各取 Top-N 候选，按 chunk_id 去重后融合，返回最终 Top-K。
"""

from __future__ import annotations

from backend.retrieval.bm25_retriever import search_bm25
from backend.retrieval.embedding import embed_query
from backend.retrieval.vector_store import search_similar_chunks

RRF_K = 60


def _chunk_key(hit: dict) -> str:
    """用 chunk_id 去重；缺失时退回 source/page/text。"""
    metadata = hit.get("metadata") or {}
    chunk_id = metadata.get("chunk_id")
    if chunk_id:
        return str(chunk_id)
    return f"{metadata.get('source')}|{metadata.get('page')}|{hit.get('text', '')}"


def rrf_fuse(
    vector_hits: list[dict],
    bm25_hits: list[dict],
    top_k: int = 5,
    k: int = RRF_K,
) -> list[dict]:
    """按 RRF 融合两路检索结果。

    Args:
        vector_hits: 向量检索结果（按相似度降序）。
        bm25_hits: BM25 检索结果（按得分降序）。
        top_k: 最终返回条数。
        k: RRF 平滑常数。

    Returns:
        融合结果，按 RRF 得分降序；每条包含 ``text``、``metadata``、
        ``distance``（= 1 - rrf_score，仅为字段兼容，不具余弦含义）、
        ``rrf_score``、``vector_rank``、``bm25_rank``、``matched_by``。
    """
    if top_k <= 0:
        return []

    scores: dict[str, float] = {}
    payload: dict[str, dict] = {}
    ranks: dict[str, dict[str, int]] = {}

    for source_name, hits in (("vector", vector_hits), ("bm25", bm25_hits)):
        for rank, hit in enumerate(hits, start=1):
            key = _chunk_key(hit)
            payload.setdefault(key, hit)
            ranks.setdefault(key, {})[source_name] = rank
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    # 得分降序；同分时按 chunk_id 排序，保证结果稳定可复现
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))

    results: list[dict] = []
    for key, score in ordered[:top_k]:
        hit = payload[key]
        rank_info = ranks[key]
        results.append(
            {
                "text": hit.get("text", ""),
                "metadata": dict(hit.get("metadata") or {}),
                "distance": 1.0 - score,
                "rrf_score": score,
                "vector_rank": rank_info.get("vector"),
                "bm25_rank": rank_info.get("bm25"),
                "matched_by": [
                    name for name in ("vector", "bm25") if name in rank_info
                ],
            }
        )

    return results


def search_hybrid(
    query: str,
    top_k: int = 5,
    candidate_k: int | None = None,
    rrf_k: int = RRF_K,
) -> list[dict]:
    """向量 + BM25 混合检索，返回 RRF 融合后的 Top-K。

    候选深度默认等于 ``top_k``（两路各取 Top-K 融合）。显式传入更大的
    ``candidate_k`` 可以做更深候选融合，但向量侧的弱相关文档会因“两路都
    出现”而在 RRF 中累积分数，可能挤掉只在单路排名靠前的强命中，需按
    实际语料权衡。

    Args:
        query: 用户问题。
        top_k: 最终返回条数。
        candidate_k: 每路检索的候选条数；默认等于 ``top_k``。
        rrf_k: RRF 平滑常数。
    """
    if not isinstance(query, str) or not query.strip() or top_k <= 0:
        return []

    candidate_k = max(candidate_k or top_k, top_k)

    query_embedding = embed_query(query)
    vector_hits = search_similar_chunks(query_embedding, top_k=candidate_k)
    bm25_hits = search_bm25(query, top_k=candidate_k)

    if not vector_hits and not bm25_hits:
        return []

    return rrf_fuse(vector_hits, bm25_hits, top_k=top_k, k=rrf_k)
