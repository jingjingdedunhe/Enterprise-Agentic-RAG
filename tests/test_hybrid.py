"""测试混合检索（向量 + BM25 + RRF）。

用法（项目根目录）：
    C:\\RAG\\P1\\Prag\\Scripts\\python.exe tests\\test_hybrid.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.retrieval.bm25_retriever import rebuild_index, search_bm25  # noqa: E402
from backend.services.rag_service import RAGService  # noqa: E402
from evaluate_baseline import load_queries  # noqa: E402

TOP_K = 5
TARGET_SOURCE = "environmental_policy.pdf"
CN_QUERY = "Velvera 怎么参与以旧换新（Trade-In）计划？"


def format_hit(rank: int, hit: dict, scheme: str) -> str:
    metadata = hit.get("metadata") or {}
    source = metadata.get("source", "unknown")
    page = metadata.get("page", "N/A")

    if scheme == "vector":
        extra = f"distance {float(hit['distance']):.4f}"
    elif scheme == "bm25":
        extra = f"BM25 score {float(hit['score']):.4f}"
    else:
        extra = (
            f"RRF {float(hit['rrf_score']):.5f} "
            f"(V={hit.get('vector_rank')}, B={hit.get('bm25_rank')})"
        )

    return f"    {rank}. {source} | page {page} | {extra}"


def run_vector(rag: RAGService, query: str) -> tuple[list[dict], float]:
    started = time.perf_counter()
    hits = rag.retrieve(query, top_k=TOP_K)
    return hits, (time.perf_counter() - started) * 1000


def run_bm25(query: str) -> tuple[list[dict], float]:
    started = time.perf_counter()
    hits = search_bm25(query, top_k=TOP_K)
    return hits, (time.perf_counter() - started) * 1000


def run_hybrid(rag: RAGService, query: str) -> tuple[list[dict], float]:
    started = time.perf_counter()
    hits = rag.retrieve_hybrid(query, top_k=TOP_K)
    return hits, (time.perf_counter() - started) * 1000


def print_scheme(name: str, hits: list[dict], elapsed_ms: float, scheme: str) -> None:
    print(f"\n【{name}】耗时 {elapsed_ms:.1f} ms | 命中 {len(hits)} 条")
    for rank, hit in enumerate(hits, start=1):
        print(format_hit(rank, hit, scheme))


def source_rank(hits: list[dict], target: str) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        if (hit.get("metadata") or {}).get("source") == target:
            return rank
    return None


def main() -> int:
    queries = load_queries()
    first_query = str(queries.iloc[0]["Query"]).strip()
    first_type = queries.iloc[0]["Query Type"]

    print("=" * 92)
    print("混合检索测试：向量 + BM25 + RRF（k=60，候选 N=Top-K）")

    started = time.perf_counter()
    index_size = rebuild_index()
    build_ms = (time.perf_counter() - started) * 1000
    print(f"BM25 索引: {index_size} 个 chunk，构建耗时 {build_ms:.1f} ms")

    rag = RAGService()

    # 预热向量模型，避免第一条查询的计时包含模型加载
    warm_started = time.perf_counter()
    rag.retrieve("warmup", top_k=1)
    warm_ms = (time.perf_counter() - warm_started) * 1000
    print(f"向量模型预热: {warm_ms:.1f} ms（不计入下方计时）")
    print("=" * 92)

    # ---------- 第 1 题：三种方案对比 ----------
    print(f"\n第 1 题 [{first_type}]: {first_query}")

    vector_hits, vector_ms = run_vector(rag, first_query)
    bm25_hits, bm25_ms = run_bm25(first_query)
    hybrid_hits, hybrid_ms = run_hybrid(rag, first_query)

    print_scheme("仅向量", vector_hits, vector_ms, "vector")
    print_scheme("仅 BM25", bm25_hits, bm25_ms, "bm25")
    print_scheme("混合 RRF", hybrid_hits, hybrid_ms, "hybrid")

    # ---------- 中文查询：Trade-In ----------
    print("\n" + "=" * 92)
    print(f"核心测试（中文查询）: {CN_QUERY}")
    print("=" * 92)

    cn_vector_hits, cn_vector_ms = run_vector(rag, CN_QUERY)
    cn_bm25_hits, cn_bm25_ms = run_bm25(CN_QUERY)
    cn_hybrid_hits, cn_hybrid_ms = run_hybrid(rag, CN_QUERY)

    print_scheme("仅向量", cn_vector_hits, cn_vector_ms, "vector")
    print_scheme("仅 BM25", cn_bm25_hits, cn_bm25_ms, "bm25")
    print_scheme("混合 RRF", cn_hybrid_hits, cn_hybrid_ms, "hybrid")

    vector_rank = source_rank(cn_vector_hits, TARGET_SOURCE)
    bm25_rank = source_rank(cn_bm25_hits, TARGET_SOURCE)
    hybrid_rank = source_rank(cn_hybrid_hits, TARGET_SOURCE)

    def rank_text(rank: int | None) -> str:
        return f"第 {rank} 名" if rank else "未命中"

    print("\n" + "-" * 92)
    print(f"目标文档 {TARGET_SOURCE} 的位置：")
    print(f"  仅向量: {rank_text(vector_rank)}")
    print(f"  仅 BM25: {rank_text(bm25_rank)}")
    print(f"  混合 RRF: {rank_text(hybrid_rank)}")

    if hybrid_rank is None:
        print(f"\n❌ 混合检索 Top-{TOP_K} 未包含 {TARGET_SOURCE}")
        return 1

    print(f"\n✅ 混合检索 Top-{TOP_K} 成功包含 {TARGET_SOURCE}（第 {hybrid_rank} 名）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
