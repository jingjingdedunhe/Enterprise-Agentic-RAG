"""对比混合 RRF 与「混合 RRF + Reranker」的排序与延迟。

用法（项目根目录）：
    C:\\RAG\\P1\\Prag\\Scripts\\python.exe tests\\test_rerank.py
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

from backend.retrieval.bm25_retriever import rebuild_index  # noqa: E402
from backend.retrieval.reranker import MODEL_NAME, get_reranker  # noqa: E402
from backend.services.rag_service import RAGService  # noqa: E402
from evaluate_baseline import load_queries  # noqa: E402

import torch  # noqa: E402

TOP_K = 5
CANDIDATE_K = 20


def chunk_key(hit: dict) -> str:
    metadata = hit.get("metadata") or {}
    return metadata.get("chunk_id") or f"{metadata.get('source')}|{metadata.get('page')}"


def describe(hit: dict) -> str:
    metadata = hit.get("metadata") or {}
    return f"{metadata.get('source')} | page {metadata.get('page')}"


def timed(callable_obj):
    started = time.perf_counter()
    result = callable_obj()
    return result, (time.perf_counter() - started) * 1000


def print_hybrid(hits: list[dict], elapsed_ms: float) -> None:
    print(f"\n【混合 RRF Top-{len(hits)}】耗时 {elapsed_ms:.1f} ms")
    for rank, hit in enumerate(hits, start=1):
        print(
            f"  {rank}. {describe(hit)} | RRF {float(hit.get('rrf_score', 0.0)):.5f}"
        )


def print_reranked(hits: list[dict], elapsed_ms: float, overhead_ms: float) -> None:
    print(
        f"\n【混合 RRF + Reranker】返回 {len(hits)} 条，耗时 {elapsed_ms:.1f} ms"
        f"（比仅 RRF 多 {overhead_ms:+.1f} ms；阈值 0.3，低于阈值的候选已丢弃）"
    )
    for rank, hit in enumerate(hits, start=1):
        print(
            f"  {rank}. {describe(hit)} | rerank {float(hit.get('rerank_score', 0.0)):.4f}"
            f" | RRF 第 {hit.get('rrf_rank')} 名"
        )


def main() -> int:
    queries = load_queries().head(2)
    rag = RAGService()

    print("=" * 92)
    print("Reranker 对比测试（混合 RRF vs 混合 RRF + Reranker）")
    print(f"Reranker 模型: {MODEL_NAME}")
    print(
        f"PyTorch: {torch.__version__} | CUDA 可用: {torch.cuda.is_available()}"
        f" | 设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}"
    )
    print(f"流程: 混合检索 Top-{CANDIDATE_K} -> Reranker -> Top-{TOP_K}")

    started = time.perf_counter()
    index_size = rebuild_index()
    print(f"BM25 索引: {index_size} 个 chunk，构建 { (time.perf_counter() - started) * 1000:.1f} ms")

    # 预热：向量模型 + Reranker 模型
    rag.retrieve("warmup", top_k=1)
    load_started = time.perf_counter()
    get_reranker()
    load_seconds = time.perf_counter() - load_started
    # 预热推理，排除 CUDA kernel 初始化与 GPU 升频对第一题计时的影响
    warm_pairs = [("warmup question", "warmup passage")] * CANDIDATE_K
    for _ in range(3):
        get_reranker().predict(warm_pairs, batch_size=32, show_progress_bar=False)
    rag.retrieve_hybrid_rerank("warmup", top_k=1)
    print(f"Reranker 模型加载: {load_seconds:.2f} s（不计入下方计时）")
    print("=" * 92)

    hybrid_times: list[float] = []
    rerank_times: list[float] = []
    overheads: list[float] = []

    for order, row in enumerate(queries.itertuples(index=False), start=1):
        query_type, raw_query = row
        query = str(raw_query).strip()

        print(f"\n第 {order} 题 [{query_type}]: {query}")

        hybrid_hits, hybrid_ms = timed(
            lambda: rag.retrieve_hybrid(query, top_k=TOP_K)
        )
        rerank_hits, rerank_ms = timed(
            lambda: rag.retrieve_hybrid_rerank(query, top_k=TOP_K)
        )
        overhead = rerank_ms - hybrid_ms

        hybrid_times.append(hybrid_ms)
        rerank_times.append(rerank_ms)
        overheads.append(overhead)

        print_hybrid(hybrid_hits, hybrid_ms)
        print_reranked(rerank_hits, rerank_ms, overhead)

        hybrid_ids = {chunk_key(hit) for hit in hybrid_hits}
        rerank_ids = {chunk_key(hit) for hit in rerank_hits}

        entered = [hit for hit in rerank_hits if chunk_key(hit) not in hybrid_ids]
        dropped = [hit for hit in hybrid_hits if chunk_key(hit) not in rerank_ids]

        print("\n  变化:")
        if entered:
            for hit in entered:
                print(
                    f"    + 新进入 Top-{TOP_K}: {describe(hit)}"
                    f"（RRF 第 {hit.get('rrf_rank')} 名）"
                )
        if dropped:
            for hit in dropped:
                print(f"    - 被挤出 Top-{TOP_K}: {describe(hit)}")
        if not entered and not dropped:
            print("    Top-5 集合无变化（顺序可能变化）")

    print("\n" + "=" * 92)
    print("延迟汇总:")
    for index in range(len(hybrid_times)):
        print(
            f"  第 {index + 1} 题: 混合 RRF {hybrid_times[index]:.1f} ms"
            f" -> +Reranker {rerank_times[index]:.1f} ms"
            f"（额外 {overheads[index]:+.1f} ms）"
        )
    print(
        f"  平均: RRF {sum(hybrid_times) / len(hybrid_times):.1f} ms"
        f" | RRF+Reranker {sum(rerank_times) / len(rerank_times):.1f} ms"
        f" | 额外 {sum(overheads) / len(overheads):+.1f} ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
