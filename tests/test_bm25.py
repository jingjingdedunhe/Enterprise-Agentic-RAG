"""测试 BM25 关键词检索：基线里的前 10 个 Velvera 问题。

用法（项目根目录）：
    C:\\RAG\\P1\\Prag\\Scripts\\python.exe tests\\test_bm25.py
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
from evaluate_baseline import load_queries  # noqa: E402

TOP_K = 5


def main() -> int:
    queries = load_queries()

    print("=" * 92)
    print("BM25 关键词检索测试（Top-5）")
    print(f"参与评估问题数: {len(queries)}")

    started = time.perf_counter()
    index_size = rebuild_index()
    build_ms = (time.perf_counter() - started) * 1000
    print(f"BM25 索引构建完成: {index_size} 个 chunk，耗时 {build_ms:.1f} ms")
    print("=" * 92)

    timings: list[float] = []
    structure_ok = True

    for order, (query_type, raw_query) in enumerate(
        queries.itertuples(index=False, name=None), start=1
    ):
        query = str(raw_query).strip()

        started = time.perf_counter()
        hits = search_bm25(query, top_k=TOP_K)
        elapsed_ms = (time.perf_counter() - started) * 1000
        timings.append(elapsed_ms)

        print("-" * 92)
        print(f"[{order:2d}/{len(queries)}] [{query_type}] {query}")
        print(f"  检索耗时: {elapsed_ms:.1f} ms | 命中 {len(hits)} 条")

        if not hits:
            print("  (没有关键词命中的 chunk)")
            continue

        for rank, hit in enumerate(hits, start=1):
            metadata = hit.get("metadata") or {}
            missing = {"text", "metadata", "distance"} - hit.keys()
            if missing:
                structure_ok = False
            source = metadata.get("source", "unknown")
            page = metadata.get("page", "N/A")
            score = float(hit["score"])
            print(f"    {rank}. {source} | page {page} | BM25 score {score:.4f}")

    print("\n" + "=" * 92)
    if timings:
        total_ms = sum(timings)
        print(
            f"检索耗时: 总计 {total_ms:.1f} ms | "
            f"平均 {total_ms / len(timings):.1f} ms | "
            f"最快 {min(timings):.1f} ms | 最慢 {max(timings):.1f} ms"
        )
    print(f"结果结构校验（text/metadata/distance）: {'通过' if structure_ok else '失败'}")
    return 0 if structure_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
