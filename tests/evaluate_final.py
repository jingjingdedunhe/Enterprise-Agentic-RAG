"""任务 2.6：四种检索方案的全量对比实验。

A. 仅向量
B. 仅 BM25
C. 混合 RRF
D. 混合 RRF + Reranker（阈值 0.3）

评估标准来自 CSV 的 Supporting Facts：把 .md 文件名映射为 .pdf，
统计每个方案 Top-5 中来源正确的条数、命中率与平均耗时。

用法（项目根目录）：
    C:\\RAG\\P1\\Prag\\Scripts\\python.exe tests\\evaluate_final.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd  # noqa: E402

from backend.retrieval.bm25_retriever import rebuild_index, search_bm25  # noqa: E402
from backend.retrieval.reranker import MODEL_NAME, get_reranker  # noqa: E402
from backend.services.rag_service import RAGService  # noqa: E402

CSV_PATH = BASE_DIR / "data" / "Dataset categories - queries.csv"
CATEGORY = "Velvera Technologies"
LIMIT = 10
TOP_K = 5
SCORE_THRESHOLD = 0.3


def load_rows() -> list[dict]:
    """读取前 10 个问题，并解析每题的期望来源（.pdf）。"""
    df = pd.read_csv(CSV_PATH, encoding="gb18030")
    df.columns = [str(column).strip() for column in df.columns]

    mask = (
        df["Enterprise Name"].astype(str).str.strip().str.casefold()
        == CATEGORY.casefold()
    )
    subset = df.loc[mask].head(LIMIT)

    rows: list[dict] = []
    for _, row in subset.iterrows():
        raw_facts = row.get("Supporting Facts")
        facts: list[dict] = []
        if isinstance(raw_facts, str) and raw_facts.strip():
            try:
                parsed = json.loads(raw_facts)
                if isinstance(parsed, list):
                    facts = parsed
            except json.JSONDecodeError:
                facts = []

        expected = {
            Path(str(fact.get("filename", ""))).name
            for fact in facts
            if fact.get("filename")
        }
        expected = {
            Path(filename).with_suffix(".pdf").name for filename in expected
        }

        rows.append(
            {
                "query_type": str(row["Query Type"]),
                "query": str(row["Query"]).strip(),
                "expected": expected,
            }
        )
    return rows


def count_correct(hits: list[dict], expected: set[str]) -> int:
    return sum(
        1
        for hit in hits
        if (hit.get("metadata") or {}).get("source") in expected
    )


def timed(callable_obj):
    started = time.perf_counter()
    result = callable_obj()
    return result, (time.perf_counter() - started) * 1000


def main() -> int:
    rows = load_rows()
    rag = RAGService()

    print("=" * 100)
    print("任务 2.6 四种检索方案对比（Top-5）")
    print("A 仅向量 | B 仅 BM25 | C 混合 RRF | D 混合 RRF + Reranker")
    print(f"Reranker: {MODEL_NAME} | 阈值: {SCORE_THRESHOLD}（低于阈值的候选丢弃，D 可能少于 5 条）")
    print("正确条数 = 返回结果中 source 属于该题 Supporting Facts 的条数")

    started = time.perf_counter()
    index_size = rebuild_index()
    print(f"BM25 索引: {index_size} chunks，构建 { (time.perf_counter() - started) * 1000:.1f} ms")

    # 预热：向量模型、Reranker 模型、完整 D 流程
    rag.retrieve("warmup", top_k=1)
    load_started = time.perf_counter()
    get_reranker()
    print(f"Reranker 模型加载: {time.perf_counter() - load_started:.2f} s（不计入计时）")
    rag.retrieve_hybrid_rerank("warmup", top_k=1)
    print("=" * 100)

    schemes = [
        ("A 仅向量", rag.retrieve),
        ("B 仅 BM25", search_bm25),
        ("C 混合 RRF", rag.retrieve_hybrid),
        ("D RRF+Rerank", rag.retrieve_hybrid_rerank),
    ]
    scheme_names = [name for name, _ in schemes]
    stats = {
        name: {"correct": 0, "returned": 0, "hit5": 0, "top1": 0, "ms": 0.0}
        for name in scheme_names
    }
    d_hits_by_row: dict[int, list[dict]] = {}

    for row_index, row in enumerate(rows, start=1):
        query = row["query"]
        expected = row["expected"]
        expected_text = ", ".join(sorted(expected)) if expected else "无"

        print(
            f"\n[{row_index:2d}/{len(rows)}] [{row['query_type']}] {query}"
        )
        print(f"      期望来源: {expected_text}")

        for name, retrieve in schemes:
            hits, elapsed_ms = timed(
                lambda: retrieve(query, top_k=TOP_K)
            )
            correct = count_correct(hits, expected)
            returned = len(hits)

            stats[name]["correct"] += correct
            stats[name]["returned"] += returned
            stats[name]["hit5"] += 1 if correct > 0 else 0
            stats[name]["top1"] += (
                1
                if hits
                and (hits[0].get("metadata") or {}).get("source") in expected
                else 0
            )
            stats[name]["ms"] += elapsed_ms

            if name == "D RRF+Rerank":
                d_hits_by_row[row_index] = hits

            print(
                f"      {name:<14} 正确 {correct}/{returned}"
                f" | 耗时 {elapsed_ms:>8.1f} ms"
            )

    # ---------- 汇总 ----------
    total_questions = len(rows)
    print("\n" + "=" * 100)
    print("汇总（正确/返回 = 全部题目的正确条数 / 返回条数）")
    print(
        f"{'方案':<16}{'正确/返回':>12}{'正确率':>10}{'Hit@5':>9}"
        f"{'Top-1':>8}{'平均正确':>9}{'平均返回':>9}{'平均耗时':>12}"
    )
    for name in scheme_names:
        item = stats[name]
        returned = item["returned"]
        precision = item["correct"] / returned if returned else 0.0
        print(
            f"{name:<16}"
            f"{item['correct']:>5}/{returned:<6}"
            f"{precision:>9.1%}"
            f"{item['hit5']:>6}/{total_questions:<2}"
            f"{item['top1']:>5}/{total_questions:<2}"
            f"{item['correct'] / total_questions:>9.1f}"
            f"{returned / total_questions:>9.1f}"
            f"{item['ms'] / total_questions:>10.1f} ms"
        )

    # ---------- 第 2 题阈值效果 ----------
    if len(rows) >= 2:
        q2_hits = d_hits_by_row.get(2, [])
        print("\n" + "-" * 100)
        print(f"第 2 题（Trade-In）D 方案在阈值 {SCORE_THRESHOLD} 下最终返回 "
              f"{len(q2_hits)} 条上下文：")
        for rank, hit in enumerate(q2_hits, start=1):
            metadata = hit.get("metadata") or {}
            print(
                f"  {rank}. {metadata.get('source')} | page {metadata.get('page')}"
                f" | rerank {float(hit.get('rerank_score', 0.0)):.4f}"
                f" | RRF 第 {hit.get('rrf_rank')} 名"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
