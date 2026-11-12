"""阶段 1 基线评估：基础向量检索（Top-5）。

读取 ``data/Dataset categories - queries.csv`` 中前 10 个
Velvera Technologies 的问题，逐个调用 ``RAGService.retrieve``，
打印每个问题的 Top-5 来源（source / page）与检索耗时。

用法（项目根目录）：
    C:\\RAG\\P1\\Prag\\Scripts\\python.exe tests\\evaluate_baseline.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Windows 控制台默认编码可能不是 UTF-8，统一输出编码避免中文乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd  # noqa: E402

from backend.services.document_service import DocumentService  # noqa: E402
from backend.services.rag_service import RAGService  # noqa: E402

CSV_PATH = BASE_DIR / "data" / "Dataset categories - queries.csv"
CATEGORY = "Velvera Technologies"
LIMIT = 10
TOP_K = 5

# CSV 不是 UTF-8 编码，实测 GB18030 才能正确还原弯引号等字符
CSV_ENCODING = "gb18030"


def load_queries(limit: int = LIMIT) -> pd.DataFrame:
    """读取并筛选前 limit 个目标企业的问题，保持 CSV 原始顺序。"""
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"找不到评估数据: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH, encoding=CSV_ENCODING)
    df.columns = [str(col).strip() for col in df.columns]

    missing = {"Enterprise Name", "Query Type", "Query"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少必要列: {sorted(missing)}")

    mask = (
        df["Enterprise Name"].astype(str).str.strip().str.casefold()
        == CATEGORY.casefold()
    )
    subset = df.loc[mask, ["Query Type", "Query"]].copy()
    subset = subset.dropna(subset=["Query"])
    subset = subset[subset["Query"].astype(str).str.strip() != ""]

    return subset.head(limit)


def main() -> int:
    queries = load_queries()

    print("=" * 92)
    print("阶段 1 基线：基础向量检索（Top-5）")
    print(f"数据文件: {CSV_PATH.name}")
    print(f"企业筛选: {CATEGORY} | 参与评估问题数: {len(queries)}")
    print(f"知识库总块数: {DocumentService().get_total_chunks()}")
    print("=" * 92)

    rag = RAGService()

    # 预热：首次检索会加载 embedding 模型、初始化 Chroma，排除它对单题耗时的影响
    warm_start = time.perf_counter()
    rag.retrieve("warmup", top_k=1)
    warm_elapsed = (time.perf_counter() - warm_start) * 1000
    print(f"预热完成（模型/索引加载）: {warm_elapsed:.1f} ms（不计入下方统计）\n")

    timings: list[float] = []
    failures: list[str] = []

    for index, (query_type, raw_query) in enumerate(
        queries.itertuples(index=False, name=None), start=1
    ):
        query = str(raw_query).strip()
        print("-" * 92)
        print(f"[{index:2d}/{len(queries)}] [{query_type}] {query}")

        started = time.perf_counter()
        try:
            hits = rag.retrieve(query, top_k=TOP_K)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000
            failures.append(query)
            print(f"  检索失败（{elapsed_ms:.1f} ms）: {type(exc).__name__}: {exc}")
            continue

        elapsed_ms = (time.perf_counter() - started) * 1000
        timings.append(elapsed_ms)
        print(f"  检索耗时: {elapsed_ms:.1f} ms | 命中 {len(hits)} 条")

        if not hits:
            print("  (无检索结果)")
            continue

        for rank, hit in enumerate(hits, start=1):
            metadata = hit.get("metadata") or {}
            source = metadata.get("source", "unknown")
            page = metadata.get("page", "N/A")
            distance = hit.get("distance")
            distance_text = (
                f"{float(distance):.4f}"
                if isinstance(distance, (int, float))
                else "N/A"
            )
            print(f"    {rank}. {source} | page {page} | distance {distance_text}")

    # 汇总
    print("\n" + "=" * 92)
    print(f"评估完成: 成功 {len(timings)} 题，失败 {len(failures)} 题")
    if timings:
        total_ms = sum(timings)
        print(
            f"检索耗时: 总计 {total_ms / 1000:.3f} s | "
            f"平均 {total_ms / len(timings):.1f} ms | "
            f"最快 {min(timings):.1f} ms | 最慢 {max(timings):.1f} ms"
        )
    if failures:
        print("失败问题:")
        for query in failures:
            print(f"  - {query}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
