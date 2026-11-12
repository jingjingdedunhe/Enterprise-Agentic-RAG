"""测试脚本：验证 LLM 接入与严格 RAG 回答。

用法（在项目根目录 C:\\RAG\\P1 下执行）：
    C:\\RAG\\P1\\Prag\\Scripts\\python.exe tests\\test_generation.py
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

from backend.generation.answer_generator import (  # noqa: E402
    REFUSAL_TEXT,
    generate_answer,
)
from backend.services.rag_service import RAGService  # noqa: E402
from config.settings import settings  # noqa: E402

TOP_K = 5


def show_hits(hits: list[dict]) -> None:
    print(f"检索到 {len(hits)} 条上下文:")
    for index, hit in enumerate(hits, start=1):
        metadata = hit["metadata"]
        print(
            f"  [{index}] {metadata.get('source')} | page={metadata.get('page')} "
            f"| distance={hit['distance']:.4f}"
        )


def run_case(rag: RAGService, title: str, query: str) -> str:
    print("=" * 78)
    print(title)
    print(f"问题: {query}")

    hits = rag.retrieve(query, top_k=TOP_K)
    show_hits(hits)

    started = time.perf_counter()
    answer = generate_answer(query, hits)
    elapsed = time.perf_counter() - started

    print(f"LLM 回答（耗时 {elapsed:.1f}s）:")
    print("-" * 78)
    print(answer)
    print("-" * 78)
    print()
    return answer


def main() -> int:
    print(f"LLM 模型: {settings.LLM_MODEL}")
    print(f"Base URL: {settings.LLM_BASE_URL}")
    if not settings.LLM_API_KEY:
        print("❌ 未读取到 LLM_API_KEY，请检查 .env")
        return 1

    rag = RAGService()
    checks: list[bool] = []

    try:
        # 测试 1：有答案
        answer_1 = run_case(
            rag,
            "测试 1：有答案（价格查询）",
            "What is the price of the Velvera 65W GaN Charger?",
        )
        has_price = any(token in answer_1 for token in ("3,499", "3499", "$42"))
        checks.append(has_price)
        print(f"{'✅' if has_price else '❌'} 回答包含充电器价格（3,499 / $42）")
        has_source = "引用来源" in answer_1
        checks.append(has_source)
        print(f"{'✅' if has_source else '❌'} 回答末尾包含引用来源")

        # 测试 2：无答案，必须拒答
        answer_2 = run_case(
            rag,
            "测试 2：无答案（宠物保险）",
            "Does Velvera support pet insurance?",
        )
        refused = REFUSAL_TEXT in answer_2
        checks.append(refused)
        print(f"{'✅' if refused else '❌'} 明确拒答：包含“{REFUSAL_TEXT}”")

        # 额外：空上下文直接拒答，不调用 LLM
        empty_answer = generate_answer("Any question?", [])
        empty_refused = REFUSAL_TEXT in empty_answer
        checks.append(empty_refused)
        print(f"{'✅' if empty_refused else '❌'} 空上下文直接拒答: {empty_answer!r}")
    except Exception as exc:
        print(f"❌ 测试失败: {type(exc).__name__}: {exc}")
        return 1

    print()
    if all(checks):
        print("全部测试通过 ✅")
        return 0

    print("部分测试未通过 ❌")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
