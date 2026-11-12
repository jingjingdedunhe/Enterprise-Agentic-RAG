"""测试查询改写、检索、证据判断三个节点（不涉及 LangGraph 图结构）。

用法（项目根目录）：
    C:\\RAG\\P1\\Prag\\Scripts\\python.exe tests\\test_nodes.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.agent.nodes import (  # noqa: E402
    evidence_grader_node,
    query_rewriter_node,
    retriever_node,
)
from backend.agent.state import AgentState  # noqa: E402
from backend.services.rag_service import RAGService  # noqa: E402
from config.settings import settings  # noqa: E402

CASES = [
    {
        "name": "用例 1（有答案）",
        "question": "Velvera 怎么参与以旧换新计划？",
        "expected": True,
    },
    {
        "name": "用例 2（无答案）",
        "question": "Velvera 公司提供免费午餐吗？",
        "expected": False,
    },
]


def configure_evidence_logging() -> None:
    """把 nodes 模块的 evidence_grader 日志输出到控制台。"""
    node_logger = logging.getLogger("backend.agent.nodes")
    node_logger.setLevel(logging.INFO)
    node_logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    node_logger.addHandler(handler)
    node_logger.propagate = False


def new_state(question: str) -> AgentState:
    return {
        "question": question,
        "rewritten_question": "",
        "chunks": [],
        "is_enough": False,
        "retry_count": 0,
        "answer": "",
        "citations": [],
        "error": "",
    }


def run_case(case: dict) -> bool:
    state = new_state(case["question"])

    print("\n" + "=" * 92)
    print(f"{case['name']}")
    print(f"原始问题: {state['question']}")
    print("=" * 92)

    # 1. 查询改写
    started = time.perf_counter()
    state.update(query_rewriter_node(state))
    rewrite_ms = (time.perf_counter() - started) * 1000
    print(f"\n【query_rewriter_node】{rewrite_ms:.1f} ms")
    print(f"改写后: {state['rewritten_question']}")
    print(f"retry_count: {state['retry_count']}")
    if state["error"]:
        print(f"error: {state['error']}")

    # 2. 检索
    started = time.perf_counter()
    state.update(retriever_node(state))
    retrieve_ms = (time.perf_counter() - started) * 1000
    print(f"\n【retriever_node】{retrieve_ms:.1f} ms | 召回 {len(state['chunks'])} 条")
    if state["error"]:
        print(f"error: {state['error']}")
    for rank, hit in enumerate(state["chunks"][:3], start=1):
        metadata = hit.get("metadata") or {}
        print(
            f"  {rank}. {metadata.get('source')} | page {metadata.get('page')}"
            f" | rerank {float(hit.get('rerank_score', 0.0)):.4f}"
        )

    # 3. 证据判断（reason 由 evidence_grader_node 的日志输出）
    print("\n【evidence_grader_node】")
    started = time.perf_counter()
    state.update(evidence_grader_node(state))
    grade_ms = (time.perf_counter() - started) * 1000
    print(
        f"is_enough: {state['is_enough']} | 预期: {case['expected']}"
        f" | 耗时: {grade_ms:.1f} ms"
    )

    passed = state["is_enough"] == case["expected"]
    print(f"结果: {'✅ 通过' if passed else '❌ 失败'}")
    return passed


def main() -> int:
    configure_evidence_logging()

    print(f"LLM 模型: {settings.LLM_MODEL}")

    # 预热检索链路，避免把模型冷启动算进节点耗时
    started = time.perf_counter()
    RAGService().retrieve_hybrid_rerank("warmup", top_k=1)
    print(f"检索模型预热: {time.perf_counter() - started:.1f} s（不计入节点耗时）")

    results = [run_case(case) for case in CASES]

    print("\n" + "=" * 92)
    print("测试汇总:")
    for case, passed in zip(CASES, results):
        print(f"  {'✅' if passed else '❌'} {case['name']}: {case['question']}")

    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
