"""验证证据判断 Prompt 放宽后：3 个假阴性转 True，且负例仍为 False。

用法（项目根目录）：
    C:\\RAG\\P1\\Prag\\Scripts\\python.exe tests\\test_evidence_grader.py
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
from evaluate_experiment import load_queries  # noqa: E402

# 评估报告中出现假阴性的 3 题（用原始 idx 定位）
POSITIVE_INDICES = [140, 71, 47]

NEGATIVE_QUESTIONS = [
    "Velvera 公司提供免费午餐吗？",
    "Velvera 是否计划在火星建立研发中心？",
]


def configure_logging() -> None:
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


def run_pipeline(question: str) -> AgentState:
    state = new_state(question)
    state.update(query_rewriter_node(state))
    state.update(retriever_node(state))
    state.update(evidence_grader_node(state))
    return state


def main() -> int:
    configure_logging()
    rows = {row["index"]: row for row in load_queries()}

    started_warmup = time.perf_counter()
    RAGService().retrieve_hybrid_rerank("warmup", top_k=1)
    print(f"检索模型预热: {time.perf_counter() - started_warmup:.1f} s")

    checks: list[tuple[str, bool]] = []

    print("\n" + "=" * 100)
    print("假阴性用例（期望 is_enough=True）")
    print("=" * 100)

    for index in POSITIVE_INDICES:
        row = rows[index]
        question = row["query"]
        state = run_pipeline(question)

        print(f"\nidx={index} | expected={sorted(row['expected'])}")
        print(f"Q: {question}")
        print(f"改写后: {state['rewritten_question']}")
        print(f"召回 chunks: {len(state['chunks'])}")
        for rank, chunk in enumerate(state["chunks"][:3], start=1):
            metadata = chunk.get("metadata") or {}
            print(
                f"  [{rank}] {metadata.get('source')} | page {metadata.get('page')}"
            )
        print(f"is_enough: {state['is_enough']}")
        checks.append((f"idx={index}", state["is_enough"] is True))

    print("\n" + "=" * 100)
    print("负例（期望 is_enough=False，防止过度放宽）")
    print("=" * 100)

    for question in NEGATIVE_QUESTIONS:
        state = run_pipeline(question)
        print(f"\nQ: {question}")
        print(f"改写后: {state['rewritten_question']}")
        print(f"召回 chunks: {len(state['chunks'])}")
        print(f"is_enough: {state['is_enough']}")
        checks.append((question, state["is_enough"] is False))

    print("\n" + "=" * 100)
    print("校验汇总:")
    for name, passed in checks:
        print(f"  {'✅' if passed else '❌'} {name}")

    return 0 if all(passed for _, passed in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
