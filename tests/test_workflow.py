"""端到端测试 Agentic RAG LangGraph 工作流。

用法（项目根目录）：
    C:\\RAG\\P1\\Prag\\Scripts\\python.exe tests\\test_workflow.py
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

from backend.agent.nodes import REFUSAL_ANSWER  # noqa: E402
from backend.agent.workflow import create_initial_state, workflow  # noqa: E402
from backend.services.rag_service import RAGService  # noqa: E402
from config.settings import settings  # noqa: E402

CASES = [
    {
        "name": "用例 1（有答案）",
        "question": "Velvera 怎么参与以旧换新计划？",
        "expect": "generator",
    },
    {
        "name": "用例 2（无答案）",
        "question": "Velvera 公司提供免费午餐吗？",
        "expect": "refusal",
    },
    {
        "name": "用例 3（极难纯中文问题）",
        "question": "Velvera 是否计划在火星建立研发中心？",
        "expect": "retry",
    },
]


def configure_logging() -> None:
    node_logger = logging.getLogger("backend.agent.nodes")
    node_logger.setLevel(logging.INFO)
    node_logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    node_logger.addHandler(handler)
    node_logger.propagate = False


def format_update(node: str, update: dict) -> str:
    if node in {"query_rewriter", "query_rewrite_retry"}:
        lines = [
            f"rewritten_question: {update.get('rewritten_question', '')}",
            f"retry_count: {update.get('retry_count', '')}",
        ]
        if update.get("error"):
            lines.append(f"error: {update['error']}")
        return "\n".join(lines)

    if node == "retriever":
        chunks = update.get("chunks") or []
        lines = [f"chunks: {len(chunks)}"]
        for rank, chunk in enumerate(chunks[:3], start=1):
            metadata = chunk.get("metadata") or {}
            lines.append(
                f"  [{rank}] {metadata.get('source')} | page {metadata.get('page')}"
                f" | rerank {float(chunk.get('rerank_score', 0.0)):.4f}"
            )
        if update.get("error"):
            lines.append(f"error: {update['error']}")
        return "\n".join(lines)

    if node == "evidence_grader":
        return f"is_enough: {update.get('is_enough')}"

    if node in {"generator", "refusal"}:
        answer = str(update.get("answer", ""))
        preview = answer if len(answer) <= 400 else answer[:400] + "..."
        return f"answer:\n{preview}"

    return str(update)


def run_case(case: dict) -> tuple[bool, dict, list[str]]:
    question = case["question"]
    state: dict = create_initial_state(question)
    path: list[str] = []

    print("\n" + "=" * 96)
    print(case["name"])
    print(f"问题: {question}")
    print("=" * 96)

    started = time.perf_counter()
    try:
        for step in workflow.stream(state, stream_mode="updates"):
            for node_name, update in step.items():
                path.append(node_name)
                state.update(update)
                print(f"\n>>> 节点: {node_name}")
                print(format_update(node_name, update))
    except Exception as exc:
        print(f"\n❌ 工作流执行失败: {type(exc).__name__}: {exc}")
        return False, state, path

    elapsed = time.perf_counter() - started
    print("\n" + "-" * 96)
    print(f"节点路径: {' -> '.join(path)}")
    print(f"最终 retry_count: {state.get('retry_count')}")
    print(f"最终 is_enough: {state.get('is_enough')}")
    print(f"最终 answer:\n{state.get('answer')}")
    print(f"总耗时: {elapsed:.1f} s")

    if case["expect"] == "generator":
        passed = (
            "generator" in path
            and bool(state.get("answer"))
            and state.get("answer") != REFUSAL_ANSWER
        )
    elif case["expect"] == "refusal":
        passed = (
            "query_rewrite_retry" in path
            and "refusal" in path
            and state.get("is_enough") is False
            and state.get("answer") == REFUSAL_ANSWER
        )
    else:  # retry
        passed = "query_rewrite_retry" in path

    print(f"结果: {'✅ 通过' if passed else '❌ 失败'}")
    return passed, state, path


def main() -> int:
    configure_logging()
    print(f"LLM 模型: {settings.LLM_MODEL}")

    started = time.perf_counter()
    RAGService().retrieve_hybrid_rerank("warmup", top_k=1)
    print(f"检索模型预热: {time.perf_counter() - started:.1f} s（不计入用例耗时）")

    results = []
    for case in CASES:
        passed, state, path = run_case(case)
        results.append((case, passed, path))

    print("\n" + "=" * 96)
    print("测试汇总:")
    for case, passed, path in results:
        print(f"  {'✅' if passed else '❌'} {case['name']}: {' -> '.join(path)}")

    return 0 if all(passed for _, passed, _ in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
