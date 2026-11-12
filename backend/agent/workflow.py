"""Agentic RAG 工作流组装。

图结构::

    START -> query_rewriter -> retriever -> evidence_grader
                 ^                                |
                 |                                | is_enough=True
                 |                                v
                 |                            generator -> END
                 |
                 +---- query_rewrite_retry <---- is_enough=False 且 retry_count < 2
                                                    |
                                                    +----> refusal -> END
                                                         (retry_count >= 2)

``query_rewrite_retry`` 是带上一轮失败反馈的重试改写节点；首次改写仍由
``query_rewriter`` 完成。
"""

from __future__ import annotations

from contextvars import ContextVar

from langgraph.graph import END, START, StateGraph

from backend.agent.nodes import (
    MAX_RETRIES,
    evidence_grader_node,
    generate_answer_node,
    query_rewrite_retry_node,
    query_rewriter_node,
    refusal_node,
    retriever_node,
)
from backend.agent.state import AgentState


def create_initial_state(
    question: str,
    history: list[dict] | None = None,
) -> AgentState:
    """构造工作流初始状态。"""
    return {
        "question": question,
        "rewritten_question": "",
        "chunks": [],
        "is_enough": False,
        "retry_count": 0,
        "answer": "",
        "citations": [],
        "error": "",
        "history": list(history or []),
    }


def route_after_grader(state: AgentState) -> str:
    """证据判断后的路由。

    - 证据充足：生成答案；
    - 证据不足且还有重试次数：带反馈重新改写；
    - 达到最大重试次数：拒答。
    """
    if state.get("is_enough"):
        return "generator"

    if int(state.get("retry_count", 0)) < MAX_RETRIES:
        return "query_rewrite_retry"

    return "refusal"


def build_workflow():
    """构建并编译 LangGraph 工作流。"""
    builder = StateGraph(AgentState)

    builder.add_node("query_rewriter", query_rewriter_node)
    builder.add_node("query_rewrite_retry", query_rewrite_retry_node)
    builder.add_node("retriever", retriever_node)
    builder.add_node("evidence_grader", evidence_grader_node)
    builder.add_node("generator", generate_answer_node)
    builder.add_node("refusal", refusal_node)

    builder.add_edge(START, "query_rewriter")
    builder.add_edge("query_rewriter", "retriever")
    builder.add_edge("retriever", "evidence_grader")

    builder.add_conditional_edges(
        "evidence_grader",
        route_after_grader,
        {
            "generator": "generator",
            "query_rewrite_retry": "query_rewrite_retry",
            "refusal": "refusal",
        },
    )

    builder.add_edge("query_rewrite_retry", "retriever")
    builder.add_edge("generator", END)
    builder.add_edge("refusal", END)

    return builder.compile()


# 模块级已编译图，可直接 workflow.invoke(initial_state)
workflow = build_workflow()

# 最近一次 invoke_workflow 实际经过的节点（按执行顺序）
_last_trace: ContextVar[tuple[str, ...]] = ContextVar(
    "agent_workflow_trace", default=()
)


def get_last_trace() -> list[str]:
    """返回当前上下文最近一次 ``invoke_workflow`` 的节点路径。"""
    return list(_last_trace.get())


def invoke_workflow(
    question: str,
    history: list[dict] | None = None,
    recursion_limit: int = 50,
) -> AgentState:
    """便捷入口：输入问题（可选历史），跑完整图并返回最终状态。

    内部使用 ``stream`` 执行以记录真实节点路径，可通过
    :func:`get_last_trace` 读取；返回结构与 ``workflow.invoke`` 一致。
    """
    initial_state = create_initial_state(question, history=history)
    state: dict = dict(initial_state)
    trace: list[str] = []

    for step in workflow.stream(
        initial_state,
        config={"recursion_limit": recursion_limit},
        stream_mode="updates",
    ):
        for node_name, update in step.items():
            trace.append(node_name)
            if update:
                state.update(update)

    _last_trace.set(tuple(trace))
    return state  # type: ignore[return-value]
