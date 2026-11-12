"""Agentic RAG 工作流的状态定义。"""

from typing import TypedDict


class AgentState(TypedDict, total=True):
    """LangGraph 工作流中流转的共享状态。

    各字段的初始值约定（由工作流入口构造，不在 TypedDict 中强制）：

    - ``rewritten_question``：``""``
    - ``chunks``：``[]``
    - ``is_enough``：``False``
    - ``retry_count``：``0``
    - ``answer``：``""``
    - ``citations``：``[]``
    - ``error``：``""``

    Attributes:
        question: 用户的原始提问。
        rewritten_question: Agent 改写后的问题，初始为空字符串。
        chunks: 当前检索到的上下文片段（text + metadata）。
        is_enough: 当前证据是否充足，默认 False。
        retry_count: 当前重试次数，默认 0。
        answer: 最终生成的答案。
        citations: 提取的引用来源。
        error: 记录异常信息。
        history: 最近几轮对话历史（role/content），用于指代消解。
    """

    question: str
    rewritten_question: str
    chunks: list[dict]
    is_enough: bool
    retry_count: int
    answer: str
    citations: list[dict]
    error: str
    history: list[dict]
