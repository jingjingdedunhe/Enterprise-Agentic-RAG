"""对话历史工具。

统一处理“最近 N 轮对话”的格式化与检索查询拼接，供生成器、Agent
改写节点、Streamlit 和 FastAPI 复用。
"""

from __future__ import annotations


def format_history(
    history: list[dict] | None,
    max_messages: int = 6,
    max_chars: int = 600,
) -> str:
    """把历史消息格式化为 Prompt 片段（默认最近 3 轮 / 6 条）。"""
    messages = [
        message
        for message in (history or [])
        if message.get("role") in {"user", "assistant"}
        and str(message.get("content", "")).strip()
    ]
    selected = messages[-max_messages:]

    lines: list[str] = []
    for message in selected:
        role = "用户" if message["role"] == "user" else "助手"
        content = " ".join(str(message.get("content", "")).split())
        if len(content) > max_chars:
            content = content[:max_chars] + "..."
        lines.append(f"{role}: {content}")

    return "\n".join(lines)


def build_retrieval_query(
    query: str,
    history: list[dict] | None,
    max_user_turns: int = 2,
) -> str:
    """用最近几轮用户问题拼接检索查询，帮助普通检索理解省略/指代。

    示例：历史问过 "What is Velvera One?"，当前问 "What is its price?"，
    检索查询会变成 "What is Velvera One? What is its price?"。
    """
    user_messages = [
        str(message.get("content", "")).strip()
        for message in (history or [])
        if message.get("role") == "user"
        and str(message.get("content", "")).strip()
    ]
    recent = user_messages[-max_user_turns:]
    if not recent:
        return query

    return " ".join(recent + [query])
