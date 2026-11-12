"""AppTest：验证两轮对话的指代消解与“清空当前对话”。

用法（项目根目录）：
    C:\\RAG\\P1\\Prag\\Scripts\\python.exe tests\\test_chat_history.py
"""

from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = str(BASE_DIR / "app" / "main.py")
FIRST_QUESTION = "What is Velvera One?"
SECOND_QUESTION = "What is its price?"


def has_price(answer: str) -> bool:
    return any(token in answer for token in ("₹", "$", "price", "Price"))


def select_mode(at: AppTest, keyword: str) -> None:
    options = at.sidebar.radio[0].options
    target = next(option for option in options if keyword in option)
    at.sidebar.radio[0].set_value(target).run()


def click_sidebar_button(at: AppTest, text: str) -> None:
    target = next(button for button in at.sidebar.button if text in button.label)
    target.click().run()


def test_agent_mode() -> bool:
    print("=" * 90)
    print("场景 1：Agentic RAG 两轮对话")
    print("=" * 90)

    at = AppTest.from_file(APP_PATH, default_timeout=300).run()
    select_mode(at, "Agentic")
    print("当前模式:", at.sidebar.radio[0].value)

    at.chat_input[0].set_value(FIRST_QUESTION).run()
    first = at.session_state["messages"][-1]
    print("第一轮回答:", first["content"][:180])

    at.chat_input[0].set_value(SECOND_QUESTION).run()
    messages = at.session_state["messages"]
    second = messages[-1]
    meta = second.get("agent_meta") or {}
    answer = second.get("content", "")

    print("\n第二轮改写:", meta.get("rewritten_question"))
    print("第二轮 is_enough:", meta.get("is_enough"))
    print("第二轮 retry_count:", meta.get("retry_count"))
    print("第二轮 answer:", answer[:300])
    print("消息条数:", len(messages))

    checks = {
        "两轮共 4 条消息": len(messages) == 4,
        "改写包含 Velvera One": "velvera one" in str(meta.get("rewritten_question", "")).lower(),
        "第二轮回答包含价格": has_price(answer),
        "第二轮回答不是拒答": "根据现有资料无法回答" not in answer,
        "Agent 面板存在": any("Agent 思考" in e.label for e in at.expander),
    }
    for name, passed in checks.items():
        print(f"  {'✅' if passed else '❌'} {name}")

    # 清空当前对话
    click_sidebar_button(at, "清空当前对话")
    cleared = len(at.session_state["messages"]) == 0
    print(f"\n清空对话后消息数: {len(at.session_state['messages'])}")
    print(f"  {'✅' if cleared else '❌'} 清空当前对话")
    checks["清空当前对话"] = cleared
    return all(checks.values())


def test_hybrid_mode() -> bool:
    print("\n" + "=" * 90)
    print("场景 2：普通混合检索两轮对话")
    print("=" * 90)

    at = AppTest.from_file(APP_PATH, default_timeout=300).run()
    select_mode(at, "混合检索")
    print("当前模式:", at.sidebar.radio[0].value)

    at.chat_input[0].set_value(FIRST_QUESTION).run()
    at.chat_input[0].set_value(SECOND_QUESTION).run()
    messages = at.session_state["messages"]
    second = messages[-1]
    answer = second.get("content", "")

    print("第一轮回答:", messages[-3]["content"][:150])
    print("\n第二轮 answer:", answer[:300])
    print("消息条数:", len(messages))

    checks = {
        "两轮共 4 条消息": len(messages) == 4,
        "第二轮回答包含价格": has_price(answer),
        "第二轮回答不是拒答": "根据现有资料无法回答" not in answer,
    }
    for name, passed in checks.items():
        print(f"  {'✅' if passed else '❌'} {name}")
    return all(checks.values())


def main() -> int:
    results = [test_agent_mode(), test_hybrid_mode()]
    print("\n" + "=" * 90)
    print("多轮对话测试汇总:")
    print(f"  {'✅' if results[0] else '❌'} Agentic RAG 两轮指代")
    print(f"  {'✅' if results[1] else '❌'} 普通混合检索两轮指代")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
