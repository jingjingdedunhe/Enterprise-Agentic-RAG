"""FastAPI 接口实测：健康检查、问答、文档上传/列表/删除。

用法（先启动 uvicorn，项目根目录）：
    C:\\RAG\\P1\\Prag\\Scripts\\python.exe tests\\test_api.py
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
SAMPLE_PDF = BASE_DIR / "data" / "sample_docs" / "accessories.pdf"


def print_json(label: str, data: object) -> None:
    print(f"\n{label}:")
    print(json.dumps(data, ensure_ascii=False, indent=2))


def get_documents() -> dict:
    response = requests.get(f"{BASE_URL}/documents", timeout=60)
    response.raise_for_status()
    return response.json()


def main() -> int:
    checks: list[tuple[str, bool]] = []

    # 1. 健康检查
    health = requests.get(f"{BASE_URL}/health", timeout=10)
    health.raise_for_status()
    print_json("GET /health ->", health.json())
    checks.append(("/health 返回 status=ok", health.json().get("status") == "ok"))

    # 2. 基础问答
    chat_payload = {
        "query": "What is the price of the Velvera 65W GaN Charger?",
        "mode": "hybrid",
    }
    chat = requests.post(f"{BASE_URL}/chat", json=chat_payload, timeout=300)
    chat.raise_for_status()
    chat_data = chat.json()
    print(f"\nPOST /chat status: {chat.status_code}")
    print(f"  answer: {chat_data['answer'][:180]}")
    print(f"  citations: {len(chat_data.get('citations', []))}")
    checks.append(("/chat 返回答案与引用", bool(chat_data.get("answer"))))

    # 3. Agentic 问答
    agent_payload = {"query": "Velvera 怎么参与以旧换新计划？"}
    agent = requests.post(
        f"{BASE_URL}/agent/chat", json=agent_payload, timeout=600
    )
    agent.raise_for_status()
    agent_data = agent.json()
    print(f"\nPOST /agent/chat status: {agent.status_code}")
    print(f"  rewritten_question: {agent_data.get('rewritten_question')}")
    print(f"  retry_count: {agent_data.get('retry_count')}")
    print(f"  is_enough: {agent_data.get('is_enough')}")
    print(f"  chunks: {len(agent_data.get('chunks', []))}")
    print(f"  citations: {len(agent_data.get('citations', []))}")
    print(f"  answer: {agent_data.get('answer', '')[:180]}")
    checks.append(
        (
            "/agent/chat 返回完整 state",
            bool(agent_data.get("answer"))
            and "rewritten_question" in agent_data
            and "retry_count" in agent_data,
        )
    )

    # 4. 文档列表（上传前）
    before = get_documents()
    before_sources = {item["source"] for item in before["documents"]}
    print(
        f"\nGET /documents（上传前）: {before['total_documents']} 个文档，"
        f"{before['total_chunks']} 个 chunks"
    )

    test_source = f"api_test_upload_{uuid.uuid4().hex[:8]}.pdf"
    uploaded = False

    try:
        # 5. 上传 PDF
        with open(SAMPLE_PDF, "rb") as handle:
            response = requests.post(
                f"{BASE_URL}/documents/upload",
                files={"file": (test_source, handle, "application/pdf")},
                timeout=300,
            )
        response.raise_for_status()
        upload_data = response.json()
        uploaded = True
        print_json("POST /documents/upload ->", upload_data)
        checks.append(
            (
                "上传新 PDF 成功",
                upload_data.get("source") == test_source
                and upload_data.get("chunks_added", 0) > 0,
            )
        )

        # 6. 列表确认新文件出现
        after = get_documents()
        after_sources = {item["source"] for item in after["documents"]}
        uploaded_info = next(
            (
                item
                for item in after["documents"]
                if item["source"] == test_source
            ),
            None,
        )
        print(
            f"\nGET /documents（上传后）: {after['total_documents']} 个文档，"
            f"{after['total_chunks']} 个 chunks"
        )
        print(f"  新文档: {uploaded_info}")
        checks.append(("文档列表出现新文件", test_source in after_sources))
        checks.append(
            (
                "新文件 chunk 数与上传结果一致",
                uploaded_info is not None
                and uploaded_info["chunk_count"]
                == upload_data.get("chunks_added"),
            )
        )

        # 7. 删除新文件
        deleted = requests.delete(
            f"{BASE_URL}/documents/{test_source}", timeout=120
        )
        deleted.raise_for_status()
        delete_data = deleted.json()
        uploaded = False  # 已删除，finally 不再重复删
        print_json("DELETE /documents/{source} ->", delete_data)
        checks.append(
            (
                "删除返回的 chunk 数正确",
                delete_data.get("deleted_chunks")
                == upload_data.get("chunks_added"),
            )
        )

        # 8. 列表确认文件消失
        final = get_documents()
        final_sources = {item["source"] for item in final["documents"]}
        print(
            f"\nGET /documents（删除后）: {final['total_documents']} 个文档，"
            f"{final['total_chunks']} 个 chunks"
        )
        checks.append(("删除后文档消失", test_source not in final_sources))
        checks.append(
            (
                "记录数恢复到上传前",
                final["total_documents"] == before["total_documents"]
                and final["total_chunks"] == before["total_chunks"],
            )
        )
    finally:
        if uploaded:
            try:
                requests.delete(
                    f"{BASE_URL}/documents/{test_source}", timeout=120
                )
            except Exception:
                pass

    print("\n" + "=" * 80)
    print("接口测试汇总:")
    for name, passed in checks:
        print(f"  {'✅' if passed else '❌'} {name}")

    return 0 if all(passed for _, passed in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
