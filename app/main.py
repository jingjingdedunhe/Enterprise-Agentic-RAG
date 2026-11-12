"""Enterprise Agentic RAG - Streamlit 最小交互界面。

启动方式（项目根目录）：
    C:\\RAG\\P1\\Prag\\Scripts\\streamlit.exe run app/main.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# 保证从 app/ 启动时也能导入项目根目录下的 backend / config
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from backend.agent.workflow import get_last_trace, invoke_workflow  # noqa: E402
from backend.generation.answer_generator import generate_answer  # noqa: E402
from backend.generation.history import build_retrieval_query  # noqa: E402
from backend.services.document_service import DocumentService  # noqa: E402
from backend.services.rag_service import RAGService  # noqa: E402
from config.settings import settings  # noqa: E402

st.set_page_config(
    page_title="Enterprise Agentic RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

TOP_K = 5
HYBRID_CANDIDATE_K = 20
RERANK_SCORE_THRESHOLD = 0.3
PREVIEW_LENGTH = 300
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
HISTORY_ROUNDS = 3

MODE_HYBRID = "混合检索+重排 (推荐)"
MODE_VECTOR = "纯向量检索"
MODE_AGENT = "Agentic RAG (自主纠错工作流)"
RETRIEVAL_MODES = [MODE_HYBRID, MODE_VECTOR, MODE_AGENT]

AGENT_NODE_LABELS = {
    "query_rewriter": "改写问题",
    "retriever": "混合检索",
    "evidence_grader": "证据判断",
    "query_rewrite_retry": "重试改写",
    "generator": "生成回答",
    "refusal": "拒答",
}

document_service = DocumentService()
rag_service = RAGService()

if "messages" not in st.session_state:
    st.session_state.messages = []


def get_recent_history() -> list[dict]:
    """取最近 HISTORY_ROUNDS 轮的对话，传给后端做指代消解。"""
    recent = st.session_state.messages[-(HISTORY_ROUNDS * 2):]
    return [
        {"role": message["role"], "content": message["content"]}
        for message in recent
        if message.get("role") in {"user", "assistant"}
    ]


def render_sources(chunks: list[dict]) -> None:
    """在回答下方展示引用来源折叠面板。"""
    if not chunks:
        return

    with st.expander("查看引用来源"):
        for index, chunk in enumerate(chunks, start=1):
            metadata = chunk.get("metadata") or {}
            source = metadata.get("source", "未知来源")
            page = metadata.get("page", "N/A")

            # 混合/重排结果没有“余弦距离”语义，按实际字段展示分数；
            # 相邻扩展 chunk 没有重排分数，单独标注。
            if chunk.get("expanded"):
                score_text = "**相邻补充上下文**"
            elif "rerank_score" in chunk:
                score_text = f"**重排分数：** `{float(chunk['rerank_score']):.4f}`"
                if "rrf_score" in chunk:
                    score_text += f"　**RRF 分数：** `{float(chunk['rrf_score']):.5f}`"
            elif "rrf_score" in chunk:
                score_text = f"**RRF 分数：** `{float(chunk['rrf_score']):.5f}`"
            elif isinstance(chunk.get("distance"), (int, float)):
                score_text = f"**余弦距离：** `{float(chunk['distance']):.4f}`"
            else:
                score_text = "**分数：** `N/A`"

            st.markdown(
                f"**[{index}] 来源：** `{source}`　"
                f"**页码：** {page}　{score_text}"
            )

            text = (chunk.get("text") or "").strip()
            preview = text[:PREVIEW_LENGTH]
            if len(text) > PREVIEW_LENGTH:
                preview += "..."
            st.caption("原文预览")
            st.text(preview)

            if index < len(chunks):
                st.divider()


def evidence_progression(meta: dict) -> str:
    """按重试次数还原每次证据判断的结果。"""
    attempts = max(1, int(meta.get("retry_count", 1) or 1))
    results = ["不足"] * attempts
    if meta.get("is_enough"):
        results[-1] = "充足"
    return " → ".join(results)


def render_agent_trace(trace: list[str], meta: dict, chunk_count: int) -> None:
    """展示 Agent 实际经过的节点和关键状态。"""
    if not trace:
        return

    with st.expander("🧠 查看 Agent 思考与检索过程"):
        path = " → ".join(
            AGENT_NODE_LABELS.get(node, node) for node in trace
        )
        st.markdown(f"**节点路径：** {path}")
        st.markdown(f"**证据判断：** {evidence_progression(meta)}")

        col_retry, col_enough, col_chunks = st.columns(3)
        col_retry.metric("重试次数", int(meta.get("retry_count", 0)))
        col_enough.metric(
            "证据是否充足", "是" if meta.get("is_enough") else "否"
        )
        col_chunks.metric("检索上下文", f"{chunk_count} 条")

        rewritten = meta.get("rewritten_question") or "（未改写）"
        st.markdown(f"**改写后的问题：** {rewritten}")

        if meta.get("error"):
            st.warning(f"Agent 记录的错误：{meta['error']}")


# ---------------------------- 侧边栏 ----------------------------
with st.sidebar:
    st.title("Enterprise Agentic RAG")
    st.caption("企业知识库智能问答 · 回答严格基于已入库文档")
    st.divider()

    flash = st.session_state.pop("_flash", None)
    if flash:
        st.success(flash)

    try:
        documents = document_service.list_documents()
        total_chunks = sum(item["chunk_count"] for item in documents)
        st.metric("知识库总块数", total_chunks)
    except Exception as exc:
        documents = []
        total_chunks = 0
        st.metric("知识库总块数", 0)
        st.error(f"读取知识库失败：{type(exc).__name__}: {exc}")

    st.divider()
    retrieval_mode = st.radio(
        "检索模式",
        RETRIEVAL_MODES,
        index=0,
        help=(
            "混合检索+重排：向量 + BM25 各取 Top-20，RRF 融合后用 "
            "bge-reranker 重排，并按 0.3 阈值截断。纯向量：仅 Chroma 余弦检索。"
            "Agentic RAG：LangGraph 自主改写、检索、证据判断；证据不足时"
            "带反馈重试，最多 2 次，仍不足则拒答。"
        ),
    )
    if retrieval_mode == MODE_HYBRID:
        st.caption(
            f"Top-{HYBRID_CANDIDATE_K} 召回 → Reranker → 阈值 {RERANK_SCORE_THRESHOLD}"
        )
    elif retrieval_mode == MODE_VECTOR:
        st.caption(f"Chroma 余弦相似度 Top-{TOP_K}")
    else:
        st.caption("LangGraph：改写 → 检索 → 证据判断 → 重试/生成/拒答")

    st.caption(f"当前模型：`{settings.LLM_MODEL}`")
    st.caption(f"向量库：`{settings.CHROMA_PATH}`")
    st.caption(f"会话记忆：最近 {HISTORY_ROUNDS} 轮")
    st.divider()

    with st.expander("📤 上传文档", expanded=False):
        uploaded_file = st.file_uploader(
            "选择 PDF 或 DOCX（≤20MB）",
            type=["pdf", "docx"],
            key="kb_file_uploader",
        )
        if uploaded_file is not None:
            file_size = len(uploaded_file.getvalue())
            st.caption(
                f"待入库：{uploaded_file.name}（{file_size / 1024:.1f} KB）"
            )

        if st.button(
            "上传并入库",
            key="kb_ingest_button",
            width="stretch",
        ):
            if uploaded_file is None:
                st.warning("请先选择文件。")
            else:
                filename = Path(uploaded_file.name).name
                suffix = Path(filename).suffix.lower()
                file_bytes = uploaded_file.getvalue()

                if suffix not in {".pdf", ".docx"}:
                    st.error("只支持 .pdf 或 .docx 文件。")
                elif len(file_bytes) > MAX_UPLOAD_BYTES:
                    st.error("文件超过 20MB 限制。")
                else:
                    try:
                        with st.spinner(f"正在解析并入库 {filename}..."):
                            with tempfile.TemporaryDirectory(
                                prefix="rag_ui_upload_"
                            ) as temp_dir:
                                temp_path = Path(temp_dir) / filename
                                temp_path.write_bytes(file_bytes)
                                added = document_service.ingest_file(
                                    str(temp_path)
                                )
                        st.session_state["_flash"] = (
                            f"已入库 {filename}，新增 {added} 个 chunk。"
                        )
                        st.rerun()
                    except Exception as exc:
                        st.error(
                            f"上传失败：{type(exc).__name__}: {exc}"
                        )

    with st.expander(f"📄 文档列表（{len(documents)}）", expanded=False):
        if documents:
            dataframe = pd.DataFrame(documents).rename(
                columns={"source": "文件名", "chunk_count": "Chunk 数"}
            )
            st.dataframe(
                dataframe,
                width="stretch",
                hide_index=True,
            )
            st.caption(
                f"共 {len(documents)} 个文档，{total_chunks} 个 chunk。"
            )
        else:
            st.caption("知识库为空，请先上传文档。")

    with st.expander("🗑️ 删除文档", expanded=False):
        if documents:
            selected_source = st.selectbox(
                "选择要删除的文档",
                options=[item["source"] for item in documents],
                key="kb_delete_source",
            )
            if st.button(
                "删除选中文档",
                key="kb_delete_button",
                width="stretch",
            ):
                try:
                    before = document_service.get_total_chunks()
                    document_service.delete_file(selected_source)
                    after = document_service.get_total_chunks()
                    st.session_state["_flash"] = (
                        f"已删除 {selected_source}，"
                        f"移除 {before - after} 个 chunk。"
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(
                        f"删除失败：{type(exc).__name__}: {exc}"
                    )
        else:
            st.caption("暂无可删除的文档。")

    st.divider()

    if st.button(
        "🧹 清空当前对话",
        key="clear_chat_button",
        width="stretch",
    ):
        st.session_state.messages = []
        st.session_state["_flash"] = "当前对话已清空。"
        st.rerun()

    st.caption("清空对话不会删除知识库文档。")

    if st.button(
        "🗑️ 清空当前知识库",
        key="clear_kb_button",
        width="stretch",
    ):
        try:
            document_service.clear_all()
            st.session_state["_flash"] = "知识库已清空。"
            st.rerun()
        except Exception as exc:
            st.error(f"清空知识库失败：{type(exc).__name__}: {exc}")

    st.caption("⚠️ 清空知识库操作不可恢复，请谨慎使用。")

# ---------------------------- 主区域 ----------------------------
st.header("💬 企业知识库问答")
st.caption("回答内容仅来自已入库文档；资料不足时模型会明确拒答，并在下方给出引用来源。")

if not st.session_state.messages:
    st.info("先在侧边栏确认知识库已有数据，然后在下方输入问题开始对话。")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            if message.get("agent_trace"):
                render_agent_trace(
                    message["agent_trace"],
                    message.get("agent_meta") or {},
                    len(message.get("chunks") or []),
                )
            render_sources(message.get("chunks") or [])

prompt = st.chat_input("请输入你的问题，例如：What is the price of the Velvera 65W GaN Charger?")

if prompt and prompt.strip():
    prompt = prompt.strip()

    # 先取历史（不包含当前问题），再展示/追加当前提问
    history = get_recent_history()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 检索 + 生成（普通模式） / Agent 工作流（Agentic 模式）
    with st.chat_message("assistant"):
        chunks: list[dict] = []
        answer: str | None = None
        agent_trace: list[str] = []
        agent_meta: dict | None = None

        try:
            if retrieval_mode == MODE_AGENT:
                with st.spinner("Agent 正在改写问题、检索并判断证据..."):
                    agent_state = invoke_workflow(prompt, history=history)
                    agent_trace = get_last_trace()

                chunks = agent_state.get("chunks") or []
                answer = (agent_state.get("answer") or "").strip()
                agent_meta = {
                    "retry_count": int(agent_state.get("retry_count", 0)),
                    "is_enough": bool(agent_state.get("is_enough", False)),
                    "rewritten_question": agent_state.get("rewritten_question", ""),
                    "error": agent_state.get("error", ""),
                }

                if answer:
                    st.markdown(answer)
                else:
                    st.warning(
                        "Agent 没有生成最终答案，请查看下方过程面板中的错误信息。"
                    )

                render_agent_trace(agent_trace, agent_meta, len(chunks))
                render_sources(chunks)

            else:
                if retrieval_mode == MODE_VECTOR:
                    spinner_text = "正在向量检索并生成回答..."
                else:
                    spinner_text = "正在混合检索、重排并生成回答..."

                with st.spinner(spinner_text):
                    retrieval_query = build_retrieval_query(prompt, history)
                    if retrieval_mode == MODE_VECTOR:
                        chunks = rag_service.retrieve(
                            retrieval_query, top_k=TOP_K
                        )
                    else:
                        chunks = rag_service.retrieve_hybrid_rerank(
                            retrieval_query,
                            top_k=TOP_K,
                            candidate_k=HYBRID_CANDIDATE_K,
                            score_threshold=RERANK_SCORE_THRESHOLD,
                        )

                    if not chunks:
                        st.warning(
                            "知识库中没有检索到相关内容。"
                            "请确认文档已入库，或换一种问法后重试。"
                        )
                    answer = generate_answer(
                        prompt, chunks, history=history
                    )

                st.markdown(answer)
                render_sources(chunks)
                if retrieval_mode == MODE_HYBRID and 0 < len(chunks) < TOP_K:
                    st.caption(
                        f"阈值 {RERANK_SCORE_THRESHOLD} 截断后保留 {len(chunks)} 条上下文。"
                    )

        except Exception as exc:
            st.error(f"处理失败：{type(exc).__name__}: {exc}")
            st.info(
                "请检查网络，以及 .env 中的 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL "
                "是否正确，或稍后重试。"
            )

        if answer:
            message = {"role": "assistant", "content": answer, "chunks": chunks}
            if agent_trace:
                message["agent_trace"] = agent_trace
                message["agent_meta"] = agent_meta or {}
            st.session_state.messages.append(message)
