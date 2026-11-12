"""Agentic RAG 工作流节点（当前仅前两个）。

本模块只包含可独立调用的节点函数，不包含 LangGraph 的图结构：

- ``query_rewriter_node``：把中文问题改写为更适合中英文检索的查询；
- ``retriever_node``：调用现有混合检索 + Reranker 链路召回上下文。

节点返回的是“状态增量”字典，由调用方（或后续的 LangGraph）合并进 AgentState。
"""

from __future__ import annotations

import json
import logging
import re
import threading

from backend.agent.state import AgentState
from backend.generation.answer_generator import generate_answer as generate_rag_answer
from backend.generation.history import format_history
from backend.generation.llm_client import LLMClient
from backend.retrieval.vector_store import get_or_create_collection
from backend.services.rag_service import RAGService

TOP_K = 5
CANDIDATE_K = 20
SCORE_THRESHOLD = 0.3
MAX_RETRIES = 2
REFUSAL_ANSWER = "根据现有资料无法回答该问题（或证据不足）。"
ADJACENT_RADIUS = 2
# 实测 300 字会把 Trade-In 参与方式的关键句（trade-in URL、积分说明）截掉；
# 500 字仍可控，同时保留完整的语义。
CHUNK_PREVIEW_CHARS = 500

logger = logging.getLogger(__name__)

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

REWRITE_SYSTEM_PROMPT = """你是企业知识库检索的查询改写器。请把用户问题改写成适合中英文混合检索的查询。

规则：
1. 保留品牌名、产品名、数字等关键实体（例如 Velvera）。
2. 如果问题包含中文业务术语或专有名词，必须补充对应的英文关键词，例如：
   - 以旧换新 → Trade-In, Recycling
   - 回收 → Recycling
   - 潜水 → diving
   - 售后 → after-sales service
   - 保修 → warranty
3. 中文原词与英文关键词都要保留，方便 BM25 同时命中中英文文档。
4. 不要回答问题，不要解释，不要添加资料中没有的事实。
5. 只输出一行改写后的查询，不要引号、编号或“改写后：”之类的前缀。

示例：
输入：怎么申请售后保修？
输出：How to request after-sales service warranty 售后 保修 申请

6. 如果提供了【历史对话】，先结合历史理解当前问题中的代词或省略
   （例如“它”“这个产品”“its”），把指代对象补全成具体产品名/实体，
   再进行英文关键词扩展。
"""

EVIDENCE_SYSTEM_PROMPT = """你是企业知识库问答的证据审核员。请判断【参考资料】是否足以回答【用户问题】。

判断步骤：
1. 先识别问题的核心诉求：问的是哪个对象、动作、属性、地点或数据。
2. 检查资料是否直接给出该核心诉求的答案，而不是仅仅“同主题”。
3. 满足以下条件才判 enough=true：资料明确给出针对问题对象的具体动作、步骤、地点、数值、条件、原因或服务内容中的至少一项。即使资料不完整、缺少次要细节，也可以判 true。
4. 以下情况判 enough=false：
   - 资料只有同主题背景、产品介绍或参数罗列，没有回答问题的核心动作/原因/地点/服务；
   - 只出现了相关关键词，但信息与问题所问的对象或动作不匹配；
   - 资料与问题完全无关。
5. 不要因为缺少完整步骤或全部细节就判 false；判断的是“资料里是否有直接支撑答案的核心信息”。

示例：
- 问“如何拆卸键盘”，资料写了“存放前需拆卸、磁吸连接、无法识别时拆卸重连” → true（有直接动作/机制）。
- 问“为什么适合专业用途”，资料只列屏幕和处理器参数，没有解释专业用途的原因 → false（参数不等于原因）。
- 问“提供哪些产品服务”，资料只讲企业采购计划、没有售后/保修/维修服务 → false（服务类型不匹配）。

特别注意：对于询问“产品服务/售后服务/保修/维修”的问题，如果资料只描述 Corporate Booking Plan、企业批量采购或预订方案，必须判 false。企业采购方案不等于产品售后服务，即使资料里出现 service 字样也不能判 true。

只输出一个 JSON 对象，不要 Markdown 代码块，不要额外解释：
{"enough": true, "reason": "简要中文理由"}
或
{"enough": false, "reason": "简要中文理由"}
"""

_llm_client: LLMClient | None = None
_llm_lock = threading.Lock()


def _get_llm_client() -> LLMClient:
    """进程内复用 LLMClient。"""
    global _llm_client

    if _llm_client is not None:
        return _llm_client

    with _llm_lock:
        if _llm_client is None:
            _llm_client = LLMClient()
        return _llm_client


def _clean_rewrite(raw: str) -> str:
    """清理模型输出，只保留一行有效查询。"""
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return ""

    text = lines[0]
    for prefix in ("改写后：", "改写后:", "改写：", "改写:", "查询：", "查询:", "Query:", "query:"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break

    return text.strip().strip("\"'“”‘’")


def _rewrite_question(
    question: str,
    feedback: str = "",
    history_text: str = "",
) -> str:
    """调用 LLM 改写问题；可附带历史对话与上一轮失败反馈。"""
    user_content = ""
    if history_text:
        user_content += f"【历史对话】\n{history_text}\n\n"
    user_content += f"【当前问题】\n{question}\n"

    if feedback:
        user_content += (
            f"上一轮反馈：\n{feedback}\n"
            "请换一种思路改写，补充更多英文同义词、上位词或具体业务词，"
            "不要重复上一次的查询。\n"
        )
    user_content += "请输出改写后的检索查询："

    raw = _get_llm_client().chat(
        [
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.0,
        max_tokens=160,
    )
    return _clean_rewrite(raw)


def query_rewriter_node(state: AgentState) -> dict:
    """改写用户问题，返回 ``rewritten_question`` 与递增的 ``retry_count``。

    有历史对话时会结合历史消解“它/its”等指代；中文问题还会补充英文
    检索关键词。LLM 调用失败时回退到原问题并写入 ``error``。
    """
    question = (state.get("question") or "").strip()
    retry_count = int(state.get("retry_count", 0)) + 1
    history_text = format_history(state.get("history") or [])

    if not question:
        return {
            "rewritten_question": "",
            "retry_count": retry_count,
            "error": "用户问题为空，无法改写。",
        }

    # 无历史的纯英文问题通常不需要改写，直接作为检索查询
    if not _CJK_RE.search(question) and not history_text:
        return {"rewritten_question": question, "retry_count": retry_count}

    try:
        rewritten = _rewrite_question(
            question, history_text=history_text
        )
    except Exception as exc:  # 保持节点可独立调用，不中断工作流
        return {
            "rewritten_question": question,
            "retry_count": retry_count,
            "error": f"查询改写失败，已回退原问题：{type(exc).__name__}: {exc}",
        }

    if not rewritten:
        return {
            "rewritten_question": question,
            "retry_count": retry_count,
            "error": "查询改写结果为空，已回退原问题。",
        }

    return {"rewritten_question": rewritten, "retry_count": retry_count}


def query_rewrite_retry_node(state: AgentState) -> dict:
    """检索证据不足时的重试改写：带上一轮反馈，换一种关键词组合。

    - 若 ``retry_count >= MAX_RETRIES``，不再调用 LLM，写入 ``error``；
    - 否则再次改写，并把 ``retry_count`` 加一；
    - 改写失败时回退到上一轮查询或原问题，保证工作流可继续。
    """
    question = (state.get("question") or "").strip()
    retry_count = int(state.get("retry_count", 0))

    if retry_count >= MAX_RETRIES:
        return {
            "error": f"已达到最大重试次数（{retry_count}），不再改写。",
        }

    if not question:
        return {
            "error": "用户问题为空，无法重试改写。",
            "retry_count": retry_count + 1,
        }

    previous_query = (state.get("rewritten_question") or "").strip()
    history_text = format_history(state.get("history") or [])
    sources = []
    for chunk in (state.get("chunks") or [])[:5]:
        source = (chunk.get("metadata") or {}).get("source")
        if source:
            sources.append(str(source))

    feedback_parts = []
    if previous_query:
        feedback_parts.append(f"上一次查询：{previous_query}")
    feedback_parts.append("上一次检索结果不足以回答原始问题。")
    if sources:
        feedback_parts.append(f"上次命中的来源：{', '.join(sources)}")
    if state.get("error"):
        feedback_parts.append(f"上次错误：{state['error']}")
    feedback = "\n".join(feedback_parts)

    try:
        rewritten = _rewrite_question(
            question,
            feedback=feedback,
            history_text=history_text,
        )
    except Exception as exc:
        fallback = previous_query or question
        return {
            "rewritten_question": fallback,
            "retry_count": retry_count + 1,
            "error": f"重试改写失败，已回退上一轮查询：{type(exc).__name__}: {exc}",
        }

    if not rewritten:
        fallback = previous_query or question
        return {
            "rewritten_question": fallback,
            "retry_count": retry_count + 1,
            "error": "重试改写结果为空，已回退上一轮查询。",
        }

    return {
        "rewritten_question": rewritten,
        "retry_count": retry_count + 1,
        "error": "",
    }


def _expand_adjacent_chunks(chunks: list[dict]) -> list[dict]:
    """为每个命中 chunk 补充同页前后 2 个邻居（2-hop）。

    PDF 切分会把“Setup Overview / 门店列表 / 完整流程”切到相隔 1–2 个
    chunk 的位置，Top-K 检索只命中局部时，grader 和生成器都会缺信息。
    这里扩展 ±ADJACENT_RADIUS 个邻居，并保留 expanded 标记。
    """
    if not chunks:
        return chunks

    existing_ids = {
        str((chunk.get("metadata") or {}).get("chunk_id"))
        for chunk in chunks
        if (chunk.get("metadata") or {}).get("chunk_id")
    }
    pages: dict[tuple[str, object], set[int]] = {}

    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        source = metadata.get("source")
        page = metadata.get("page")
        index = metadata.get("chunk_index")
        if source is None or not isinstance(index, int):
            continue
        pages.setdefault((str(source), page), set()).add(index)

    if not pages:
        return chunks

    collection = get_or_create_collection()
    source_cache: dict[str, list[tuple[str, dict]]] = {}
    additions: list[dict] = []

    try:
        for (source, page), hit_indices in pages.items():
            if source not in source_cache:
                result = collection.get(
                    where={"source": source},
                    include=["documents", "metadatas"],
                )
                source_cache[source] = list(
                    zip(
                        result.get("documents") or [],
                        result.get("metadatas") or [],
                    )
                )

            page_chunks = {
                metadata["chunk_index"]: (text or "", metadata)
                for text, metadata in source_cache[source]
                if metadata.get("page") == page
                and isinstance(metadata.get("chunk_index"), int)
            }

            for hit_index in sorted(hit_indices):
                for offset in range(-ADJACENT_RADIUS, ADJACENT_RADIUS + 1):
                    if offset == 0:
                        continue
                    item = page_chunks.get(hit_index + offset)
                    if not item:
                        continue
                    text, metadata = item
                    chunk_id = metadata.get("chunk_id")
                    if chunk_id and str(chunk_id) in existing_ids:
                        continue
                    if chunk_id:
                        existing_ids.add(str(chunk_id))
                    additions.append(
                        {
                            "text": text,
                            "metadata": dict(metadata),
                            "expanded": True,
                        }
                    )
    except Exception:
        # 扩展失败不影响主检索链路
        return chunks

    return chunks + additions


def retriever_node(state: AgentState) -> dict:
    """用改写后的问题（为空则用原问题）执行混合检索 + 重排。

    Returns:
        ``{"chunks": [...]}``；检索异常时返回空列表并写入 ``error``。
    """
    query = (state.get("rewritten_question") or "").strip()
    if not query:
        query = (state.get("question") or "").strip()

    if not query:
        return {"chunks": [], "error": "检索查询为空。"}

    try:
        chunks = RAGService().retrieve_hybrid_rerank(
            query,
            top_k=TOP_K,
            candidate_k=CANDIDATE_K,
            score_threshold=SCORE_THRESHOLD,
        )
    except Exception as exc:
        return {
            "chunks": [],
            "error": f"检索失败：{type(exc).__name__}: {exc}",
        }

    return {"chunks": _expand_adjacent_chunks(chunks)}


def _build_evidence_context(chunks: list[dict]) -> str:
    """把检索片段拼成编号上下文，每条截取前 300 字。"""
    blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        metadata = chunk.get("metadata") or {}
        source = metadata.get("source", "unknown")
        page = metadata.get("page", "N/A")
        blocks.append(
            f"[{index}] source: {source}, page: {page}\n"
            f"{text[:CHUNK_PREVIEW_CHARS]}"
        )
    return "\n\n".join(blocks)


def _parse_evidence_json(raw: str) -> tuple[bool, str]:
    """解析 LLM 返回的 JSON；容忍 ```json 代码块包裹。"""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("返回内容中找不到 JSON 对象")

    data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("JSON 顶层不是对象")

    enough = data.get("enough")
    if isinstance(enough, str):
        enough = enough.strip().lower() == "true"
    else:
        enough = bool(enough)

    reason = str(data.get("reason", "")).strip()
    return enough, reason


def evidence_grader_node(state: AgentState) -> dict:
    """判断当前 chunks 是否足以回答原始 question。

    - chunks 为空：直接返回 ``{"is_enough": False}``，不调用 LLM；
    - 否则让 LLM 输出 ``{"enough": bool, "reason": str}``；
    - LLM 调用或 JSON 解析失败：安全返回 ``{"is_enough": False}``。

    LLM 的 reason 会写入 ``backend.agent.nodes`` 日志，便于调试。
    """
    chunks = state.get("chunks") or []
    if not chunks:
        logger.info("[evidence_grader] enough=False reason=检索结果为空，无需调用 LLM")
        return {"is_enough": False}

    question = (state.get("question") or "").strip()
    if not question:
        logger.info("[evidence_grader] enough=False reason=用户问题为空")
        return {"is_enough": False}

    context = _build_evidence_context(chunks)
    if not context:
        logger.info("[evidence_grader] enough=False reason=参考资料为空")
        return {"is_enough": False}

    raw = ""
    try:
        raw = _get_llm_client().chat(
            [
                {"role": "system", "content": EVIDENCE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"【用户问题】\n{question}\n\n"
                        f"【参考资料】\n{context}\n\n"
                        "请判断这些资料是否足以回答该问题，并按要求输出 JSON。"
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=256,
        )
        enough, reason = _parse_evidence_json(raw)
    except Exception as exc:
        logger.info(
            "[evidence_grader] enough=False reason=判断失败: %s: %s | raw=%s",
            type(exc).__name__,
            exc,
            raw[:200],
        )
        return {"is_enough": False}

    logger.info(
        "[evidence_grader] enough=%s reason=%s",
        enough,
        reason or "(LLM 未给出 reason)",
    )
    return {"is_enough": enough}


def generate_answer_node(state: AgentState) -> dict:
    """基于原始问题和已检索 chunks 生成最终答案。"""
    question = (state.get("question") or "").strip()
    chunks = state.get("chunks") or []

    try:
        answer = generate_rag_answer(
            question,
            chunks,
            history=state.get("history") or [],
        )
    except Exception as exc:
        return {
            "answer": "抱歉，生成答案时出现错误，请稍后重试。",
            "error": f"生成答案失败：{type(exc).__name__}: {exc}",
        }

    return {"answer": answer}


def refusal_node(state: AgentState) -> dict:
    """证据不足时的固定拒答。"""
    return {"answer": REFUSAL_ANSWER}
