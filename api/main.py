"""Enterprise Agentic RAG - FastAPI 后端服务。

提供：
- GET  /health       健康检查
- POST /chat         基础问答（vector / hybrid）
- POST /agent/chat   Agentic RAG 工作流问答

启动：
    C:\\RAG\\P1\\Prag\\Scripts\\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.concurrency import run_in_threadpool

from backend.agent.workflow import invoke_workflow
from backend.generation.answer_generator import generate_answer
from backend.generation.history import build_retrieval_query
from backend.services.document_service import DocumentService
from backend.services.rag_service import RAGService
from config.settings import settings

logger = logging.getLogger(__name__)

TOP_K = 5
CANDIDATE_K = 20
SCORE_THRESHOLD = 0.3
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_QUERY_LENGTH = 1000
ALLOWED_SUFFIXES = {".pdf", ".docx"}
SCRIPT_PATTERN = re.compile(r"<\s*script\b", re.IGNORECASE)


class SecretRedactionFilter(logging.Filter):
    """确保日志消息中不会出现配置的 LLM API Key。"""

    def __init__(self, secret: str) -> None:
        super().__init__()
        self.secret = secret

    def filter(self, record: logging.LogRecord) -> bool:
        if not self.secret:
            return True
        try:
            message = record.getMessage()
        except Exception:
            return True
        if self.secret in message:
            record.msg = message.replace(self.secret, "***")
            record.args = ()
        return True


def redact_secret(text: str) -> str:
    secret = settings.LLM_API_KEY
    if secret:
        return text.replace(secret, "***")
    return text


def log_exception(context: str, exc: Exception) -> None:
    """记录异常，但不输出可能包含密钥的 traceback/消息。"""
    logger.error(
        "%s: %s: %s",
        context,
        type(exc).__name__,
        redact_secret(str(exc)),
    )


def install_secret_redaction() -> None:
    """把脱敏过滤器挂到现存日志 handler 和常用 logger 上。"""
    if not settings.LLM_API_KEY:
        return
    redaction = SecretRedactionFilter(settings.LLM_API_KEY)

    for handler in logging.getLogger().handlers:
        handler.addFilter(redaction)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "api", "backend"):
        target = logging.getLogger(name)
        target.addFilter(redaction)
        for handler in target.handlers:
            handler.addFilter(redaction)


install_secret_redaction()

app = FastAPI(
    title="Enterprise Agentic RAG API",
    description="Velvera 企业知识库检索与 Agentic RAG 问答服务",
    version="0.1.0",
)

rag_service = RAGService()
document_service = DocumentService()


# ---------------------------- Pydantic 模型 ----------------------------
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="用户问题")

    @field_validator("query")
    @classmethod
    def _validate_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query 不能为空")
        if len(value) > MAX_QUERY_LENGTH:
            raise ValueError(f"query 不能超过 {MAX_QUERY_LENGTH} 个字符")
        if SCRIPT_PATTERN.search(value):
            raise ValueError("query 包含不允许的脚本内容")
        return value


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)


class ChatRequest(QueryRequest):
    mode: Literal["vector", "hybrid"] = Field(
        default="hybrid",
        description="检索模式：vector=基础向量；hybrid=混合检索+重排",
    )
    history: list[HistoryMessage] = Field(
        default_factory=list,
        description="最近几轮对话（role/content），用于指代消解",
    )


class AgentChatRequest(QueryRequest):
    history: list[HistoryMessage] = Field(
        default_factory=list,
        description="最近几轮对话（role/content），用于指代消解",
    )


class HealthResponse(BaseModel):
    status: str


class Citation(BaseModel):
    source: str
    page: int | str
    chunk_id: str | None = None
    score: float | None = None
    score_type: str | None = None


class ChunkMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    source: str
    page: int | str
    file_type: str | None = None
    chunk_id: str | None = None
    chunk_index: int | None = None


class Chunk(BaseModel):
    model_config = ConfigDict(extra="allow")

    text: str
    metadata: ChunkMetadata
    distance: float | None = None
    rerank_score: float | None = None
    rrf_score: float | None = None
    rrf_rank: int | None = None
    vector_rank: int | None = None
    bm25_rank: int | None = None
    expanded: bool | None = None


class ChatResponse(BaseModel):
    answer: str
    mode: str
    citations: list[Citation]


class AgentChatResponse(BaseModel):
    question: str
    rewritten_question: str
    chunks: list[Chunk]
    is_enough: bool
    retry_count: int
    answer: str
    citations: list[Citation]
    error: str


class DocumentInfo(BaseModel):
    source: str
    chunk_count: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total_documents: int
    total_chunks: int


class DocumentUploadResponse(BaseModel):
    source: str
    chunks_added: int
    total_chunks: int


class DocumentDeleteResponse(BaseModel):
    source: str
    deleted_chunks: int
    remaining_chunks: int


# ---------------------------- 工具函数 ----------------------------
def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def build_citations(chunks: list[dict], answer: str = "") -> list[Citation]:
    """从答案的引用编号映射到 chunk 元数据，并按 chunk_id 去重。

    答案里的 ``[1] source: ...`` 编号对应生成时的上下文顺序（chunks 顺序）。
    如果答案明确拒答，则返回空引用；如果模型没有输出编号，则回退到全部
    chunk，避免引用信息完全丢失。
    """
    if "根据现有资料无法回答" in answer or "引用来源：无" in answer:
        return []

    cited_indices = [
        int(match.group(1))
        for match in re.finditer(r"\[(\d+)\]\s*source:", answer or "")
    ]
    if cited_indices:
        selected: list[dict] = []
        for index in cited_indices:
            if 1 <= index <= len(chunks):
                selected.append(chunks[index - 1])
        if selected:
            chunks = selected

    citations: list[Citation] = []
    seen: set[str] = set()

    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        source = str(metadata.get("source", "unknown"))
        page = metadata.get("page", "N/A")
        chunk_id = metadata.get("chunk_id")
        key = str(chunk_id) if chunk_id else f"{source}|{page}"
        if key in seen:
            continue
        seen.add(key)

        score = None
        score_type = None
        if _number(chunk.get("rerank_score")) is not None:
            score = _number(chunk.get("rerank_score"))
            score_type = "rerank_score"
        elif _number(chunk.get("rrf_score")) is not None:
            score = _number(chunk.get("rrf_score"))
            score_type = "rrf_score"
        elif _number(chunk.get("distance")) is not None:
            score = _number(chunk.get("distance"))
            score_type = "distance"

        citations.append(
            Citation(
                source=source,
                page=page if isinstance(page, (int, str)) else str(page),
                chunk_id=str(chunk_id) if chunk_id else None,
                score=score,
                score_type=score_type,
            )
        )

    return citations


def collect_document_infos() -> list[DocumentInfo]:
    """从 DocumentService 获取每个 source 的 chunk 数。"""
    return [
        DocumentInfo(**item)
        for item in document_service.list_documents()
    ]


# ---------------------------- 接口 ----------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """基础问答：检索 Top-5 后调用生成器。"""
    try:
        history = [message.model_dump() for message in request.history]
        retrieval_query = build_retrieval_query(request.query, history)

        if request.mode == "vector":
            chunks = rag_service.retrieve(retrieval_query, top_k=TOP_K)
        else:
            chunks = rag_service.retrieve_hybrid_rerank(
                retrieval_query,
                top_k=TOP_K,
                candidate_k=CANDIDATE_K,
                score_threshold=SCORE_THRESHOLD,
            )

        answer = generate_answer(request.query, chunks, history=history)
        return ChatResponse(
            answer=answer,
            mode=request.mode,
            citations=build_citations(chunks, answer),
        )
    except Exception as exc:
        log_exception("/chat failed", exc)
        raise HTTPException(
            status_code=500,
            detail=redact_secret(
                f"问答处理失败：{type(exc).__name__}: {exc}"
            ),
        ) from exc


@app.post("/agent/chat", response_model=AgentChatResponse)
def agent_chat(request: AgentChatRequest) -> AgentChatResponse:
    """Agentic RAG：改写 -> 检索 -> 证据判断 -> 重试/生成/拒答。"""
    try:
        history = [message.model_dump() for message in request.history]
        state = invoke_workflow(request.query, history=history)
        chunks = state.get("chunks") or []
        answer = state.get("answer", "")

        return AgentChatResponse(
            question=state.get("question", request.query),
            rewritten_question=state.get("rewritten_question", ""),
            chunks=chunks,
            is_enough=bool(state.get("is_enough", False)),
            retry_count=int(state.get("retry_count", 0)),
            answer=answer,
            citations=build_citations(chunks, answer),
            error=state.get("error", ""),
        )
    except Exception as exc:
        log_exception("/agent/chat failed", exc)
        raise HTTPException(
            status_code=500,
            detail=redact_secret(
                f"Agent 工作流处理失败：{type(exc).__name__}: {exc}"
            ),
        ) from exc


@app.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(..., description="PDF 或 DOCX 文件，最大 20MB"),
) -> DocumentUploadResponse:
    """上传文档并入库，返回新增 chunk 数。"""
    filename = Path(file.filename or "").name
    if not filename:
        raise HTTPException(status_code=400, detail="缺少文件名。")

    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {suffix or '(无扩展名)'}，仅允许 .pdf / .docx",
        )

    try:
        with tempfile.TemporaryDirectory(prefix="rag_upload_") as temp_dir:
            temp_path = Path(temp_dir) / filename
            total_bytes = 0

            with open(temp_path, "wb") as output:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    if total_bytes > MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail="文件超过 20MB 限制。",
                        )
                    output.write(chunk)

            added = await run_in_threadpool(
                document_service.ingest_file, str(temp_path)
            )
            total_chunks = await run_in_threadpool(
                document_service.get_total_chunks
            )

        return DocumentUploadResponse(
            source=filename,
            chunks_added=int(added),
            total_chunks=int(total_chunks),
        )
    except HTTPException:
        raise
    except Exception as exc:
        log_exception("/documents/upload failed", exc)
        raise HTTPException(
            status_code=500,
            detail=redact_secret(
                f"文档入库失败：{type(exc).__name__}: {exc}"
            ),
        ) from exc
    finally:
        await file.close()


@app.get("/documents", response_model=DocumentListResponse)
def list_documents() -> DocumentListResponse:
    """列出知识库中的文档及各自 chunk 数。"""
    try:
        documents = collect_document_infos()
        return DocumentListResponse(
            documents=documents,
            total_documents=len(documents),
            total_chunks=sum(item.chunk_count for item in documents),
        )
    except Exception as exc:
        log_exception("/documents failed", exc)
        raise HTTPException(
            status_code=500,
            detail=redact_secret(
                f"读取文档列表失败：{type(exc).__name__}: {exc}"
            ),
        ) from exc


@app.delete("/documents/{source_name}", response_model=DocumentDeleteResponse)
def delete_document(source_name: str) -> DocumentDeleteResponse:
    """按来源文件名删除文档的全部 chunk。"""
    if not source_name or Path(source_name).name != source_name:
        raise HTTPException(status_code=400, detail="非法的文件名。")

    try:
        before = document_service.get_total_chunks()
        document_service.delete_file(source_name)
        after = document_service.get_total_chunks()
        deleted = int(before - after)

        if deleted <= 0:
            raise HTTPException(
                status_code=404,
                detail=f"未找到文档: {source_name}",
            )

        return DocumentDeleteResponse(
            source=source_name,
            deleted_chunks=deleted,
            remaining_chunks=int(after),
        )
    except HTTPException:
        raise
    except Exception as exc:
        log_exception("/documents/{source_name} failed", exc)
        raise HTTPException(
            status_code=500,
            detail=redact_secret(
                f"删除文档失败：{type(exc).__name__}: {exc}"
            ),
        ) from exc


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """输入校验失败统一返回 400（长度、脚本、空 query 等）。"""
    errors = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", []))
        message = str(error.get("msg", "输入无效"))
        errors.append(f"{location}: {message}")

    return JSONResponse(
        status_code=400,
        content={
            "detail": "无效的用户输入",
            "errors": errors,
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """兜底异常处理：避免 FastAPI 默认隐藏错误细节。"""
    log_exception(
        f"Unhandled error: {request.method} {request.url.path}", exc
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": redact_secret(
                f"服务器内部错误：{type(exc).__name__}: {exc}"
            ),
        },
    )
