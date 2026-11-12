"""BM25 关键词检索。

从 Chroma 中导出全部 chunk，构建进程内 BM25（Okapi）索引，提供与向量检索
一致的结果结构：``{"text", "metadata", "distance"}``。

注意：BM25 得分越大越相关，这里额外返回原始 ``score``，并把
``distance`` 定义为 ``1 - score``（同样满足“越小越相似”的约定）。
"""

from __future__ import annotations

import re
import threading

from rank_bm25 import BM25Okapi

from backend.retrieval.vector_store import (
    get_data_version,
    get_document_count,
    get_or_create_collection,
)

# 英文/数字按词切分；中文按单字切分（当前语料以英文为主）
_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")

# 少量英文停用词，避免 what/is/the 之类的词稀释关键词权重
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "can", "could", "did", "do", "does", "for", "from", "had", "has", "have",
    "he", "her", "here", "him", "his", "how", "i", "if", "in", "into", "is",
    "it", "its", "me", "my", "of", "on", "or", "our", "ours", "she", "should",
    "so", "than", "that", "the", "their", "theirs", "them", "then", "there",
    "these", "they", "this", "those", "to", "was", "we", "were", "what",
    "when", "where", "which", "who", "why", "will", "with", "would", "you",
    "your", "yours", "s",
}

_FETCH_BATCH_SIZE = 500

_lock = threading.Lock()
_bm25: BM25Okapi | None = None
_chunks: list[dict] = []
_indexed_count = -1
_indexed_version = -1


def _tokenize(text: str) -> list[str]:
    """小写化并切分为检索 token。"""
    tokens = _TOKEN_RE.findall(text.lower())
    return [token for token in tokens if token not in _STOPWORDS]


def _load_chunks_from_chroma() -> list[dict]:
    """分页导出 Chroma 中的全部文本与元数据。"""
    collection = get_or_create_collection()
    chunks: list[dict] = []
    offset = 0

    while True:
        result = collection.get(
            limit=_FETCH_BATCH_SIZE,
            offset=offset,
            include=["documents", "metadatas"],
        )
        ids = result.get("ids") or []
        if not ids:
            break

        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        for text, metadata in zip(documents, metadatas):
            chunks.append(
                {"text": text or "", "metadata": dict(metadata or {})}
            )

        offset += len(ids)

    return chunks


def _ensure_index(force: bool = False) -> bool:
    """按需构建/复用索引；集合记录数变化时自动重建。

    Returns:
        True 表示索引可用；False 表示知识库为空。
    """
    global _bm25, _chunks, _indexed_count, _indexed_version

    current_count = get_document_count()
    current_version = get_data_version()
    index_fresh = (
        _bm25 is not None
        and _indexed_count == current_count
        and _indexed_version == current_version
    )
    if not force and index_fresh:
        return True

    with _lock:
        # 双检：并发时避免重复构建
        current_count = get_document_count()
        current_version = get_data_version()
        index_fresh = (
            _bm25 is not None
            and _indexed_count == current_count
            and _indexed_version == current_version
        )
        if not force and index_fresh:
            return True

        # 记录开始构建时的版本；若构建期间发生写入，下次查询会因版本不一致而重建
        build_version = current_version
        chunks = _load_chunks_from_chroma()
        if not chunks:
            _bm25 = None
            _chunks = []
            _indexed_count = 0
            _indexed_version = build_version
            return False

        tokenized_corpus = [_tokenize(chunk["text"]) for chunk in chunks]
        if not any(tokenized_corpus):
            _bm25 = None
            _chunks = chunks
            _indexed_count = len(chunks)
            _indexed_version = build_version
            return False

        _bm25 = BM25Okapi(tokenized_corpus)
        _chunks = chunks
        _indexed_count = len(chunks)
        _indexed_version = build_version
        return True


def invalidate_index() -> None:
    """手动使索引失效，下次检索时重建。

    正常情况下无需调用：``vector_store`` 的写入/删除会递增数据版本号，
    ``_ensure_index`` 会自动发现版本变化并重建。
    """
    global _indexed_version

    with _lock:
        _indexed_version = -1


def rebuild_index() -> int:
    """强制重建 BM25 索引，返回参与建索引的 chunk 数。"""
    _ensure_index(force=True)
    return len(_chunks)


def search_bm25(query: str, top_k: int = 5) -> list[dict]:
    """BM25 关键词检索。

    Args:
        query: 用户问题。
        top_k: 返回条数。

    Returns:
        ``[{"text", "metadata", "distance", "score"}]``，按 BM25 得分降序；
        只返回得分大于 0（即至少命中一个关键词）的 chunk。
    """
    if not isinstance(query, str) or not query.strip() or top_k <= 0:
        return []

    if not _ensure_index():
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    assert _bm25 is not None
    scores = _bm25.get_scores(query_tokens)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    results: list[dict] = []
    for index in ranked:
        score = float(scores[index])
        if score <= 0:
            break  # 已按降序排列，后续都不相关

        chunk = _chunks[index]
        results.append(
            {
                "text": chunk["text"],
                "metadata": dict(chunk["metadata"]),
                "distance": 1.0 - score,
                "score": score,
            }
        )
        if len(results) >= top_k:
            break

    return results
