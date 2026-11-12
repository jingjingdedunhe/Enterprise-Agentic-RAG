"""Chroma 向量库封装。

持久化目录来自 ``config.settings.CHROMA_PATH``（默认 ``data/chroma_db``）。
集合使用余弦距离（Chroma 返回的 distance = 1 - 余弦相似度，越小越相似）。
"""

from __future__ import annotations

import threading
from pathlib import Path

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings

from config.settings import settings

COLLECTION_NAME = "enterprise_rag"
DISTANCE_METRIC = "cosine"

_client: ClientAPI | None = None
_collection: Collection | None = None
# 可重入：get_or_create_collection() 持锁时会再次调用 get_chroma_client()
_lock = threading.RLock()

# 数据版本号：任何写入/删除都会递增，供 BM25 等进程内索引判断是否需要重建。
# 仅靠 count() 无法识别“同数量替换”（先删 11 条再加 11 条）。
_data_version = 0


def get_data_version() -> int:
    """返回知识库数据版本号。"""
    return _data_version


def _bump_data_version() -> None:
    global _data_version
    with _lock:
        _data_version += 1



def get_chroma_client() -> ClientAPI:
    """返回进程内共享的 Chroma 持久化客户端（单例）。"""
    global _client

    if _client is not None:
        return _client

    with _lock:
        if _client is not None:
            return _client

        db_path = Path(settings.CHROMA_PATH)
        try:
            db_path.mkdir(parents=True, exist_ok=True)
            _client = chromadb.PersistentClient(
                path=str(db_path),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        except Exception as exc:
            raise RuntimeError(
                f"连接 Chroma 向量库失败，请检查路径和读写权限: {db_path}\n"
                f"原始错误：{type(exc).__name__}: {exc}"
            ) from exc

        return _client


def get_or_create_collection() -> Collection:
    """返回向量库集合（单例），首次创建时指定余弦距离。"""
    global _collection

    if _collection is not None:
        return _collection

    with _lock:
        if _collection is not None:
            return _collection

        try:
            _collection = get_chroma_client().get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": DISTANCE_METRIC},
            )
        except Exception as exc:
            raise RuntimeError(
                f"打开 Chroma 集合失败: {COLLECTION_NAME}\n"
                f"原始错误：{type(exc).__name__}: {exc}"
            ) from exc

        return _collection


def add_chunks_to_store(
    chunks: list[dict],
    embeddings: list[list[float]] | None = None,
) -> None:
    """把切分后的 chunk 写入 Chroma。

    ``chunk_id`` 作为 Chroma 的 id，``text`` 作为 documents，
    ``metadata`` 原样作为 metadatas。

    Args:
        chunks: ``split_documents`` 的输出，metadata 中必须含 ``chunk_id``。
        embeddings: 与 chunks 一一对应的向量；不传时自动调用
            :func:`backend.retrieval.embedding.embed_texts` 计算。
    """
    if not chunks:
        return

    if embeddings is None:
        from backend.retrieval.embedding import embed_texts

        embeddings = embed_texts([chunk["text"] for chunk in chunks])

    if len(embeddings) != len(chunks):
        raise ValueError(
            f"embeddings 数量（{len(embeddings)}）与 chunks 数量（{len(chunks)}）不一致"
        )

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for index, chunk in enumerate(chunks):
        metadata = dict(chunk.get("metadata") or {})
        chunk_id = metadata.get("chunk_id")
        if not chunk_id:
            raise ValueError(f"第 {index} 个 chunk 缺少 metadata['chunk_id']")

        ids.append(str(chunk_id))
        documents.append(chunk["text"])
        metadatas.append(metadata)

    get_or_create_collection().add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    _bump_data_version()


def search_similar_chunks(
    query_embedding: list[float],
    top_k: int = 5,
) -> list[dict]:
    """按查询向量检索最相似的 Top-K 个文本块。

    Returns:
        ``[{"text": str, "metadata": dict, "distance": float}]``，
        ``distance`` 为余弦距离（0 表示完全一致）。
    """
    if not query_embedding or top_k <= 0:
        return []

    collection = get_or_create_collection()
    if collection.count() == 0:
        return []

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = result.get("documents") or []
    if not documents:
        return []

    metadatas = result.get("metadatas") or [[]]
    distances = result.get("distances") or [[]]

    hits: list[dict] = []
    for text, metadata, distance in zip(documents[0], metadatas[0], distances[0]):
        hits.append(
            {
                "text": text,
                "metadata": metadata or {},
                "distance": float(distance),
            }
        )
    return hits


def delete_documents_by_source(source: str) -> None:
    """删除某个来源文件（如 ``accessories.pdf``）对应的全部记录。"""
    get_or_create_collection().delete(where={"source": source})
    _bump_data_version()


def get_document_count() -> int:
    """返回集合中的记录总数。"""
    return get_or_create_collection().count()


def clear_store(batch_size: int = 1000) -> None:
    """清空集合中的全部记录，但保留集合本身及其 cosine 配置。

    分批读取 id 再删除，避免一次请求携带过多 id。
    """
    collection = get_or_create_collection()

    while True:
        ids = collection.get(limit=batch_size, include=[]).get("ids") or []
        if not ids:
            break
        collection.delete(ids=ids)

    _bump_data_version()

