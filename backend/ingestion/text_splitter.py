"""文本切分模块。

把 ``load_document`` 返回的页面/文档级文本，切分为适合向量化的 chunk。
切分参数与分隔符按 RAG MVP 约定固定为 500 / 80 字符和中文标点优先。
"""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
SEPARATORS = ["\n\n", "\n", "。", "！", "？", " ", ""]

# 无状态、线程安全，模块加载时构建一次即可复用
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=SEPARATORS,
    length_function=len,
)


def split_documents(documents: list[dict]) -> list[dict]:
    """把文档列表切分为 chunk 列表。

    每个输入元素来自 :func:`backend.ingestion.document_loader.load_document`，
    形如 ``{"text": str, "metadata": {...}}``。每个输出元素保留原有元数据，
    并追加 ``chunk_index``（页/文档内从 0 开始的序号）和 ``chunk_id``
    （``{source}_p{page}_c{chunk_index}``）。

    Args:
        documents: 待切分的文档列表。

    Returns:
        切片列表；空白切片会被丢弃，输入为空时返回空列表。
    """
    chunks: list[dict] = []

    for document in documents:
        text = document.get("text", "")
        if not text or not text.strip():
            continue

        # 复制元数据，避免多个 chunk 共享同一个 dict
        base_metadata = dict(document.get("metadata") or {})
        source = base_metadata.get("source", "unknown")
        page = base_metadata.get("page", "N/A")

        chunk_index = 0
        for piece in _splitter.split_text(text):
            piece = piece.strip()
            if not piece:  # 理论上不会出现，防御性过滤
                continue

            metadata = dict(base_metadata)
            metadata["chunk_index"] = chunk_index
            metadata["chunk_id"] = f"{source}_p{page}_c{chunk_index}"

            chunks.append({"text": piece, "metadata": metadata})
            chunk_index += 1

    return chunks
