"""文档入库服务。

把「解析 → 切分 → 向量化 → 写入 Chroma」串成一个入口，供 API/脚本复用。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from backend.ingestion.document_loader import load_document
from backend.ingestion.text_splitter import split_documents
from backend.retrieval.embedding import embed_texts
from backend.retrieval.vector_store import (
    add_chunks_to_store,
    clear_store,
    delete_documents_by_source,
    get_document_count,
    get_or_create_collection,
)


class DocumentService:
    """文档入库与维护。"""

    def ingest_file(self, file_path: str) -> int:
        """解析并入库单个文件，返回写入的 chunk 数。

        采用「先向量化、再删旧、后写入」的顺序：
        - 向量化失败时保留库中原有数据；
        - 写入前按 ``source`` 删除旧切片，因此重复入库不会产生重复记录。

        Args:
            file_path: PDF / DOCX 文件路径。

        Returns:
            实际写入的 chunk 数；文档没有可提取文本时返回 0。
        """
        path = Path(file_path)
        source = path.name

        documents = load_document(str(path))
        chunks = split_documents(documents)

        if not chunks:
            # 文档变为空白时，清掉历史切片，避免残留
            delete_documents_by_source(source)
            return 0

        embeddings = embed_texts([chunk["text"] for chunk in chunks])

        delete_documents_by_source(source)
        add_chunks_to_store(chunks, embeddings)
        return len(chunks)

    def delete_file(self, source_name: str) -> None:
        """按来源文件名删除该文件的全部切片。"""
        delete_documents_by_source(source_name)

    def get_total_chunks(self) -> int:
        """返回向量库中的总块数。"""
        return get_document_count()

    def list_documents(self) -> list[dict]:
        """返回知识库中每个来源文件及其 chunk 数，按文件名排序。"""
        collection = get_or_create_collection()
        counts: Counter[str] = Counter()
        offset = 0
        batch_size = 1000

        while True:
            result = collection.get(
                limit=batch_size,
                offset=offset,
                include=["metadatas"],
            )
            ids = result.get("ids") or []
            if not ids:
                break

            for metadata in result.get("metadatas") or []:
                source = (metadata or {}).get("source")
                if source:
                    counts[str(source)] += 1

            offset += len(ids)

        return [
            {"source": source, "chunk_count": count}
            for source, count in sorted(counts.items())
        ]

    def clear_all(self) -> None:
        """清空知识库中的全部切片（保留集合配置）。"""
        clear_store()
