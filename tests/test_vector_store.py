"""测试脚本：验证 Chroma 向量库的写入、检索、删除与持久化。

用法（在项目根目录 C:\\RAG\\P1 下执行）：
    C:\\RAG\\P1\\Prag\\Scripts\\python.exe tests\\test_vector_store.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Windows 控制台默认编码可能不是 UTF-8，统一输出编码避免中文乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.ingestion.document_loader import load_document  # noqa: E402
from backend.ingestion.text_splitter import split_documents  # noqa: E402
from backend.retrieval.embedding import embed_query, embed_texts  # noqa: E402
from backend.retrieval.vector_store import (  # noqa: E402
    add_chunks_to_store,
    delete_documents_by_source,
    get_document_count,
    search_similar_chunks,
)
from config.settings import settings  # noqa: E402

SAMPLE_DIR = BASE_DIR / "data" / "sample_docs"
PREVIEW_LENGTH = 50
QUERY = "What is the price of the Velvera 65W GaN Charger?"


def pick_sample_pdf(directory: Path) -> Path:
    pdf_files = sorted(path for path in directory.glob("*.pdf") if path.is_file())
    if not pdf_files:
        raise SystemExit(f"未找到测试用的 PDF 文件，请检查目录: {directory}")
    return pdf_files[0]


def preview(text: str, limit: int = PREVIEW_LENGTH) -> str:
    snippet = text[:limit].replace("\r", " ").replace("\n", " ")
    return f"{snippet}..." if len(text) > limit else snippet


def main() -> int:
    pdf_path = pick_sample_pdf(SAMPLE_DIR)
    source = pdf_path.name
    print(f"测试文件: {source}")

    # 1) 清理该来源的旧数据
    count_before_clean = get_document_count()
    delete_documents_by_source(source)
    count_after_clean = get_document_count()
    print(
        f"清理来源 {source!r}: 清理前 {count_before_clean} 条，"
        f"清理后 {count_after_clean} 条"
    )
    if count_after_clean:
        print(f"注意: 库中仍有其他来源的 {count_after_clean} 条记录（本次不清理）")
    print()

    # 2) 解析 -> 切分 -> 向量化
    documents = load_document(str(pdf_path))
    chunks = split_documents(documents)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(texts)
    print(f"解析: {len(documents)} 条页面文本")
    print(f"切分: {len(chunks)} 个 chunk")
    print(f"向量化: {len(embeddings)} 条向量，维度 {len(embeddings[0]) if embeddings else 0}")

    # 3) 写入前后数量
    count_before_add = get_document_count()
    add_chunks_to_store(chunks, embeddings)
    count_after_add = get_document_count()
    print(f"写入: 写入前 {count_before_add} 条，写入后 {count_after_add} 条")
    assert count_after_add - count_before_add == len(chunks), "新增记录数与 chunk 数不一致"

    # 空列表应当直接返回，不改变库内容
    add_chunks_to_store([])
    assert get_document_count() == count_after_add, "空列表写入不应改变记录数"
    print("空列表校验: add_chunks_to_store([]) 未改变记录数 ✅")
    print()

    # 4) 检索 Top-3
    query_embedding = embed_query(QUERY)
    results = search_similar_chunks(query_embedding, top_k=3)
    print(f"查询: {QUERY}")
    print(f"命中 {len(results)} 条（余弦距离，越小越相似）:")
    for rank, hit in enumerate(results, start=1):
        meta = hit["metadata"]
        print(f"  [{rank}] source={meta.get('source')} | page={meta.get('page')} "
              f"| distance={hit['distance']:.4f} | chunk_id={meta.get('chunk_id')}")
        print(f"      {preview(hit['text'])}")
    assert results, "检索结果为空"
    print()

    # 5) 持久化说明
    total = get_document_count()
    print(f"脚本结束前记录数: {total}")
    print(
        f"数据已持久化到 {settings.CHROMA_PATH}，"
        "进程重启后仍可读取（可另开进程调用 get_document_count() 验证）。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
