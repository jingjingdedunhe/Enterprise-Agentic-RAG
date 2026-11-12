"""测试脚本：验证 DocumentService 与 RAGService 的完整流程。

用法（在项目根目录 C:\\RAG\\P1 下执行）：
    C:\\RAG\\P1\\Prag\\Scripts\\python.exe tests\\test_services.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Windows 控制台默认编码可能不是 UTF-8，统一输出编码避免中文乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.services.document_service import DocumentService  # noqa: E402
from backend.services.rag_service import RAGService  # noqa: E402

SAMPLE_DIR = BASE_DIR / "data" / "sample_docs"
QUERY = "What is the price of the Velvera 65W GaN Charger?"
PREVIEW_LENGTH = 120


def preview(text: str, limit: int = PREVIEW_LENGTH) -> str:
    snippet = " ".join(text.split())
    return f"{snippet[:limit]}..." if len(snippet) > limit else snippet


def main() -> int:
    pdf_files = sorted(path for path in SAMPLE_DIR.glob("*.pdf") if path.is_file())
    if not pdf_files:
        raise SystemExit(f"未找到 PDF 文件，请检查目录: {SAMPLE_DIR}")

    service = DocumentService()
    rag = RAGService()

    print(f"待入库 PDF: {len(pdf_files)} 个")
    print(f"入库前总块数: {service.get_total_chunks()}")
    print("-" * 78)

    started = time.perf_counter()
    chunks_by_source: dict[str, int] = {}
    failures: list[tuple[str, str]] = []

    for index, pdf_path in enumerate(pdf_files, start=1):
        try:
            added = service.ingest_file(str(pdf_path))
        except Exception as exc:
            failures.append((pdf_path.name, f"{type(exc).__name__}: {exc}"))
            print(f"[{index:2d}/{len(pdf_files)}] {pdf_path.name:<42} 失败: {type(exc).__name__}")
            continue

        chunks_by_source[pdf_path.name] = added
        print(
            f"[{index:2d}/{len(pdf_files)}] {pdf_path.name:<42} "
            f"+{added:3d} 块 | 累计 {service.get_total_chunks()}"
        )

    elapsed = time.perf_counter() - started
    total = service.get_total_chunks()
    print("-" * 78)
    print(
        f"入库完成: 成功 {len(chunks_by_source)} 个文件，失败 {len(failures)} 个，"
        f"耗时 {elapsed:.1f}s"
    )
    print(f"最终总块数: {total}")
    if failures:
        print("失败文件:")
        for name, reason in failures:
            print(f"  - {name}: {reason}")

    # 2) RAG 检索
    print()
    print(f"RAG 查询: {QUERY}")
    results = rag.retrieve(QUERY, top_k=3)
    print(f"Top-{len(results)} 结果:")
    for rank, hit in enumerate(results, start=1):
        meta = hit["metadata"]
        print(
            f"  [{rank}] source={meta.get('source')} | page={meta.get('page')} "
            f"| distance={hit['distance']:.4f}"
        )
        print(f"      {preview(hit['text'])}")
    assert results, "检索结果为空"

    # 3) 删除某个文件，观察总块数变化
    target = "accessories.pdf"
    expected = chunks_by_source.get(target, 0)
    before_delete = service.get_total_chunks()
    service.delete_file(target)
    after_delete = service.get_total_chunks()
    print()
    print(
        f"删除 {target}: 删除前 {before_delete} 块，删除后 {after_delete} 块，"
        f"减少 {before_delete - after_delete} 块（该文件本次入库 {expected} 块）"
    )
    if expected:
        assert before_delete - after_delete == expected, "删除数量与入库数量不一致"

    print(
        f"其余 {len(chunks_by_source) - (1 if expected else 0)} 个文件的数据仍保留在 "
        "data/chroma_db，进程重启后依然存在。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
