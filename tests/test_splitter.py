"""测试脚本：验证 document_loader + text_splitter 的端到端效果。

用法（在项目根目录 C:\\RAG\\P1 下执行）：
    C:\\RAG\\P1\\Prag\\Scripts\\python.exe tests\\test_splitter.py
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

SAMPLE_DIR = BASE_DIR / "data" / "sample_docs"
PREVIEW_LENGTH = 50


def pick_sample_pdf(directory: Path) -> Path:
    """取目录下任意一个 PDF（按文件名排序取第一个），保证结果可复现。"""
    pdf_files = sorted(path for path in directory.glob("*.pdf") if path.is_file())
    if not pdf_files:
        raise SystemExit(f"未找到测试用的 PDF 文件，请检查目录: {directory}")
    return pdf_files[0]


def preview(text: str, limit: int = PREVIEW_LENGTH) -> str:
    """取前 limit 个字符，把换行替换为空格以便单行显示。"""
    snippet = text[:limit].replace("\r", " ").replace("\n", " ")
    return f"{snippet}..." if len(text) > limit else snippet


def main() -> int:
    pdf_path = pick_sample_pdf(SAMPLE_DIR)
    print(f"测试文件: {pdf_path.name}")

    try:
        documents = load_document(str(pdf_path))
        chunks = split_documents(documents)
    except Exception as exc:  # 测试脚本需要把失败原因完整暴露出来
        print(f"处理失败: {type(exc).__name__}: {exc}")
        return 1

    print(f"解析结果: {len(documents)} 条非空白页文本")
    print(f"切分结果: 共生成 {len(chunks)} 个 chunk（chunk_size=500, overlap=80）")

    if not chunks:
        print("没有生成任何 chunk。")
        return 0

    # 基础校验：必需字段存在、chunk_id 唯一且编号正确
    required = {"source", "page", "file_type", "chunk_index", "chunk_id"}
    assert all(required <= chunk["metadata"].keys() for chunk in chunks), "元数据字段缺失"
    chunk_ids = [chunk["metadata"]["chunk_id"] for chunk in chunks]
    assert len(chunk_ids) == len(set(chunk_ids)), "chunk_id 存在重复"
    for chunk in chunks:
        meta = chunk["metadata"]
        expected = f"{meta['source']}_p{meta['page']}_c{meta['chunk_index']}"
        assert meta["chunk_id"] == expected, f"chunk_id 格式错误: {meta['chunk_id']}"
    print("元数据校验: 字段完整、chunk_id 唯一且格式正确 ✅")
    print()

    for order, chunk in enumerate(chunks[:3], start=1):
        meta = chunk["metadata"]
        print(f"----- 第 {order} 个 chunk -----")
        print(f"  chunk_id    : {meta['chunk_id']}")
        print(f"  chunk_index : {meta['chunk_index']}")
        print(f"  source      : {meta['source']}")
        print(f"  page        : {meta['page']}")
        print(f"  文本长度    : {len(chunk['text'])} 字符")
        print(f"  内容预览    : {preview(chunk['text'])}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
