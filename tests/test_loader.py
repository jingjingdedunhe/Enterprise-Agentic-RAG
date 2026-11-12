"""测试脚本：验证 backend.ingestion.document_loader.load_document 的输出。

用法（在项目根目录 C:\\RAG\\P1 下执行）：
    C:\\RAG\\P1\\Prag\\Scripts\\python.exe tests\\test_loader.py
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

SAMPLE_DIR = BASE_DIR / "data" / "sample_docs"
PREVIEW_LENGTH = 200


def pick_sample_pdf(directory: Path) -> Path:
    """取目录下任意一个 PDF（按文件名排序取第一个），保证结果可复现。"""
    pdf_files = sorted(path for path in directory.glob("*.pdf") if path.is_file())
    if not pdf_files:
        raise SystemExit(f"未找到测试用的 PDF 文件，请检查目录: {directory}")
    return pdf_files[0]


def preview_text(text: str, limit: int = PREVIEW_LENGTH) -> str:
    """把多行文本压成一行并截断，方便在终端查看。"""
    flattened = " ".join(text.split())
    if len(flattened) <= limit:
        return flattened
    return f"{flattened[:limit]}..."


def main() -> int:
    pdf_path = pick_sample_pdf(SAMPLE_DIR)
    print(f"测试文件: {pdf_path.name}")
    print(f"文件路径: {pdf_path}")

    try:
        documents = load_document(str(pdf_path))
    except Exception as exc:  # 测试脚本需要把失败原因完整暴露出来
        print(f"解析失败: {type(exc).__name__}: {exc}")
        return 1

    print(f"解析成功: 共得到 {len(documents)} 条非空白页文本")
    if not documents:
        print("文件没有可提取的文本（可能是扫描版或空白文档）。")
        return 0

    if len(documents) < 2:
        print(f"注意: 该文件只有 {len(documents)} 条非空内容，不足两页。")

    for index, document in enumerate(documents[:2], start=1):
        metadata = document["metadata"]
        print()
        print(f"----- 第 {index} 条 -----")
        print(f"  source    : {metadata['source']}")
        print(f"  page      : {metadata['page']} (类型: {type(metadata['page']).__name__})")
        print(f"  file_type : {metadata['file_type']}")
        print(f"  文本长度  : {len(document['text'])} 字符")
        print(f"  内容预览  : {preview_text(document['text'])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
