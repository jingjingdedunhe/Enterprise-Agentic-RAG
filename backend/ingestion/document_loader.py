"""文档加载模块。

将 PDF / Word 文件解析为统一的片段列表，供后续切分与向量化使用。
每个片段的结构固定为：

    {"text": "提取出的文本内容",
     "metadata": {"source": "文件名", "page": 页码(整数，Word 为 'N/A'),
                  "file_type": "pdf" 或 "word"}}
"""

from __future__ import annotations

from pathlib import Path

import docx
import fitz  # PyMuPDF
from docx.opc.exceptions import PackageNotFoundError

# 支持的扩展名 -> 返回结果中的 file_type
_SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".pdf": "pdf",
    ".docx": "word",
}


def _extract_pdf(path: Path) -> list[dict]:
    """按页提取 PDF 文本，跳过空白页。"""
    results: list[dict] = []

    try:
        with fitz.open(path) as doc:
            if doc.needs_pass:
                raise ValueError(f"PDF 文件已加密，无法读取: {path.name}")

            for page_index in range(doc.page_count):
                page = doc.load_page(page_index)
                text = page.get_text("text").strip()
                if not text:  # 空白页不加入结果
                    continue

                results.append(
                    {
                        "text": text,
                        "metadata": {
                            "source": path.name,
                            "page": page_index + 1,
                            "file_type": "pdf",
                        },
                    }
                )
    except RuntimeError as exc:
        # PyMuPDF 在文件损坏 / 非 PDF 内容时抛 RuntimeError 的子类
        raise ValueError(f"无法解析 PDF 文件（可能已损坏）: {path.name}") from exc

    return results


def _extract_word(path: Path) -> list[dict]:
    """提取 Word 文档的段落，按出现顺序拼接为一段文本。"""
    try:
        document = docx.Document(str(path))
    except PackageNotFoundError as exc:
        raise ValueError(f"无法解析 Word 文件（可能已损坏）: {path.name}") from exc

    paragraphs = [para.text.strip() for para in document.paragraphs]
    text = "\n".join(paragraph for paragraph in paragraphs if paragraph)

    if not text:  # 空文档不加入结果
        return []

    return [
        {
            "text": text,
            "metadata": {
                "source": path.name,
                "page": "N/A",  # Word 没有页的概念
                "file_type": "word",
            },
        }
    ]


def load_document(file_path: str) -> list[dict]:
    """解析文档，返回非空文本片段列表。

    Args:
        file_path: PDF 或 DOCX 文件路径。

    Returns:
        [{"text": str, "metadata": {"source": str, "page": int | "N/A",
                                    "file_type": "pdf" | "word"}}]

    Raises:
        FileNotFoundError: 文件不存在，或路径不是文件。
        ValueError: 扩展名不受支持，或文件已加密 / 已损坏。
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    if not path.is_file():
        raise FileNotFoundError(f"路径不是一个文件: {file_path}")

    extension = path.suffix.lower()
    file_type = _SUPPORTED_EXTENSIONS.get(extension)

    if file_type is None:
        if extension == ".doc":
            raise ValueError(
                f"不支持的格式: {extension}（python-docx 仅支持 .docx，"
                f"请先用 Word 另存为 .docx）"
            )
        raise ValueError(
            f"不支持的格式: {extension or '(无扩展名)'}，"
            f"目前仅支持: {', '.join(_SUPPORTED_EXTENSIONS)}"
        )

    if file_type == "pdf":
        return _extract_pdf(path)

    return _extract_word(path)
