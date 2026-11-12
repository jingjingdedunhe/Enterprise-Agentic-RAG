"""最小入库链路冒烟测试：PDF 解析 + 切分。

用法（项目根目录）：
    C:\\RAG\\P1\\Prag\\Scripts\\python.exe tests\\test_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from backend.ingestion.document_loader import load_document  # noqa: E402
from backend.ingestion.text_splitter import split_documents  # noqa: E402

# 1. 找到你的测试文档
sample_dir = BASE_DIR / "data" / "sample_docs"
# 找一个 PDF 文件
pdf_files = list(sample_dir.glob("*.pdf"))

if not pdf_files:
    print("❌ 错误：请先在 data/sample_docs 文件夹里放一份 PDF 文档！")
    exit()

test_file = pdf_files[0]
print(f"🔍 正在解析文件: {test_file.name}")

# 2. 解析文档
try:
    raw_docs = load_document(str(test_file))
    print(f"✅ 解析成功，提取了 {len(raw_docs)} 页的内容。")
except Exception as e:
    print(f"❌ 解析失败: {e}")
    exit()

# 3. 切分文档
chunks = split_documents(raw_docs)
print(f"✅ 切片成功，共生成了 {len(chunks)} 个文本块 (Chunks)。\n")

# 4. 打印前 3 个 Chunk 看看效果
print("-" * 40)
for i, chunk in enumerate(chunks[:3]):
    print(f"【Chunk {i+1}】元数据: {chunk['metadata']}")
    print(f"内容预览: {chunk['text'][:150]}...\n")
print("-" * 40)