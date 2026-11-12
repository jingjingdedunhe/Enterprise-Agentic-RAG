"""测试脚本：验证 Embedding 模块的维度、归一化与语义相似度。

用法（在项目根目录 C:\\RAG\\P1 下执行）：
    C:\\RAG\\P1\\Prag\\Scripts\\python.exe tests\\test_embedding.py

首次运行会下载 BAAI/bge-small-zh-v1.5（约 100MB）。
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

import numpy as np  # noqa: E402
import torch  # noqa: E402

from backend.retrieval.embedding import (  # noqa: E402
    MODEL_NAME,
    embed_query,
    embed_texts,
    get_device,
    get_embedding_model,
)

EXPECTED_DIM = 512


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """按定义计算余弦相似度（向量已归一化时等价于点积，这里仍显式计算）。"""
    a = np.asarray(vec_a, dtype=np.float64)
    b = np.asarray(vec_b, dtype=np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> int:
    print(f"PyTorch: {torch.__version__} | CUDA 可用: {torch.cuda.is_available()}")
    print(f"推理设备: {get_device()}")
    print(f"模型名称: {MODEL_NAME}")

    started = time.perf_counter()
    get_embedding_model()
    elapsed = time.perf_counter() - started
    print(f"模型加载完成，耗时 {elapsed:.1f}s（本地无缓存时会先下载模型）")
    print()

    text_a = "今天天气真不错"
    text_b = "今天阳光明媚"
    text_c = "Velvera Watch的电池续航"

    vec_a, vec_b, vec_c = embed_texts([text_a, text_b, text_c])

    dim_a, dim_b, dim_c = len(vec_a), len(vec_b), len(vec_c)
    print(f"句子 A: {text_a}")
    print(f"句子 B: {text_b}")
    print(f"句子 C: {text_c}")
    print(f"向量维度: A={dim_a}, B={dim_b}, C={dim_c}（预期 {EXPECTED_DIM}）")

    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    print(f"向量 L2 范数: A={norm_a:.6f}, B={norm_b:.6f}（归一化后应为 1）")
    print()

    sim_related = cosine_similarity(vec_a, vec_b)
    sim_unrelated = cosine_similarity(vec_a, vec_c)

    print("【语义相近】")
    print(f'  cos("{text_a}", "{text_b}") = {sim_related:.4f}')
    print("【语义无关】")
    print(f'  cos("{text_a}", "{text_c}") = {sim_unrelated:.4f}')
    print(f"  相近句比无关句高 {sim_related - sim_unrelated:.4f}")
    print()

    query = "Velvera Watch 的电池能用多久？"
    query_vec = embed_query(query)
    print(f"embed_query 冒烟测试: {query!r} -> 维度 {len(query_vec)}")

    # 基础校验
    assert dim_a == dim_b == dim_c == EXPECTED_DIM, f"维度不是 {EXPECTED_DIM}: {dim_a}"
    assert abs(norm_a - 1.0) < 1e-3 and abs(norm_b - 1.0) < 1e-3, "向量未正确归一化"
    assert sim_related > sim_unrelated, "相近句相似度应高于无关句"
    assert len(query_vec) == EXPECTED_DIM, "embed_query 维度不正确"
    print()
    print("断言校验: 维度、归一化、相似度排序均通过 ✅")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
