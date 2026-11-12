"""Embedding 模块。

封装 sentence-transformers 的 BGE 中文向量模型，提供：

- ``get_embedding_model()``：懒加载并缓存模型（避免重复占用显存/内存）；
- ``embed_texts(texts)``：批量把文本编码为归一化向量；
- ``embed_query(query)``：编码单个查询，并自动加上 BGE 推荐的检索指令。

所有向量均做了 L2 归一化，因此余弦相似度可以直接用点积计算。
"""

from __future__ import annotations

import os
import threading

import torch
from sentence_transformers import SentenceTransformer

# 模型名与配置文件中的 EMBEDDING_MODEL 默认值保持一致
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

# BGE 中文模型检索时的推荐指令：查询需要加，文档（passage）不加。
# 如需让 embed_query 与 embed_texts 完全同分布，可设置 BGE_QUERY_INSTRUCTION=""。
QUERY_INSTRUCTION = os.getenv(
    "BGE_QUERY_INSTRUCTION", "为这个句子生成表示以用于检索相关文章："
)

BATCH_SIZE = 32

_model: SentenceTransformer | None = None
_model_lock = threading.Lock()


def get_device() -> str:
    """返回推理设备：有可用 CUDA 时用 ``cuda``，否则用 ``cpu``。

    可用环境变量 ``EMBEDDING_DEVICE=cpu|cuda`` 强制指定；未设置时自动检测。
    """
    override = os.getenv("EMBEDDING_DEVICE", "").strip().lower()
    if override:
        if override not in {"cpu", "cuda"}:
            raise ValueError(
                f"EMBEDDING_DEVICE 只能是 'cpu' 或 'cuda'，当前为: {override!r}"
            )
        if override == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "EMBEDDING_DEVICE=cuda，但当前 PyTorch 未检测到可用 CUDA。"
                "请安装与本机 CUDA 匹配的 PyTorch 版本，或改用 EMBEDDING_DEVICE=cpu。"
            )
        return override

    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_model_source() -> str:
    """优先返回本地缓存中的模型快照路径。

    HuggingFace 在国内经常不可达，若每次都让 SentenceTransformer 联网解析
    ``main`` 分支，会先经历多轮 HEAD 超时重试。缓存在手时直接加载本地路径，
    既快又完全离线；缓存不存在时才回退到模型名，由底层下载。
    """
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(MODEL_NAME, local_files_only=True)
    except Exception:
        return MODEL_NAME


def get_embedding_model() -> SentenceTransformer:
    """加载模型并缓存（进程内单例）。首次调用可能需要下载模型文件。"""
    global _model

    if _model is not None:
        return _model

    # 双检锁：避免并发请求同时加载模型、重复占用显存
    with _model_lock:
        if _model is not None:
            return _model

        device = get_device()
        model_source = _resolve_model_source()
        try:
            _model = SentenceTransformer(model_source, device=device)
        except Exception as exc:
            raise RuntimeError(
                f"加载 Embedding 模型失败：{MODEL_NAME}（device={device}）。\n"
                "如果是首次运行，请确认网络可以访问 HuggingFace；国内网络可先设置 "
                "HF_ENDPOINT=https://hf-mirror.com 再重试。\n"
                f"原始错误：{type(exc).__name__}: {exc}"
            ) from exc

        return _model


def _is_oom_error(exc: BaseException) -> bool:
    """识别 CUDA OOM、CPU 内存不足，以及部分后端只给出文本的情况。"""
    return isinstance(
        exc, (torch.cuda.OutOfMemoryError, MemoryError)
    ) or "out of memory" in str(exc).lower()


def _raise_oom(exc: BaseException) -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    raise RuntimeError(
        "Embedding 推理时显存/内存不足（OOM）。可尝试：\n"
        "  1) 关闭其他占用显存的程序；\n"
        f"  2) 减小批量大小（当前 BATCH_SIZE={BATCH_SIZE}）；\n"
        "  3) 设置环境变量 EMBEDDING_DEVICE=cpu 改用 CPU 推理；\n"
        "  4) 缩短单条输入文本。"
    ) from exc


def embed_texts(texts: list[str]) -> list[list[float]]:
    """把一批文本编码为向量列表（L2 归一化，维度由模型决定）。

    Args:
        texts: 文本列表。

    Returns:
        与输入一一对应的向量列表；空输入返回 ``[]``。
    """
    if not texts:
        return []

    model = get_embedding_model()

    try:
        embeddings = model.encode(
            list(texts),
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    except Exception as exc:
        if _is_oom_error(exc):
            _raise_oom(exc)
        raise

    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """把单个查询编码为向量。

    会加上 BGE 推荐的查询指令（``QUERY_INSTRUCTION``），文档侧不应加，
    因此查询与文档的相似度比两边都加指令更准确。
    """
    if not isinstance(query, str):
        raise TypeError("query 必须是字符串")
    if not query.strip():
        raise ValueError("query 不能为空")

    return embed_texts([QUERY_INSTRUCTION + query])[0]
