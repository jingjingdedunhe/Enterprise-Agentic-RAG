"""Reranker 重排序。

使用 sentence-transformers 的 CrossEncoder 加载 BGE reranker，
对混合检索召回的候选做 query-passage 相关性重打分。

默认模型：``BAAI/bge-reranker-base``（可用环境变量 ``RERANKER_MODEL`` 覆盖）。
国内下载可设置 ``HF_ENDPOINT=https://hf-mirror.com``。
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder

MODEL_NAME = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
MAX_LENGTH = 512
BATCH_SIZE = 32

# 只下载推理必需的文件，避免把 onnx / pytorch_model.bin 一起拉下来
_ALLOW_PATTERNS = [
    "config.json",
    "model.safetensors",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
]

_reranker: CrossEncoder | None = None
_lock = threading.Lock()


def _get_device() -> str:
    override = os.getenv("RERANKER_DEVICE", "").strip().lower()
    if override:
        if override not in {"cpu", "cuda"}:
            raise ValueError(
                f"RERANKER_DEVICE 只能是 'cpu' 或 'cuda'，当前为: {override!r}"
            )
        if override == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "RERANKER_DEVICE=cuda，但当前 PyTorch 未检测到可用 CUDA。"
                "请安装 CUDA 版 PyTorch，或改用 RERANKER_DEVICE=cpu。"
            )
        return override
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_model_source() -> str:
    """优先使用本地缓存快照，避免 HuggingFace 不可达时反复超时。"""
    if Path(MODEL_NAME).is_dir():
        return MODEL_NAME

    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(
            MODEL_NAME,
            allow_patterns=_ALLOW_PATTERNS,
            local_files_only=True,
        )
    except Exception:
        # 本地没有完整缓存时才联网下载（仍然只取推理必需文件）
        return snapshot_download(MODEL_NAME, allow_patterns=_ALLOW_PATTERNS)


def get_reranker() -> CrossEncoder:
    """加载并缓存 CrossEncoder（进程内单例）。"""
    global _reranker

    if _reranker is not None:
        return _reranker

    with _lock:
        if _reranker is not None:
            return _reranker

        device = _get_device()
        try:
            model_source = _resolve_model_source()
            _reranker = CrossEncoder(
                model_source,
                device=device,
                max_length=MAX_LENGTH,
            )
        except Exception as exc:
            raise RuntimeError(
                f"加载 Reranker 模型失败：{MODEL_NAME}（device={device}）。\n"
                "首次运行需要下载模型；国内网络可先设置 "
                "HF_ENDPOINT=https://hf-mirror.com 再重试。\n"
                f"原始错误：{type(exc).__name__}: {exc}"
            ) from exc

        return _reranker


def _is_oom_error(exc: BaseException) -> bool:
    return isinstance(
        exc, (torch.cuda.OutOfMemoryError, MemoryError)
    ) or "out of memory" in str(exc).lower()


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    score_threshold: float = 0.3,
) -> list[dict]:
    """对候选 chunk 重新打分并返回 Top-K。

    Args:
        query: 用户问题。
        candidates: 混合检索召回的有序候选（通常 10–20 条）。
        top_k: 最终返回条数上限。
        score_threshold: 最低相关性分数；低于该值的候选直接丢弃，
            因此实际返回条数可能少于 ``top_k``。设为 0 可关闭截断。

    Returns:
        新的 Top-K 列表，保留原候选字段，并追加：
        - ``rerank_score``：CrossEncoder 相关性分数（越大越相关）；
        - ``rrf_rank``：该候选在传入列表中的原始名次（即 RRF 名次）。
    """
    if not isinstance(query, str) or not query.strip() or not candidates or top_k <= 0:
        return []

    model = get_reranker()
    pairs = [(query, (candidate.get("text") or "").strip()) for candidate in candidates]

    try:
        scores = model.predict(
            pairs,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
    except Exception as exc:
        if _is_oom_error(exc):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise RuntimeError(
                "Reranker 推理时显存/内存不足（OOM）。可尝试减小 batch_size、"
                "缩短候选文本，或设置 RERANKER_DEVICE=cpu。"
            ) from exc
        raise

    scored: list[dict] = []
    for index, (candidate, score) in enumerate(zip(candidates, scores)):
        item = dict(candidate)
        item["rerank_score"] = float(score)
        item["rrf_rank"] = index + 1
        scored.append(item)

    scored.sort(key=lambda item: item["rerank_score"], reverse=True)

    # 低于阈值直接丢弃；允许返回条数少于 top_k
    filtered = [
        item for item in scored if item["rerank_score"] >= score_threshold
    ]
    return filtered[:top_k]
