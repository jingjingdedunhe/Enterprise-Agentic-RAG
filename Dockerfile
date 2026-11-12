# syntax=docker/dockerfile:1
FROM python:3.12-slim

# 默认使用 CPU 版 PyTorch，便于在没有 NVIDIA 运行时的环境直接演示。
# GPU 构建示例（需要 NVIDIA Container Toolkit + gpus: all）：
#   docker build \
#     --build-arg TORCH_VERSION=2.14.1 \
#     --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 \
#     -t enterprise-agentic-rag:gpu .
ARG TORCH_VERSION=2.14.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    LANG=zh_CN.UTF-8 \
    LANGUAGE=zh_CN:zh \
    LC_ALL=zh_CN.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    HF_HOME=/root/.cache/huggingface \
    HF_ENDPOINT=https://hf-mirror.com \
    PYTHONPATH=/app

WORKDIR /app

# 系统依赖：编译工具、时区、中文 locale、PyTorch 运行时库
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc g++ curl ca-certificates tzdata locales libgomp1 \
    && sed -i 's/^# *zh_CN.UTF-8 UTF-8/zh_CN.UTF-8 UTF-8/' /etc/locale.gen \
    && locale-gen \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo "$TZ" > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# 单独安装 PyTorch：默认 CPU 版；GPU 构建时通过 TORCH_INDEX_URL 切换
RUN python -m pip install --no-cache-dir "torch==${TORCH_VERSION}" \
        --index-url "${TORCH_INDEX_URL}" \
        --extra-index-url "${PIP_INDEX_URL}"

# requirements.txt 中仍保留本地开发环境的 torch 版本；
# Docker 里已单独安装 PyTorch，因此过滤掉该行再安装其余依赖。
COPY requirements.txt .
RUN grep -vE '^torch([=<>!~ ]|$)' requirements.txt > /tmp/requirements-no-torch.txt \
    && python -m pip install --no-cache-dir -r /tmp/requirements-no-torch.txt

COPY api ./api
COPY app ./app
COPY backend ./backend
COPY config ./config
COPY README.md ./README.md

RUN mkdir -p /app/data/chroma_db /app/data/uploads /root/.cache/huggingface

EXPOSE 8000 8501

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
