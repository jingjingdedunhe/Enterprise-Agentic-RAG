"""项目配置。

统一从项目根目录的 .env 读取环境变量，供各模块通过 ``settings`` 单例访问。
"""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# 显式指定项目根目录的 .env，避免从其他工作目录启动时读不到
load_dotenv(dotenv_path=BASE_DIR / ".env")


class Settings:
    # 基础路径
    BASE_DIR = BASE_DIR

    # 向量模型名称
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

    # Chroma 数据库存储路径（相对路径按项目根目录解析）
    _chroma_path = Path(os.getenv("CHROMA_PATH", "data/chroma_db"))
    CHROMA_PATH = str(
        _chroma_path
        if _chroma_path.is_absolute()
        else (BASE_DIR / _chroma_path).resolve()
    )

    # 文档切片参数
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 80

    # LLM（阿里云百炼 / 通义千问，OpenAI 兼容接口）
    LLM_API_KEY = os.getenv("LLM_API_KEY", "")
    LLM_BASE_URL = os.getenv(
        "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    LLM_MODEL = os.getenv("LLM_MODEL", "qwen-turbo")


settings = Settings()
