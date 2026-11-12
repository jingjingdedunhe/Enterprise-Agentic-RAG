"""LLM 客户端封装。

使用 openai 官方 SDK，通过 OpenAI 兼容接口接入阿里云百炼（通义千问）。
"""

from __future__ import annotations

from typing import Any

import openai
from openai import OpenAI

from config.settings import settings

DEFAULT_TIMEOUT = 60.0


class LLMClient:
    """对 OpenAI SDK 的轻量封装，默认读取 .env 中的 LLM_* 配置。"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key or settings.LLM_API_KEY
        self.base_url = base_url or settings.LLM_BASE_URL
        self.model = model or settings.LLM_MODEL
        self.timeout = timeout

        if not self.api_key:
            raise ValueError(
                "缺少 LLM_API_KEY：请在项目根目录的 .env 中配置 "
                "LLM_API_KEY / LLM_BASE_URL / LLM_MODEL。"
            )

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> str:
        """发送对话消息，返回模型回复文本。

        Args:
            messages: OpenAI 格式的消息列表，如
                ``[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]``。
            temperature: 采样温度，RAG 场景默认取较低值以保证稳定。
            max_tokens: 最大生成 token 数。

        Raises:
            RuntimeError: 鉴权、限流、网络、超时或服务端错误，均转换为可读的中文提示。
        """
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except openai.AuthenticationError as exc:
            raise RuntimeError(
                "LLM 鉴权失败：请检查 .env 中的 LLM_API_KEY 是否有效。"
            ) from exc
        except openai.RateLimitError as exc:
            raise RuntimeError(
                "LLM 请求被限流或账户额度不足，请稍后重试并检查百炼控制台。"
            ) from exc
        except openai.APITimeoutError as exc:
            raise RuntimeError(
                f"LLM 请求超时（>{self.timeout:.0f}s），请检查网络或稍后重试。"
            ) from exc
        except openai.APIConnectionError as exc:
            raise RuntimeError(
                f"无法连接 LLM 服务：{self.base_url}，请检查网络和 LLM_BASE_URL。"
            ) from exc
        except openai.APIStatusError as exc:
            message = getattr(exc, "message", str(exc))
            raise RuntimeError(
                f"LLM 服务返回错误（HTTP {exc.status_code}）：{message}"
            ) from exc
        except openai.OpenAIError as exc:
            raise RuntimeError(f"LLM 调用失败：{type(exc).__name__}: {exc}") from exc

        if not response.choices:
            raise RuntimeError("LLM 返回结果为空（choices 为空）。")

        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("LLM 返回内容为空。")

        return content.strip()
