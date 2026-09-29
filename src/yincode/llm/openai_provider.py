"""使用官方 OpenAI SDK 提供正文流。"""

import asyncio
from collections.abc import AsyncIterator

from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam

from yincode.config import ProviderConfig, redact
from yincode.llm import Message, StreamEvent
from yincode.prompt import SYSTEM_PROMPT


class OpenAIProvider:
    def __init__(self, cfg: ProviderConfig, *, client: AsyncOpenAI | None = None) -> None:
        self._cfg = cfg
        self._client = client or AsyncOpenAI(
            api_key=cfg.api_key, base_url=cfg.base_url, max_retries=0
        )
        self._streams: set[AsyncStream[ChatCompletionChunk]] = set()

    @property
    def name(self) -> str:
        return self._cfg.name

    @property
    def model(self) -> str:
        return self._cfg.model

    async def stream(self, msgs: list[Message]) -> AsyncIterator[StreamEvent]:
        messages: list[ChatCompletionMessageParam] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for message in msgs:
            if message.role == "user":
                messages.append({"role": "user", "content": message.content})
            else:
                messages.append({"role": "assistant", "content": message.content})
        try:
            stream = await self._client.chat.completions.create(
                model=self.model, messages=messages, stream=True
            )
            async with stream:
                self._streams.add(stream)
                try:
                    async for chunk in stream:
                        if chunk.choices:
                            text = chunk.choices[0].delta.content
                            if text:
                                yield StreamEvent(text=text)
                finally:
                    self._streams.discard(stream)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            # 错误事件只携带脱敏后的文本，不保留包含凭证的 SDK 异常对象。
            yield StreamEvent(err=RuntimeError(redact(str(error), [self._cfg.api_key])))
        else:
            yield StreamEvent(done=True)

    async def aclose(self) -> None:
        try:
            for stream in tuple(self._streams):
                await stream.close()
        finally:
            await self._client.close()
