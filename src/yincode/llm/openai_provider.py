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
        if client is None:
            self._client = AsyncOpenAI(
                api_key=cfg.api_key,
                base_url=cfg.base_url or "https://api.openai.com/v1",
                organization="",
                project="",
                admin_api_key="",
                webhook_secret="",
                max_retries=0,
            )
            # 显式空值阻止环境身份回填，清理 SDK 无条件合入的环境请求头。
            self._client.organization = None
            self._client.project = None
            self._client.admin_api_key = None
            self._client._custom_headers = {}
            self._client._ambient_authorizations = frozenset()
        else:
            self._client = client
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
        completed = False
        try:
            stream = await self._client.chat.completions.create(
                model=self.model, messages=messages, stream=True
            )
            async with stream:
                self._streams.add(stream)
                try:
                    async for chunk in stream:
                        if chunk.choices:
                            choice = chunk.choices[0]
                            if choice.finish_reason:
                                completed = True
                            text = choice.delta.content
                            if text:
                                yield StreamEvent(text=text)
                finally:
                    self._streams.discard(stream)
            if not completed:
                raise RuntimeError("响应流提前结束，未收到完整回复")
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
