"""使用官方 Anthropic SDK 提供正文流。"""

import asyncio
from collections.abc import AsyncIterator

from anthropic import AsyncAnthropic, Omit, omit
from anthropic.lib.streaming import AsyncMessageStream
from anthropic.types import MessageParam, ThinkingConfigParam

from yincode.config import ProviderConfig, redact
from yincode.llm import Message, StreamEvent
from yincode.prompt import SYSTEM_PROMPT


class AnthropicProvider:
    def __init__(self, cfg: ProviderConfig, *, client: AsyncAnthropic | None = None) -> None:
        self._cfg = cfg
        self._client = client or AsyncAnthropic(
            api_key=cfg.api_key, base_url=cfg.base_url, max_retries=0
        )
        self._streams: set[AsyncMessageStream[None]] = set()

    @property
    def name(self) -> str:
        return self._cfg.name

    @property
    def model(self) -> str:
        return self._cfg.model

    async def stream(self, msgs: list[Message]) -> AsyncIterator[StreamEvent]:
        messages: list[MessageParam] = [
            {"role": message.role, "content": message.content} for message in msgs
        ]
        thinking: ThinkingConfigParam | Omit = (
            {"type": "enabled", "budget_tokens": 2048} if self._cfg.thinking else omit
        )
        try:
            async with self._client.messages.stream(
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=messages,
                thinking=thinking,
            ) as stream:
                self._streams.add(stream)
                try:
                    async for event in stream:
                        if (
                            event.type == "content_block_delta"
                            and event.delta.type == "text_delta"
                            and event.delta.text
                        ):
                            yield StreamEvent(text=event.delta.text)
                finally:
                    self._streams.discard(stream)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            # 创建独立异常，避免向界面传递 SDK 的请求、响应和原始异常链。
            yield StreamEvent(err=RuntimeError(redact(str(error), [self._cfg.api_key])))
        else:
            yield StreamEvent(done=True)

    async def aclose(self) -> None:
        try:
            for stream in tuple(self._streams):
                await stream.close()
        finally:
            await self._client.close()
