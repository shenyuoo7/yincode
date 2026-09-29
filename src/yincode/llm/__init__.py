"""统一消息、流事件与模型适配器接口。"""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal, Protocol

from yincode.config import ProviderConfig


@dataclass(frozen=True, slots=True)
class Message:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class StreamEvent:
    text: str = ""
    done: bool = False
    err: Exception | None = None

    def __post_init__(self) -> None:
        if sum((bool(self.text), self.done, self.err is not None)) > 1:
            raise ValueError("流事件只能包含正文、完成或错误中的一种")


class Provider(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    def stream(self, msgs: list[Message]) -> AsyncIterator[StreamEvent]: ...

    async def aclose(self) -> None: ...


def new_provider(cfg: ProviderConfig) -> Provider:
    if cfg.protocol == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(cfg)
    if cfg.protocol == "openai":
        from .openai_provider import OpenAIProvider

        return OpenAIProvider(cfg)
    raise ValueError("不支持的模型协议")
