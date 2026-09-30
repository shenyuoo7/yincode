"""统一消息、流事件与模型适配器接口。"""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from yincode.config import ProviderConfig

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_TOOL = "tool"


@dataclass(frozen=True, slots=True)
class ToolCall:
    """模型工具请求；input 保留原始 JSON 对象字符串。"""

    id: str
    name: str
    input: str


@dataclass(frozen=True, slots=True)
class ToolResult:
    """以调用 id 配对的历史结果。"""

    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """供协议适配器转换的工具定义。"""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Message:
    role: Literal["user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Usage:
    """单次模型请求的输入与输出 token 数。"""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_write: int = 0
    cache_read: int = 0


@dataclass(frozen=True, slots=True)
class System:
    """稳定指令与每次运行采集的环境分开承载。"""

    stable: str = ""
    environment: str = ""


@dataclass(frozen=True, slots=True)
class Request:
    """模型请求；reminder 只进入请求副本，不写会话历史。"""

    messages: list[Message] = field(default_factory=list)
    tools: list[ToolDefinition] = field(default_factory=list)
    system: System = field(default_factory=System)
    reminder: str = ""


@dataclass(frozen=True, slots=True)
class StreamEvent:
    text: str = ""
    done: bool = False
    err: Exception | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage | None = None

    def __post_init__(self) -> None:
        if (
            sum(
                (
                    bool(self.text),
                    self.done,
                    self.err is not None,
                    bool(self.tool_calls),
                    self.usage is not None,
                )
            )
            > 1
        ):
            raise ValueError("流事件只能包含正文、工具调用、用量、完成或错误中的一种")


class Provider(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    def stream(self, req: Request) -> AsyncIterator[StreamEvent]: ...

    async def aclose(self) -> None: ...


def new_provider(cfg: ProviderConfig) -> Provider:
    if cfg.protocol == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(cfg)
    if cfg.protocol == "openai":
        from .openai_provider import OpenAIProvider

        return OpenAIProvider(cfg)
    raise ValueError("不支持的模型协议")
