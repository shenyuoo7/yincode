"""统一工具协议、本地结果与注册中心。"""

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class Result:
    """执行结果；普通失败返回值，取消向上传播。"""

    content: str
    is_error: bool = False


@runtime_checkable
class Tool(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def parameters(self) -> dict[str, Any]: ...

    async def execute(self, args: str) -> Result: ...


from .registry import DEFAULT_TIMEOUT, Registry, new_default_registry  # noqa: E402

__all__ = ["DEFAULT_TIMEOUT", "Registry", "Result", "Tool", "new_default_registry"]
