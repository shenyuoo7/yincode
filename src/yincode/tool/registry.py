"""注册、超时、参数校验与有界文件任务。"""

import asyncio
import codecs
import os
import stat
import tempfile
from collections.abc import Callable, Iterator
from copy import deepcopy
from pathlib import Path
from threading import Event
from typing import Any

from yincode.llm import ToolDefinition
from yincode.llm._json import object_from_json

from . import Result, Tool

DEFAULT_TIMEOUT = 30.0
OUTPUT_BYTES = 30_000
READ_BYTES = 256 * 1024
READ_LINES = 2000


class Registry:
    """保留注册顺序；超时等待工具完成取消清理后返回。"""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"工具重复注册: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def definitions(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(tool.name, tool.description, deepcopy(tool.parameters))
            for tool in self._tools.values()
        ]

    def read_only_definitions(self) -> list[ToolDefinition]:
        """按注册顺序导出只读工具的定义。"""
        return [
            ToolDefinition(tool.name, tool.description, deepcopy(tool.parameters))
            for tool in self._tools.values()
            if tool.read_only
        ]

    def is_read_only(self, name: str) -> bool:
        """未知工具按有副作用处理。"""
        tool = self.get(name)
        return tool is not None and tool.read_only

    async def execute(
        self,
        name: str,
        args: str,
        timeout: float = DEFAULT_TIMEOUT,  # noqa: ASYNC109
        *,
        allowed_names: set[str] | None = None,
    ) -> Result:
        tool = self.get(name)
        if tool is None:
            return Result(f"未知工具: {name}", is_error=True)
        if allowed_names is not None and name not in allowed_names:
            return Result(f"工具 {name} 在当前模式不可用", is_error=True)
        try:
            return await asyncio.wait_for(tool.execute(args), timeout)
        except TimeoutError:
            return Result(f"工具 {name} 执行超时（{timeout}s）", is_error=True)
        except Exception as exc:
            return Result(f"工具 {name} 执行失败: {exc}", is_error=True)


def new_default_registry(cwd: Path | str | None = None) -> Registry:
    """构造六个工具，并为每个工具捕获同一绝对目录。"""
    from .bash import BashTool
    from .edit_file import EditFileTool
    from .glob_tool import GlobTool
    from .grep_tool import GrepTool
    from .read_file import ReadFileTool
    from .write_file import WriteFileTool

    directory = Path(cwd if cwd is not None else Path.cwd()).resolve()
    registry = Registry()
    for tool_type in [ReadFileTool, WriteFileTool, EditFileTool, BashTool, GlobTool, GrepTool]:
        registry.register(tool_type(cwd=directory))
    return registry


def _truncate(text: str, max_lines: int, max_bytes: int, *, force: bool = False) -> str:
    """将标记也计入行数和 UTF-8 字节上限，不截断多字节字符。"""
    lines = text.splitlines()
    raw = text.encode("utf-8")
    if not force and len(lines) <= max_lines and len(raw) <= max_bytes:
        return text
    marker = "[truncated]"
    if max_lines < 2 or max_bytes <= len(marker):
        return marker[:max_bytes]
    prefix = "\n".join(lines[: max_lines - 1])
    available = max_bytes - len(marker) - 1
    prefix = prefix.encode("utf-8")[:available].decode("utf-8", errors="ignore").rstrip("\n")
    return prefix + "\n" + marker if prefix else marker


def _parse(args: str) -> dict[str, Any]:
    return object_from_json(args)


def _string(
    data: dict[str, Any], key: str, *, default: str | None = None, empty: bool = False
) -> str:
    value = data.get(key, default)
    if not isinstance(value, str) or (not empty and not value):
        raise ValueError(f"参数 {key} 必须是{'非空' if not empty else ''}字符串")
    return value


async def _run_io[T](action: Callable[[Event], T]) -> T:
    """线程处理磁盘 IO；取消发信号，并等待任务停下，避免返回后继续写。"""
    stopped = Event()
    task = asyncio.create_task(asyncio.to_thread(action, stopped))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        stopped.set()
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not task.cancelled():
            task.exception()
        raise


def _atomic_write(path: Path, content: str, stopped: Event) -> None:
    """临时文件关闭后替换，取消前检查；失败清理拥有的临时文件。"""
    if stopped.is_set():
        return
    try:
        existing = path.stat()
    except FileNotFoundError:
        existing = None
    if (
        os.name == "nt"
        and existing is not None
        and existing.st_file_attributes & stat.FILE_ATTRIBUTE_READONLY
    ):
        raise PermissionError(f"目标文件为只读，无法写入: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(existing.st_mode) if existing is not None else None
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            for offset in range(0, len(content), 65_536):
                if stopped.is_set():
                    return
                stream.write(content[offset : offset + 65_536].encode("utf-8"))
        if not stopped.is_set():
            if mode is not None:
                temporary.chmod(mode)
            temporary.replace(path)
    finally:
        if temporary is not None:
            try:
                temporary.chmod(stat.S_IRUSR | stat.S_IWUSR)
                temporary.unlink(missing_ok=True)
            except FileNotFoundError:
                pass


class _FileTool:
    """共享元数据属性与文件工具执行边界。"""

    _name: str
    _description: str
    _parameters: dict[str, Any]
    read_only: bool

    def __init__(self, cwd: Path | str | None = None) -> None:
        self.cwd = Path(cwd if cwd is not None else Path.cwd()).resolve()

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return deepcopy(self._parameters)

    def _path(self, value: str) -> Path:
        return (self.cwd / value).resolve()

    async def execute(self, args: str) -> Result:
        try:
            return await _run_io(lambda stopped: self._execute(_parse(args), stopped))
        except Exception as exc:
            return Result(f"{self.name} 失败: {exc}", is_error=True)

    def _execute(self, data: dict[str, Any], stopped: Event) -> Result:
        raise NotImplementedError


def _files(root: Path, stopped: Event) -> Iterator[Path]:
    """流式深度优先遍历，不缓存整目录，不跟随目录链接。"""
    with os.scandir(root) as entries:
        for entry in entries:
            if stopped.is_set():
                return
            try:
                if entry.is_dir(follow_symlinks=False):
                    yield from _files(Path(entry.path), stopped)
                elif entry.is_file():
                    yield Path(entry.path)
            except OSError:
                continue


def _utf8_prefix(raw: bytes, *, partial: bool, errors: str = "strict") -> str:
    decoder = codecs.getincrementaldecoder("utf-8")(errors=errors)
    return decoder.decode(raw, final=not partial)
