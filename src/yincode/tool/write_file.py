"""创建或覆盖 UTF-8 文件。"""

from threading import Event
from typing import Any

from . import Result
from .registry import _atomic_write, _FileTool, _string


class WriteFileTool(_FileTool):
    _name = "write_file"
    read_only = False
    _description = (
        "创建或明确要求的完整覆盖 UTF-8 文件，自动创建父目录。"
        "覆盖现有文件前先用 read_file 了解原文并保留需要保留的内容；"
        "局部修改优先用 edit_file，不用 bash 拼凑文件写入命令。"
    )
    _parameters = {
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"],
    }

    def _execute(self, data: dict[str, Any], stopped: Event) -> Result:
        value = _string(data, "path")
        content = _string(data, "content", empty=True)
        path = self._path(value)
        _atomic_write(path, content, stopped)
        size = sum(
            len(content[offset : offset + 65_536].encode("utf-8"))
            for offset in range(0, len(content), 65_536)
        )
        return Result(f"已写入 {value}（{size} 字节）")
