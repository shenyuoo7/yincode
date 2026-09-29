"""创建或覆盖 UTF-8 文件。"""

from threading import Event
from typing import Any

from . import Result
from .registry import _atomic_write, _FileTool, _string


class WriteFileTool(_FileTool):
    _name = "write_file"
    _description = "创建或覆盖 UTF-8 文件，自动创建父目录。"
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
