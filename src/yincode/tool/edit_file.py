"""只替换具有唯一匹配的文本片段。"""

from threading import Event
from typing import Any

from . import Result
from .registry import _atomic_write, _FileTool, _string


class EditFileTool(_FileTool):
    _name = "edit_file"
    _description = "以 new_string 替换文件中唯一匹配的非空 old_string；匹配数不等于 1 则拒绝。"
    _parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old_string": {"type": "string", "minLength": 1},
            "new_string": {"type": "string"},
        },
        "required": ["path", "old_string", "new_string"],
    }

    def _execute(self, data: dict[str, Any], stopped: Event) -> Result:
        path = self._path(_string(data, "path"))
        old = _string(data, "old_string")
        new = _string(data, "new_string", empty=True)
        with path.open(encoding="utf-8", newline="") as stream:
            content = stream.read()
        count = 0
        start = 0
        while not stopped.is_set():
            match = content.find(old, start)
            if match < 0:
                break
            count += 1
            start = match + 1
        if count != 1:
            return Result(
                f"匹配到 {count} 处，old_string 必须唯一；请提供准确的上下文", is_error=True
            )
        if not stopped.is_set():
            _atomic_write(path, content.replace(old, new, 1), stopped)
        return Result(f"已修改 {data['path']}（替换 1 处）")
