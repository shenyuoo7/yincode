"""读取有界、带行号的 UTF-8 文本。"""

from threading import Event
from typing import Any

from . import Result
from .registry import READ_BYTES, READ_LINES, _FileTool, _string, _truncate, _utf8_prefix


class ReadFileTool(_FileTool):
    _name = "read_file"
    _description = "读取 UTF-8 文件并返回行号；最多 2000 行和 256 KiB，超出标注截断。"
    _parameters = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "要读取的文件路径"}},
        "required": ["path"],
    }

    def _execute(self, data: dict[str, Any], stopped: Event) -> Result:
        path = self._path(_string(data, "path"))
        lines: list[str] = []
        size = 0
        truncated = False
        with path.open("rb") as stream:
            for number in range(1, READ_LINES + 2):
                if stopped.is_set():
                    break
                raw = stream.readline(READ_BYTES + 1)
                if not raw:
                    break
                partial = len(raw) > READ_BYTES
                text = _utf8_prefix(raw[:READ_BYTES], partial=partial).rstrip("\r\n")
                line = f"{number:6d}\t{text}"
                lines.append(line)
                size += len(line.encode("utf-8")) + (1 if number > 1 else 0)
                if partial or size > READ_BYTES or number > READ_LINES:
                    truncated = True
                    break
        return Result(_truncate("\n".join(lines), READ_LINES, READ_BYTES, force=truncated))
