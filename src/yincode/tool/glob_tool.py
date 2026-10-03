"""有界保留排序最前的匹配文件。"""

import bisect
import fnmatch
import os
from threading import Event
from typing import Any

from yincode.permission.settings import valid_search_pattern

from . import Result
from .registry import OUTPUT_BYTES, _files, _FileTool, _string, _truncate


def _matches(path: str, pattern: str) -> bool:
    """按路径段匹配，** 可覆盖零层或多层目录。"""
    pieces = os.path.normcase(path).replace("\\", "/").split("/")
    normalized = os.path.normcase(pattern).replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    patterns = normalized.split("/")

    def match(start: int, index: int) -> bool:
        if index == len(patterns):
            return start == len(pieces)
        if patterns[index] == "**":
            return any(match(next_start, index + 1) for next_start in range(start, len(pieces) + 1))
        return (
            start < len(pieces)
            and fnmatch.fnmatchcase(pieces[start], patterns[index])
            and match(start + 1, index + 1)
        )

    return match(0, 0)


class GlobTool(_FileTool):
    _name = "glob"
    read_only = True
    _description = "按 glob 模式查找文件；支持 ** 跨目录，排序后最多返回 100 条。"
    _parameters = {
        "type": "object",
        "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
        "required": ["pattern"],
    }

    def _execute(self, data: dict[str, Any], stopped: Event) -> Result:
        root = self._path(_string(data, "path", default="."))
        pattern = _string(data, "pattern")
        if not valid_search_pattern(pattern):
            raise ValueError("pattern 必须是相对模式且不能含 ..；请用 path 指定搜索目录")
        selected: list[str] = []
        count = 0
        for path in _files(root, stopped):
            relative = path.relative_to(root).as_posix()
            if _matches(relative, pattern):
                count += 1
                bisect.insort(selected, relative)
                if len(selected) > 100:
                    selected.pop()
        return Result(
            _truncate("\n".join(selected) or "无匹配", 101, OUTPUT_BYTES, force=count > 100)
        )
