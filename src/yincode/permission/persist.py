"""精确授权同目录原子保存；磁盘提交成功后才更新内存。"""

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from yincode.llm import ToolCall

from ._types import ApprovalError
from .rule import escape_pattern
from .sandbox import sandbox_ok
from .settings import (
    PermissionsBlock,
    Settings,
    extract_target,
    friendly_name,
    read_document,
    to_rule_set,
)

if TYPE_CHECKING:
    from .engine import Engine


def rule_for(engine: "Engine", call: ToolCall) -> str:
    target, is_file, valid = extract_target(call)
    if not valid or call.name not in (
        "bash",
        "read_file",
        "write_file",
        "edit_file",
        "glob",
        "grep",
    ):
        raise ApprovalError("无法为该调用生成精确授权")
    if is_file:
        if not sandbox_ok(engine, target, internal=call.name):
            raise ApprovalError("路径无法安全授权")
        target = engine.relative_target(target)
    return f"{friendly_name(call.name)}({escape_pattern(target)})"


def persist_local_allow(engine: "Engine", call: ToolCall) -> None:
    with engine._lock:
        rule = rule_for(engine, call)
        path = Path(engine.local_path)
        # 本地权限文件也必须在项目内，拒绝重定向到项目外的链接。
        if not sandbox_ok(engine, str(path), internal="write_file") or path.is_symlink():
            raise ApprovalError("本地权限文件不在安全范围内")
        document = read_document(path)
        block = document.setdefault("permissions", {})
        allowing = block.setdefault("allow", [])
        if rule not in allowing:
            allowing.append(rule)
        updated = to_rule_set(
            Settings(
                document.get("default_mode", ""), PermissionsBlock(allowing, block.get("deny", []))
            )
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
            ) as stream:
                temporary = Path(stream.name)
                yaml.safe_dump(document, stream, allow_unicode=True, sort_keys=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            engine.local = updated
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
