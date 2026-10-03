"""独立权限配置与内置工具参数校验，不读取 provider 凭证。"""

from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Any

import yaml

from yincode.llm import ToolCall
from yincode.llm._json import object_from_json

from ._types import Category
from .rule import FRIENDLY, Rule, RuleSet, parse_rule


class SettingsError(ValueError):
    """仅报告配置结构错误，不回显 YAML 原文。"""


@dataclass
class PermissionsBlock:
    allow: list[str] = field(default_factory=list)
    deny: list[str] = field(default_factory=list)


@dataclass
class Settings:
    default_mode: str = ""
    permissions: PermissionsBlock = field(default_factory=PermissionsBlock)


def read_document(path: Path) -> dict[str, Any]:
    try:
        if not path.exists():
            return {}
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if document is None:
            return {}
        if not isinstance(document, dict):
            raise SettingsError("权限配置必须是映射")
        block = document.get("permissions", {})
        if not isinstance(block, dict):
            raise SettingsError("permissions 必须是映射")
        for key in ("allow", "deny"):
            entries = block.get(key, [])
            if not isinstance(entries, list) or not all(isinstance(item, str) for item in entries):
                raise SettingsError("allow/deny 必须是字符串列表")
            if not all(parse_rule(item)[1] for item in entries):
                raise SettingsError("权限规则语法错误")
        if not isinstance(document.get("default_mode", ""), str):
            raise SettingsError("default_mode 必须是字符串")
        return document
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise SettingsError("权限配置无法读取或格式错误") from error


def load_settings(path: str | Path) -> Settings:
    document = read_document(Path(path))
    block = document.get("permissions", {})
    return Settings(
        document.get("default_mode", ""),
        PermissionsBlock(block.get("allow", []), block.get("deny", [])),
    )


def to_rule_set(settings: Settings) -> RuleSet:
    result = RuleSet()
    for values, allowing, destination in (
        (settings.permissions.allow, True, result.allow),
        (settings.permissions.deny, False, result.deny),
    ):
        for value in values:
            rule, valid = parse_rule(value)
            if valid:
                destination.append(Rule(rule.tool, rule.pattern, allowing))
    return result


def friendly_name(internal: str) -> str:
    return FRIENDLY.get(internal, internal)


def categorize(internal: str, read_only: bool) -> Category:
    if read_only:
        return Category.READ
    if internal in ("write_file", "edit_file"):
        return Category.WRITE
    return Category.EXEC


def _string(
    data: dict[str, Any], key: str, *, default: str | None = None, empty: bool = False
) -> str:
    value = data.get(key, default)
    if not isinstance(value, str) or (not empty and not value.strip()) or "\0" in value:
        raise ValueError("字段类型或内容错误")
    return value


def valid_search_pattern(pattern: str) -> bool:
    return not (
        Path(pattern).is_absolute()
        or PureWindowsPath(pattern).drive
        or ".." in pattern.replace("\\", "/").split("/")
    )


def extract_target(call: ToolCall) -> tuple[str, bool, bool]:
    try:
        data = object_from_json(call.input)
        if call.name == "bash":
            return _string(data, "command"), False, True
        if call.name in ("read_file", "write_file", "edit_file"):
            target = _string(data, "path")
            if call.name == "write_file":
                _string(data, "content", empty=True)
            elif call.name == "edit_file":
                _string(data, "old_string")
                _string(data, "new_string", empty=True)
            return target, True, True
        if call.name in ("glob", "grep"):
            target = _string(data, "path", default=".")
            pattern = _string(data, "pattern")
            if call.name == "glob" and not valid_search_pattern(pattern):
                raise ValueError("搜索模式不能越界")
            if "glob" in data and not valid_search_pattern(_string(data, "glob")):
                raise ValueError("搜索模式不能越界")
            return target, True, True
        return "", False, True
    except (ValueError, TypeError, OSError):
        return "", False, False
