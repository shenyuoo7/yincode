"""读取项目配置，错误信息不包含配置原文或密钥。"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

import yaml


class ConfigError(Exception):
    """可直接向用户展示的配置错误。"""


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    name: str
    protocol: Literal["anthropic", "openai"]
    api_key: str = field(repr=False)
    model: str
    base_url: str | None = None
    thinking: bool = False


@dataclass(frozen=True, slots=True)
class Config:
    providers: list[ProviderConfig]


def redact(text: str, secrets: Iterable[str]) -> str:
    for secret in sorted({key for key in secrets if key}, key=len, reverse=True):
        text = text.replace(secret, "[REDACTED]")
    return text


def load(path: str | Path) -> Config:
    source = Path(path)
    try:
        content = source.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(
            f"配置文件不存在: {source}，请参考 .yincode/config.yaml.example"
        ) from None
    except (OSError, UnicodeError):
        raise ConfigError(f"无法读取 UTF-8 配置文件: {source}") from None
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as error:
        mark = getattr(error, "problem_mark", None)
        location = f"（第 {mark.line + 1} 行）" if mark is not None else ""
        raise ConfigError(f"YAML 配置格式错误{location}") from None
    if not isinstance(data, dict):
        raise ConfigError("配置根节点必须是映射")
    entries = data.get("providers")
    if not isinstance(entries, list) or not entries:
        raise ConfigError("providers 必须是非空列表")
    providers = []
    for index, entry in enumerate(entries):
        prefix = f"providers[{index}]"
        if not isinstance(entry, dict):
            raise ConfigError(f"{prefix} 必须是映射")
        values = {}
        for name in ("name", "protocol", "api_key", "model"):
            value = entry.get(name)
            if not isinstance(value, str) or not value.strip():
                raise ConfigError(f"{prefix}.{name} 必须是非空字符串")
            values[name] = value.strip()
        if values["protocol"] not in ("anthropic", "openai"):
            raise ConfigError(f"{prefix}.protocol 仅支持 anthropic 或 openai")
        thinking = entry.get("thinking", False)
        if not isinstance(thinking, bool):
            raise ConfigError(f"{prefix}.thinking 必须是布尔值")
        base_url = entry.get("base_url")
        if base_url is not None and (not isinstance(base_url, str) or not base_url.strip()):
            raise ConfigError(f"{prefix}.base_url 必须是非空字符串或 null")
        providers.append(
            ProviderConfig(
                name=values["name"],
                protocol=cast(Literal["anthropic", "openai"], values["protocol"]),
                api_key=values["api_key"],
                model=values["model"],
                base_url=base_url.strip() if base_url else None,
                thinking=thinking,
            )
        )
    return Config(providers)
