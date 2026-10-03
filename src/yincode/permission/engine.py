"""入口强制校验与四层权限决定；Ask 交由宿主人在回路处理。"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock

from yincode.llm import ToolCall

from ._types import Category, Decision, Mode, parse_mode
from .blacklist import hits_blacklist
from .rule import RuleSet
from .sandbox import eval_symlinks_or_ancestor, resolve_root, sandbox_ok
from .settings import (
    Settings,
    SettingsError,
    categorize,
    extract_target,
    friendly_name,
    load_settings,
    to_rule_set,
)


def mode_fallback(mode: Mode, category: Category) -> Decision:
    if category is Category.READ or mode is Mode.BYPASS:
        return Decision.ALLOW
    if mode is Mode.ACCEPT_EDITS and category is Category.WRITE:
        return Decision.ALLOW
    return Decision.ASK


@dataclass
class Engine:
    root: str
    resource_roots: tuple[str, ...] = ()
    user: RuleSet = field(default_factory=RuleSet)
    project: RuleSet = field(default_factory=RuleSet)
    local: RuleSet = field(default_factory=RuleSet)
    local_path: str = ""
    _start_mode: Mode = Mode.DEFAULT
    deny_all: bool = False
    warnings: list[str] = field(default_factory=list)
    _lock: RLock = field(default_factory=RLock, repr=False)

    def start_mode(self) -> Mode:
        return self._start_mode

    def relative_target(self, target: str) -> str:
        resolved = eval_symlinks_or_ancestor(str(Path(self.root) / target))
        return os.path.relpath(resolved, self.root).replace("\\", "/")

    def check(
        self, mode: Mode, call: ToolCall, read_only: bool, *, registered: bool, allowed: bool
    ) -> tuple[Decision, str]:
        if self.deny_all:
            return Decision.DENY, "权限引擎无法确定项目根，拒绝工具执行"
        if not registered:
            return Decision.DENY, f"未知工具: {call.name}"
        if not allowed or (mode is Mode.PLAN and not read_only):
            return Decision.DENY, f"工具 {call.name} 在当前模式不可用"
        target, is_file, valid = extract_target(call)
        if not valid:
            return Decision.DENY, "工具参数不合法：需要完整 JSON 对象及有效必填字段"
        if call.name == "bash" and hits_blacklist(target):
            return Decision.DENY, f"命中危险命令黑名单：{target}"
        if is_file:
            if not sandbox_ok(self, target, internal=call.name):
                return Decision.DENY, f"路径在项目目录之外或无法解析：{target}"
            try:
                target = self.relative_target(target)
            except (OSError, ValueError, RuntimeError):
                return Decision.DENY, "路径无法解析"
        friendly = friendly_name(call.name)
        with self._lock:
            for rules in (self.local, self.project, self.user):
                decision, matched = rules.match(friendly, target)
                if matched:
                    return decision, (
                        f"匹配 deny 规则：{friendly}({target})" if decision is Decision.DENY else ""
                    )
        decision = mode_fallback(mode, categorize(call.name, read_only))
        return decision, (
            f"{mode} 模式下 {categorize(call.name, read_only).name} 类操作需确认"
            if decision is Decision.ASK
            else ""
        )

    def persist_local_allow(self, call: ToolCall) -> None:
        from .persist import persist_local_allow

        persist_local_allow(self, call)


def new_engine(root: str) -> tuple[Engine, Exception | None]:
    try:
        resolved = resolve_root(root)
    except (OSError, ValueError, RuntimeError) as error:
        return Engine(root, deny_all=True), error
    engine = Engine(resolved, local_path=str(Path(resolved) / ".yincode/permissions.local.yaml"))
    paths = (
        Path.home() / ".yincode/permissions.yaml",
        Path(resolved) / ".yincode/permissions.yaml",
        Path(engine.local_path),
    )
    settings: list[Settings] = []
    for path in paths:
        try:
            settings.append(load_settings(path))
        except SettingsError:
            engine.warnings.append(f"权限配置格式错误或无法读取，已忽略：{path}")
            settings.append(Settings())
    engine.user, engine.project, engine.local = (to_rule_set(item) for item in settings)
    for item in reversed(settings):
        mode, valid = parse_mode(item.default_mode)
        if valid:
            engine._start_mode = mode
            break
    return engine, None
