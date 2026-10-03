"""权限域的共享类型，避免引擎与包门面循环导入。"""

from enum import IntEnum


class Mode(IntEnum):
    DEFAULT = 0
    ACCEPT_EDITS = 1
    PLAN = 2
    BYPASS = 3

    def __str__(self) -> str:
        return ("default", "acceptEdits", "plan", "bypassPermissions")[self.value]


def parse_mode(value: str) -> tuple[Mode, bool]:
    for mode in Mode:
        if value.casefold() == str(mode).casefold():
            return mode, True
    return Mode.DEFAULT, False


class Decision(IntEnum):
    ALLOW = 0
    DENY = 1
    ASK = 2


class Category(IntEnum):
    READ = 0
    WRITE = 1
    EXEC = 2


class Outcome(IntEnum):
    DENY_ONCE = 0
    ALLOW_ONCE = 1
    ALLOW_FOREVER = 2


class ApprovalError(ValueError):
    """规则不能安全保存时，由调用方报告未持久化。"""
