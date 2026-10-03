"""三层规则匹配；反斜杠转义通配字符和反斜杠自身。"""

import os
import re
from dataclasses import dataclass, field

from ._types import Decision

FRIENDLY = {
    "bash": "Bash",
    "read_file": "Read",
    "write_file": "Write",
    "edit_file": "Edit",
    "glob": "Glob",
    "grep": "Grep",
}


@dataclass(frozen=True)
class Rule:
    tool: str
    pattern: str
    allow: bool = False


def parse_rule(value: str) -> tuple[Rule, bool]:
    match = re.fullmatch(r"(Bash|Read|Write|Edit|Glob|Grep)(?:\((.*)\))?", value.strip(), re.DOTALL)
    if match is None:
        return Rule("", ""), False
    return Rule(match[1], match[2] or ""), True


def escape_pattern(target: str) -> str:
    return "".join("\\" + char if char in "\\*?[]" else char for char in target)


def match_pattern(pattern: str, target: str, *, kind: str) -> bool:
    if not pattern:
        return True
    fragments: list[str] = []
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == "\\" and index + 1 < len(pattern):
            index += 1
            fragments.append(re.escape(pattern[index]))
        elif char == "*":
            if index + 1 < len(pattern) and pattern[index + 1] == "*":
                index += 1
                if kind == "path" and index + 1 < len(pattern) and pattern[index + 1] == "/":
                    fragments.append("(?:.*/)?")
                    index += 1
                else:
                    fragments.append(".*")
            else:
                fragments.append(".*" if kind == "command" else "[^/]*")
        elif char == "?":
            fragments.append("." if kind == "command" else "[^/]")
        else:
            fragments.append(re.escape(char))
        index += 1
    flags = re.DOTALL | (re.IGNORECASE if kind == "path" and os.name == "nt" else 0)
    return re.fullmatch("".join(fragments), target, flags) is not None


@dataclass
class RuleSet:
    allow: list[Rule] = field(default_factory=list)
    deny: list[Rule] = field(default_factory=list)

    def match(self, friendly: str, target: str) -> tuple[Decision, bool]:
        for rules, decision in ((self.deny, Decision.DENY), (self.allow, Decision.ALLOW)):
            for rule in rules:
                if rule.tool == friendly and match_pattern(
                    rule.pattern, target, kind="command" if friendly == "Bash" else "path"
                ):
                    return decision, True
        return Decision.ASK, False
