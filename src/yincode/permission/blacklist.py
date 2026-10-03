"""内置启发式危险命令拦截；非完备防御，不可通过配置放开。"""

import re

_BLACKLIST = tuple(
    re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for pattern in (
        r"\brm\s+(?:-\S+\s+)+[\"']?(?:/\*?|~/?|\$HOME/?)[\"']?(?=\s|[;&|]|$)",
        r"\bdd\s+[^;\n]*\bof\s*=\s*[\"']?/dev/",
        r":\s*\(\s*\)\s*\{[^}]*\|[^}]*&\s*\}",
        r"\bmkfs(?:\.[\w-]+)?\b",
        r">\s*[\"']?/dev/(?:sd|hd|nvme|disk)",
        r"\bchmod\s+-R\s+0?777\s+/\s*(?:$|[;&|])",
        r"\b(?:remove-item|rm|ri|del|erase|rd|rmdir)\s+[^;\n]*(?:[a-z]:[\\/]|\$HOME|\$env:USERPROFILE)[\"']?(?=\s|$|[;&|])",
        r"\b(?:format-volume|clear-disk|initialize-disk|format\.com)\b",
        r"\bformat\s+[a-z]:",
    )
)


def hits_blacklist(command: str) -> bool:
    return any(pattern.search(command) for pattern in _BLACKLIST)
