"""先解析链接再比较边界；文件围栏不约束任意 shell 进程。"""

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import Engine


def resolve_root(root: str) -> str:
    path = Path(root).expanduser().resolve(strict=True)
    if not path.is_dir():
        raise NotADirectoryError("项目根不是目录")
    return str(path)


def eval_symlinks_or_ancestor(abs_path: str) -> str:
    path = Path(abs_path)
    remaining: list[str] = []
    while True:
        try:
            # lstat 可见悬空链接；其严格解析失败时拒绝，不误当普通新路径。
            path.lstat()
            ancestor = path.resolve(strict=True)
            return str(ancestor.joinpath(*reversed(remaining)).resolve(strict=False))
        except FileNotFoundError:
            if path.is_symlink() or path == path.parent:
                raise
            remaining.append(path.name)
            path = path.parent


def contained(root: str, path: str) -> bool:
    try:
        resolved = os.path.normcase(eval_symlinks_or_ancestor(path))
        normalized_root = os.path.normcase(root)
        return os.path.commonpath((normalized_root, resolved)) == normalized_root
    except (OSError, ValueError, RuntimeError):
        return False


def sandbox_ok(engine: "Engine", path: str, *, internal: str) -> bool:
    absolute = str(Path(engine.root) / (path or "."))
    roots: tuple[str, ...] = (engine.root,)
    if internal == "read_file":
        roots += engine.resource_roots
    for root in roots:
        # 根在引擎创建时已经规范化；再次跟随被替换的根会扩大围栏。
        if contained(root, absolute):
            return True
    return False
