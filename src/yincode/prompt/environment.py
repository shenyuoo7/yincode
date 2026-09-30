"""会话环境摘要；不收集环境变量或 Git 文件路径。"""

import asyncio
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass
class Environment:
    working_dir: str
    platform: str
    date: str
    git_status: str
    version: str
    model: str
    shell: str = ""

    def render(self) -> str:
        """空值省略，全部为空时不留下空标题。"""
        fields = [
            ("Working directory", self.working_dir),
            ("Platform", self.platform),
            ("Date", self.date),
            ("Git status", self.git_status),
            ("Version", self.version),
            ("Model", self.model),
            ("Shell", self.shell),
        ]
        lines = [f"{key}: {value}" for key, value in fields if value]
        return "Environment information:\n" + "\n".join(lines) if lines else ""


def _git_summary(cwd: str) -> str:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "-z"],
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if result.returncode != 0:
        return ""
    # -z 不把路径中的换行误当记录；重命名/复制的第二个路径不是另一个改动。
    records = iter(result.stdout.split("\0"))
    count = 0
    for record in records:
        if not record:
            continue
        count += 1
        if "R" in record[:2] or "C" in record[:2]:
            next(records, None)
    return f"{count} changed files" if count else "clean"


def _shell_name() -> str:
    if os.name == "nt":
        # 只定位可执行程序，既不采集 PATH，也不把安装路径放入提示。
        return "pwsh (PowerShell 7)" if shutil.which("pwsh") else "powershell (Windows PowerShell)"
    return "/bin/sh"


def _gather(version: str, model: str, cwd: Path | str | None) -> Environment:
    try:
        working_dir = str(Path(cwd if cwd is not None else Path.cwd()).resolve())
    except OSError:
        working_dir = ""
    git_status = _git_summary(working_dir) if working_dir else ""
    return Environment(
        working_dir,
        sys.platform,
        date.today().isoformat(),
        git_status,
        version,
        model,
        _shell_name(),
    )


async def gather_environment(
    version: str, model: str, *, cwd: Path | str | None = None
) -> Environment:
    """显式目录优先；取消仍等待有两秒超时的自有线程子过程回收。"""
    worker = asyncio.create_task(asyncio.to_thread(_gather, version, model, cwd))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        # shield 保留所有权；重复取消也不能使已启动的子过程变成后台孤儿。
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except Exception:
                # 已收尾的采集错误不能覆盖先前的取消信号。
                break
        if not worker.cancelled():
            worker.exception()
        raise
