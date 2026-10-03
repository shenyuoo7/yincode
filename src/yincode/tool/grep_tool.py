"""逐行、有界地检索文本，不把超长行静默当完整搜索。"""

import asyncio
import multiprocessing
import re
from multiprocessing.connection import Connection
from threading import Event
from typing import Any

from yincode.permission.sandbox import contained

from . import Result
from .glob_tool import _matches
from .registry import OUTPUT_BYTES, _files, _FileTool, _string, _truncate, _utf8_prefix

_LINE_BYTES = 65_536


class GrepTool(_FileTool):
    _name = "grep"
    read_only = True
    _description = (
        "用 Python 正则搜索文本，返回 file:line:content；最多 100 条，超长行标注未完整搜索。"
    )
    _parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Python 正则表达式"},
            "path": {"type": "string"},
            "glob": {"type": "string"},
        },
        "required": ["pattern"],
    }

    async def execute(self, args: str) -> Result:
        """隔离 Python 正则的 GIL 与回溯；取消时终止并回收扫描进程。"""
        context = multiprocessing.get_context("spawn")
        receiving, sending = context.Pipe(duplex=False)
        process = context.Process(target=_scan_worker, args=(str(self.cwd), args, sending))
        process.daemon = True
        starting = asyncio.create_task(asyncio.to_thread(process.start))
        reading: asyncio.Task[Result] | None = None
        finishing: asyncio.Task[None] | None = None

        def reap() -> None:
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=0.5)
                if process.is_alive():
                    process.kill()
                    process.join()

        async def cleanup() -> None:
            try:
                await starting
            except Exception:
                pass
            sending.close()
            await asyncio.to_thread(reap)
            if reading is not None:
                await asyncio.gather(reading, return_exceptions=True)
            receiving.close()
            process.close()

        async def finish() -> None:
            nonlocal finishing
            if finishing is None:
                finishing = asyncio.create_task(cleanup())
            while not finishing.done():
                try:
                    await asyncio.shield(finishing)
                except asyncio.CancelledError:
                    continue
            finishing.result()

        try:
            await asyncio.shield(starting)
            sending.close()
            while not receiving.poll():
                if process.exitcode is not None:
                    raise OSError(f"搜索进程异常退出（{process.exitcode}）")
                await asyncio.sleep(0.01)
            reading = asyncio.create_task(asyncio.to_thread(receiving.recv))
            result = await asyncio.shield(reading)
            finishing = asyncio.create_task(cleanup())
            await asyncio.shield(finishing)
            return result
        except asyncio.CancelledError:
            await finish()
            raise
        except Exception as exc:
            await finish()
            return Result(f"grep 失败: {exc}", is_error=True)

    def _execute(self, data: dict[str, Any], stopped: Event) -> Result:
        expression = re.compile(_string(data, "pattern"))
        root = self._path(_string(data, "path", default="."))
        pattern = _string(data, "glob", default="**/*")
        if not root.exists():
            raise FileNotFoundError(f"搜索路径不存在: {data.get('path', '.')}")
        paths = iter([root]) if root.is_file() else _files(root, stopped)
        hits: list[str] = []
        size = 0
        truncated = False
        incomplete = False
        for path in paths:
            # 搜索根检查不能替代逐文件检查；目录内链接可能指向根外。
            if not contained(str(root if root.is_dir() else root.parent), str(path)):
                continue
            label = path.name if root.is_file() else path.relative_to(root).as_posix()
            if not _matches(label, pattern):
                continue
            try:
                with path.open("rb") as stream:
                    number = 0
                    while not stopped.is_set():
                        raw = stream.readline(_LINE_BYTES + 1)
                        if not raw:
                            break
                        number += 1
                        if b"\0" in raw:
                            break
                        partial = len(raw) > _LINE_BYTES
                        line = _utf8_prefix(
                            raw[:_LINE_BYTES], partial=partial, errors="replace"
                        ).rstrip("\r\n")
                        if partial:
                            incomplete = True
                            while not raw.endswith(b"\n") and not stopped.is_set():
                                raw = stream.readline(_LINE_BYTES)
                                if not raw:
                                    break
                        if expression.search(line):
                            if len(hits) == 100:
                                truncated = True
                                break
                            hit = f"{label}:{number}:{line}"
                            hits.append(hit)
                            size += len(hit.encode("utf-8")) + 1
                            if size > OUTPUT_BYTES:
                                truncated = True
                                break
            except (OSError, UnicodeError):
                continue
            if truncated or stopped.is_set():
                break
        content = "\n".join(hits) or "无匹配"
        if incomplete:
            content += "\n未完整搜索：存在超长行，只检索其开头"
        return Result(_truncate(content, 103, OUTPUT_BYTES, force=truncated or incomplete))


def _scan_worker(cwd: str, args: str, sending: Connection) -> None:
    """spawn 必须使用模块顶层入口；只回传有界文本结果。"""
    from .registry import _parse

    try:
        result = GrepTool(cwd=cwd)._execute(_parse(args), Event())
    except Exception as exc:
        result = Result(_truncate(f"grep 失败: {exc}", 103, OUTPUT_BYTES), is_error=True)
    try:
        sending.send(result)
    finally:
        sending.close()
