"""显式选择 shell，并回收自己拥有的进程树。"""

import asyncio
import ctypes
import os
import shutil
import signal
from ctypes import wintypes
from pathlib import Path
from typing import Any

from . import Result
from .registry import OUTPUT_BYTES, _FileTool, _parse, _run_io, _string, _truncate


class _WindowsJob:
    """挂起启动后登记 Job，再恢复主线程，保证后代自动归属。"""

    def __init__(self) -> None:
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        for name in ["CreateJobObjectW", "OpenProcess", "OpenThread", "CreateToolhelp32Snapshot"]:
            getattr(self.kernel, name).restype = wintypes.HANDLE
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.kernel.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        self.kernel.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.c_void_p,
        ]
        self.kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self.kernel.Thread32First.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
        self.kernel.Thread32Next.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
        self.kernel.ResumeThread.argtypes = [wintypes.HANDLE]
        self.kernel.ResumeThread.restype = wintypes.DWORD
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())

        class BasicLimits(ctypes.Structure):
            _fields_ = [
                ("process_time", ctypes.c_int64),
                ("job_time", ctypes.c_int64),
                ("flags", wintypes.DWORD),
                ("minimum", ctypes.c_size_t),
                ("maximum", ctypes.c_size_t),
                ("process_limit", wintypes.DWORD),
                ("affinity", ctypes.c_size_t),
                ("priority", wintypes.DWORD),
                ("scheduling", wintypes.DWORD),
            ]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("basic", BasicLimits),
                ("io", ctypes.c_uint64 * 6),
                ("process_memory", ctypes.c_size_t),
                ("job_memory", ctypes.c_size_t),
                ("peak_process", ctypes.c_size_t),
                ("peak_job", ctypes.c_size_t),
            ]

        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.kernel.SetInformationJobObject(
            self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            self.close()
            raise ctypes.WinError(ctypes.get_last_error())

    def attach_and_resume(self, pid: int) -> None:
        process = self.kernel.OpenProcess(0x0101, False, pid)
        if not process:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not self.kernel.AssignProcessToJobObject(self.handle, process):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            self.kernel.CloseHandle(process)

        class ThreadEntry(ctypes.Structure):
            _fields_ = [
                ("size", wintypes.DWORD),
                ("usage", wintypes.DWORD),
                ("thread_id", wintypes.DWORD),
                ("process_id", wintypes.DWORD),
                ("base_priority", wintypes.LONG),
                ("delta_priority", wintypes.LONG),
                ("flags", wintypes.DWORD),
            ]

        snapshot = self.kernel.CreateToolhelp32Snapshot(4, 0)
        if snapshot == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        entry = ThreadEntry()
        entry.size = ctypes.sizeof(entry)
        try:
            found = self.kernel.Thread32First(snapshot, ctypes.byref(entry))
            while found:
                if entry.process_id == pid:
                    thread = self.kernel.OpenThread(0x0002, False, entry.thread_id)
                    if not thread:
                        raise ctypes.WinError(ctypes.get_last_error())
                    try:
                        if self.kernel.ResumeThread(thread) == 0xFFFFFFFF:
                            raise ctypes.WinError(ctypes.get_last_error())
                        return
                    finally:
                        self.kernel.CloseHandle(thread)
                found = self.kernel.Thread32Next(snapshot, ctypes.byref(entry))
            raise OSError("找不到 shell 主线程，无法恢复执行")
        finally:
            self.kernel.CloseHandle(snapshot)

    def terminate(self) -> None:
        if not self.kernel.TerminateJobObject(self.handle, 1):
            raise ctypes.WinError(ctypes.get_last_error())

    async def wait_empty(self) -> None:
        class Accounting(ctypes.Structure):
            _fields_ = [
                ("times", ctypes.c_int64 * 4),
                ("faults", wintypes.DWORD),
                ("total", wintypes.DWORD),
                ("active", wintypes.DWORD),
                ("terminated", wintypes.DWORD),
            ]

        accounting = Accounting()
        while True:
            if not self.kernel.QueryInformationJobObject(
                self.handle, 1, ctypes.byref(accounting), ctypes.sizeof(accounting), None
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            if accounting.active == 0:
                return
            await asyncio.sleep(0.01)

    def close(self) -> None:
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


class BashTool(_FileTool):
    _name = "bash"
    _parameters = {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    }

    def __init__(self, cwd: Path | str | None = None) -> None:
        super().__init__(cwd)
        self._description = (
            "在工作目录使用 PowerShell（优先 pwsh）执行命令，返回 stdout、stderr 与退出码。"
            if os.name == "nt"
            else "在工作目录使用 /bin/sh 执行命令，返回 stdout、stderr 与退出码。"
        )

    def _argv(self, command: str) -> list[str]:
        if os.name == "nt":
            shell = shutil.which("pwsh") or shutil.which("powershell")
            if shell is None:
                raise OSError("未找到 PowerShell（pwsh 或 powershell）")
            prefix = (
                "$OutputEncoding = [Console]::OutputEncoding = "
                "[System.Text.UTF8Encoding]::new($false); "
            )
            return [shell, "-NoProfile", "-NonInteractive", "-Command", prefix + command]
        return ["/bin/sh", "-c", command]

    async def execute(self, args: str) -> Result:
        try:
            command = await _run_io(lambda stopped: _string(_parse(args), "command"))
            argv = self._argv(command)
            job = _WindowsJob() if os.name == "nt" else None
        except (OSError, ValueError, TypeError) as exc:
            return Result(f"bash 失败: {exc}", is_error=True)
        process: asyncio.subprocess.Process | None = None
        readers: list[asyncio.Task[bytes]] = []
        environment = os.environ.copy()
        environment.setdefault("PYTHONIOENCODING", "utf-8")
        options: dict[str, Any] = {"cwd": self.cwd, "env": environment}
        if os.name == "nt":
            options["creationflags"] = 0x08000204  # 无窗口、新进程组、挂起启动
        else:
            options["start_new_session"] = True
        spawning = asyncio.create_task(
            asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **options,
            )
        )
        remaining = OUTPUT_BYTES
        truncated = False

        async def drain(stream: asyncio.StreamReader) -> bytes:
            nonlocal remaining, truncated
            chunks = bytearray()
            while chunk := await stream.read(8192):
                keep = min(remaining, len(chunk))
                chunks.extend(chunk[:keep])
                remaining -= keep
                truncated |= keep < len(chunk)
            return bytes(chunks)

        async def cleanup() -> None:
            nonlocal process
            if process is None:
                try:
                    process = await spawning
                except Exception:
                    return
            if job is not None:
                job.terminate()
            else:
                try:
                    kill_group = getattr(os, "killpg", None)
                    if kill_group is not None:
                        kill_group(process.pid, getattr(signal, "SIGKILL", signal.SIGTERM))
                except ProcessLookupError:
                    pass
            if process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
            await process.wait()
            if job is not None:
                await job.wait_empty()
            if readers:
                await asyncio.gather(*readers)

        async def settle() -> None:
            finishing = asyncio.create_task(cleanup())
            while not finishing.done():
                try:
                    await asyncio.shield(finishing)
                except asyncio.CancelledError:
                    continue
            finishing.result()

        try:
            process = await asyncio.shield(spawning)
            if job is not None:
                job.attach_and_resume(process.pid)
            assert process.stdout is not None and process.stderr is not None
            readers = [
                asyncio.create_task(drain(process.stdout)),
                asyncio.create_task(drain(process.stderr)),
            ]
            code = await process.wait()
            await cleanup()
            stdout, stderr = [task.result().decode("utf-8", errors="replace") for task in readers]
            content = f"exit_code: {code}\nstdout:\n{stdout}\nstderr:\n{stderr}"
            return Result(
                _truncate(content, 2000, OUTPUT_BYTES, force=truncated), is_error=code != 0
            )
        except asyncio.CancelledError:
            await settle()
            raise
        except (OSError, ValueError) as exc:
            await settle()
            return Result(f"bash 失败: {exc}", is_error=True)
        finally:
            if job is not None:
                job.close()
