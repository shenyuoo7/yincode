"""工具的磁盘副作用、边界与进程生命周期验证。"""

import asyncio
import importlib
import json
import multiprocessing
import os
import shlex
import stat
import sys
import time
from pathlib import Path

import pytest

_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"


def make_registry(cwd: Path):
    return importlib.import_module("yincode.tool").new_default_registry(cwd=cwd)


async def test_overflow_json_number_is_rejected_before_writing(tmp_path: Path):
    registry = make_registry(tmp_path)
    result = await registry.execute(
        "write_file", '{"path":"never.txt","content":"okay","extra":1e999}'
    )
    assert result.is_error
    assert not (tmp_path / "never.txt").exists()


def args(**values: object) -> str:
    return json.dumps(values, ensure_ascii=False)


def python_command(code: str) -> str:
    if os.name == "nt":
        return "& '" + sys.executable.replace("'", "''") + "' -c '" + code.replace("'", "''") + "'"
    return shlex.quote(sys.executable) + " -c " + shlex.quote(code)


def test_registry_definitions_and_duplicate_registration(tmp_path: Path):
    registry = make_registry(tmp_path)
    assert [d.name for d in registry.definitions()] == [
        "read_file",
        "write_file",
        "edit_file",
        "bash",
        "glob",
        "grep",
    ]
    for definition in registry.definitions():
        assert definition.input_schema["type"] == "object"
        assert definition.description
        assert registry.get(definition.name) is not None
    assert registry.get("missing") is None
    with pytest.raises(ValueError):
        registry.register(registry.get("read_file"))


async def test_registry_unknown_tool_is_error(tmp_path: Path):
    result = await make_registry(tmp_path).execute("missing", "{}")
    assert result.is_error
    assert "missing" in result.content


async def test_read_file_returns_numbered_utf8_lines(tmp_path: Path):
    (tmp_path / "demo.txt").write_text("你好\nsecond\n", encoding="utf-8")
    result = await make_registry(tmp_path).execute("read_file", args(path="demo.txt"))
    assert not result.is_error
    assert result.content.splitlines() == ["     1\t你好", "     2\tsecond"]


@pytest.mark.parametrize("path", ["missing", "."])
async def test_read_file_reports_unreadable_paths(tmp_path: Path, path: str):
    assert (await make_registry(tmp_path).execute("read_file", args(path=path))).is_error


async def test_write_file_creates_parent_and_overwrites_utf8(tmp_path: Path):
    registry = make_registry(tmp_path)
    path = tmp_path / "a" / "b.txt"
    for content in ["你好", ""]:
        result = await registry.execute("write_file", args(path="a/b.txt", content=content))
        assert not result.is_error
        assert path.read_text(encoding="utf-8") == content


async def test_edit_file_requires_unique_nonempty_match(tmp_path: Path):
    path = tmp_path / "demo.txt"
    path.write_text("one\none\nunique", encoding="utf-8")
    registry = make_registry(tmp_path)
    zero = await registry.execute(
        "edit_file", args(path="demo.txt", old_string="absent", new_string="x")
    )
    many = await registry.execute(
        "edit_file", args(path="demo.txt", old_string="one", new_string="x")
    )
    empty = await registry.execute(
        "edit_file", args(path="demo.txt", old_string="", new_string="x")
    )
    assert zero.is_error and many.is_error and empty.is_error
    assert zero.content != many.content
    assert "0" in zero.content and "2" in many.content
    assert path.read_text(encoding="utf-8") == "one\none\nunique"
    okay = await registry.execute(
        "edit_file", args(path="demo.txt", old_string="unique", new_string="新")
    )
    assert not okay.is_error
    assert path.read_text(encoding="utf-8") == "one\none\n新"


async def test_edit_file_rejects_overlapping_matches_without_writing(tmp_path: Path):
    path = tmp_path / "demo.txt"
    path.write_text("banana", encoding="utf-8")
    result = await make_registry(tmp_path).execute(
        "edit_file", args(path="demo.txt", old_string="ana", new_string="X")
    )
    assert result.is_error and "2" in result.content
    assert path.read_text(encoding="utf-8") == "banana"


@pytest.mark.parametrize(
    "tool,raw",
    [
        ("read_file", "[1]"),
        ("read_file", "null"),
        ("read_file", "broken"),
        ("read_file", '{"path":12}'),
        ("read_file", '{"path":""}'),
        ("write_file", '{"path":"x","content":false}'),
        ("write_file", '{"path":"x"}'),
        ("edit_file", '{"path":"x","old_string":true,"new_string":"a"}'),
        ("edit_file", '{"path":"x","old_string":"a","new_string":[]}'),
        ("bash", '{"command":12}'),
        ("bash", '{"command":""}'),
        ("glob", '{"pattern":"*","path":[]}'),
        ("glob", '{"pattern":false}'),
        ("grep", '{"pattern":"["}'),
        ("grep", '{"pattern":"x","glob":false}'),
        ("grep", '{"pattern":"x","path":false}'),
    ],
)
async def test_invalid_parameters_are_structured_errors(tmp_path: Path, tool: str, raw: str):
    implementation = make_registry(tmp_path).get(tool)
    assert implementation is not None
    result = await implementation.execute(raw)
    assert result.is_error
    assert not (tmp_path / "x").exists()


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
async def test_nonfinite_numbers_are_rejected_before_writing(tmp_path: Path, constant: str):
    raw = '{"path":"x","content":"must not write","extra":' + constant + "}"
    result = await make_registry(tmp_path).execute("write_file", raw)
    assert result.is_error
    assert not (tmp_path / "x").exists()


@pytest.mark.parametrize("content", ["x\n" * 2100, "中文" * 150_000], ids=["lines", "bytes"])
async def test_read_limits_include_marker_and_utf8_bytes(tmp_path: Path, content: str):
    (tmp_path / "large.txt").write_text(content, encoding="utf-8")
    result = await make_registry(tmp_path).execute("read_file", args(path="large.txt"))
    assert not result.is_error
    assert "[truncated]" in result.content
    assert len(result.content.encode("utf-8")) <= 256 * 1024
    assert len(result.content.splitlines()) <= 2000
    assert "�" not in result.content


async def test_glob_handles_root_and_nested_files_with_stable_limit(tmp_path: Path):
    (tmp_path / "sub").mkdir()
    for n in range(120, -1, -1):
        (tmp_path / f"{n:03}.py").write_text("", encoding="utf-8")
    (tmp_path / "sub" / "nested.py").write_text("", encoding="utf-8")
    (tmp_path / "dir.py").mkdir()
    result = await make_registry(tmp_path).execute("glob", args(pattern="**/*.py"))
    paths = [line for line in result.content.splitlines() if line != "[truncated]"]
    assert not result.is_error
    assert paths == [f"{n:03}.py" for n in range(100)]
    assert "[truncated]" in result.content
    nested = await make_registry(tmp_path).execute("glob", args(pattern="sub/*.py"))
    assert nested.content == "sub/nested.py"


@pytest.mark.parametrize("pattern", ["./*.txt", ".\\*.txt"])
async def test_glob_and_grep_accept_current_directory_pattern_prefix(tmp_path: Path, pattern: str):
    (tmp_path / "mixed.txt").write_text("target", encoding="utf-8")
    registry = make_registry(tmp_path)
    assert (await registry.execute("glob", args(pattern=pattern))).content == "mixed.txt"
    assert (
        await registry.execute("grep", args(pattern="target", glob=pattern))
    ).content == "mixed.txt:1:target"


@pytest.mark.skipif(os.name == "nt", reason="Windows chmod 不提供 POSIX 执行位")
async def test_edit_and_overwrite_preserve_existing_file_mode(tmp_path: Path):
    path = tmp_path / "script.py"
    path.write_text("old", encoding="utf-8")
    path.chmod(0o755)
    registry = make_registry(tmp_path)
    result = await registry.execute(
        "edit_file", args(path="script.py", old_string="old", new_string="new")
    )
    assert not result.is_error
    assert path.stat().st_mode & 0o777 == 0o755
    result = await registry.execute("write_file", args(path="script.py", content="overwritten"))
    assert not result.is_error
    assert path.stat().st_mode & 0o777 == 0o755


@pytest.mark.skipif(os.name != "nt", reason="Windows 只读属性的拒绝与清理行为")
@pytest.mark.parametrize("tool", ["write_file", "edit_file"])
async def test_readonly_target_errors_without_changing_content_mode_or_leaving_temp(
    tmp_path: Path, tool: str
):
    target = tmp_path / "readonly.txt"
    target.write_text("old", encoding="utf-8")
    target.chmod(0o444)
    original = target.stat()
    raw = (
        args(path="readonly.txt", content="new")
        if tool == "write_file"
        else args(path="readonly.txt", old_string="old", new_string="new")
    )
    try:
        result = await make_registry(tmp_path).execute(tool, raw)
        assert result.is_error
        assert target.read_text(encoding="utf-8") == "old"
        assert target.stat().st_mode == original.st_mode
        assert target.stat().st_file_attributes == original.st_file_attributes
        assert await asyncio.to_thread(lambda: set(tmp_path.iterdir())) == {target}
    finally:
        for path in await asyncio.to_thread(lambda: list(tmp_path.iterdir())):
            path.chmod(0o666)


@pytest.mark.skipif(os.name != "nt", reason="Windows 只读临时文件不能直接 unlink")
@pytest.mark.parametrize("tool", ["write_file", "edit_file"])
async def test_failed_replace_removes_its_readonly_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool: str
):
    target = tmp_path / "target.txt"
    target.write_text("old", encoding="utf-8")
    original = target.stat()

    def fail_replace(temporary: Path, destination: Path) -> Path:
        temporary.chmod(stat.S_IREAD)
        raise PermissionError("模拟替换失败，临时文件已变只读")

    monkeypatch.setattr(Path, "replace", fail_replace)
    raw = (
        args(path="target.txt", content="new")
        if tool == "write_file"
        else args(path="target.txt", old_string="old", new_string="new")
    )
    try:
        result = await make_registry(tmp_path).execute(tool, raw)
        assert result.is_error
        assert target.read_text(encoding="utf-8") == "old"
        assert target.stat().st_mode == original.st_mode
        assert target.stat().st_file_attributes == original.st_file_attributes
        assert await asyncio.to_thread(lambda: set(tmp_path.iterdir())) == {target}
    finally:
        for path in await asyncio.to_thread(lambda: list(tmp_path.iterdir())):
            path.chmod(0o666)


@pytest.mark.parametrize("tool", ["glob", "grep"])
async def test_no_matches_is_normal_result(tmp_path: Path, tool: str):
    result = await make_registry(tmp_path).execute(tool, args(pattern="nomatch"))
    assert not result.is_error
    assert "无匹配" in result.content


async def test_grep_reports_filename_line_and_regex_hits(tmp_path: Path):
    (tmp_path / "sample.py").write_text("nothing\n你好 42\n你好 43\n", encoding="utf-8")
    (tmp_path / "other.txt").write_text("你好 44", encoding="utf-8")
    result = await make_registry(tmp_path).execute("grep", args(pattern=r"你好 \d+", glob="*.py"))
    assert not result.is_error
    assert result.content.splitlines() == ["sample.py:2:你好 42", "sample.py:3:你好 43"]
    direct = await make_registry(tmp_path).execute("grep", args(pattern="42", path="sample.py"))
    assert direct.content == "sample.py:2:你好 42"


async def test_grep_searches_text_with_invalid_utf8_using_replacement(tmp_path: Path):
    (tmp_path / "mixed.txt").write_bytes(b"\xff target\n")
    result = await make_registry(tmp_path).execute("grep", args(pattern="target"))
    assert not result.is_error
    assert result.content == "mixed.txt:1:� target"


async def test_grep_caps_hits_and_utf8_bytes(tmp_path: Path):
    (tmp_path / "hits.txt").write_text("中文 hit\n" * 150, encoding="utf-8")
    registry = make_registry(tmp_path)
    result = await registry.execute("grep", args(pattern="hit"))
    hits = [line for line in result.content.splitlines() if line.startswith("hits.txt:")]
    assert len(hits) == 100
    assert "[truncated]" in result.content
    (tmp_path / "hits.txt").write_text(("中" * 10_000 + "hit\n") * 20, encoding="utf-8")
    result = await registry.execute("grep", args(pattern="hit"))
    assert len(result.content.encode("utf-8")) <= 30_000
    assert "[truncated]" in result.content


async def test_grep_reports_long_line_was_not_fully_searched(tmp_path: Path):
    (tmp_path / "long.txt").write_text("x" * 500_000 + "target\nnext target\n", encoding="utf-8")
    result = await make_registry(tmp_path).execute("grep", args(pattern="target"))
    assert not result.is_error
    assert "long.txt:2:next target" in result.content
    assert "未完整搜索" in result.content and "[truncated]" in result.content


async def test_grep_backtracking_timeout_keeps_loop_responsive_and_reaps_process(tmp_path: Path):
    (tmp_path / "regex.txt").write_text("a" * 22 + "!", encoding="utf-8")
    baseline = {child.pid for child in multiprocessing.active_children()}
    moments: list[float] = []

    async def heartbeat():
        while True:
            moments.append(time.monotonic())
            await asyncio.sleep(0.01)

    ticker = asyncio.create_task(heartbeat())
    started = time.monotonic()
    try:
        result = await make_registry(tmp_path).execute("grep", args(pattern="(a+)+$"), timeout=0.05)
        elapsed = time.monotonic() - started
        moments.append(time.monotonic())
        assert result.is_error and "超时" in result.content
        assert elapsed < 0.5
        assert max((b - a for a, b in zip(moments, moments[1:], strict=False)), default=0) < 0.2
        assert {child.pid for child in multiprocessing.active_children()} == baseline
    finally:
        ticker.cancel()
        with pytest.raises(asyncio.CancelledError):
            await ticker


async def test_grep_cancel_reaps_scan_process(tmp_path: Path):
    (tmp_path / "regex.txt").write_text("a" * 22 + "!", encoding="utf-8")
    baseline = {child.pid for child in multiprocessing.active_children()}
    task = asyncio.create_task(make_registry(tmp_path).execute("grep", args(pattern="(a+)+$")))
    deadline = time.monotonic() + 2
    spawned: list[int] = []
    while not spawned:
        spawned = [
            child.pid for child in multiprocessing.active_children() if child.pid not in baseline
        ]
        assert time.monotonic() < deadline, "扫描进程未启动"
        await asyncio.sleep(0.005)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert {child.pid for child in multiprocessing.active_children()} == baseline
    assert all(not process_exists(pid) for pid in spawned)


async def test_bash_uses_cwd_and_returns_both_streams_with_exit_code(tmp_path: Path):
    code = 'import os,sys; print(os.getcwd()); print("err",file=sys.stderr); sys.exit(7)'
    command = python_command(code)
    if os.name == "nt":
        command += "; exit $LASTEXITCODE"
    result = await make_registry(tmp_path).execute("bash", args(command=command))
    assert result.is_error
    assert str(tmp_path) in result.content
    assert "err" in result.content
    assert "exit_code: 7" in result.content


async def test_bash_native_failure_is_error(tmp_path: Path):
    command = python_command("import sys; sys.exit(7)")
    result = await make_registry(tmp_path).execute("bash", args(command=command))
    assert result.is_error
    assert "exit_code: 0" not in result.content


async def test_bash_reports_success_when_later_shell_command_recovers(tmp_path: Path):
    command = python_command("import sys; sys.exit(7)")
    command += "; Write-Output 'recovered'" if os.name == "nt" else "; printf recovered"
    result = await make_registry(tmp_path).execute("bash", args(command=command))
    assert not result.is_error
    assert "exit_code: 0" in result.content and "recovered" in result.content


async def test_bash_native_stdin_is_eof_and_cannot_consume_parent_draft(tmp_path: Path):
    command = python_command("import sys; print(repr(sys.stdin.read()))")
    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text(
        "import asyncio,json\n"
        "from yincode.tool import new_default_registry\n"
        "async def run():\n"
        f"    result = await new_default_registry().execute('bash', {args(command=command)!r})\n"
        "    print(json.dumps({'content':result.content,'is_error':result.is_error}))\n"
        "if __name__ == '__main__':\n"
        "    asyncio.run(run())\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(_SOURCE_ROOT)
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(wrapper),
        cwd=tmp_path,
        env=environment,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(process.communicate(b"TUI-DRAFT-SENTINEL\n"), 10)
    assert process.returncode == 0, stderr.decode(errors="replace")
    result = json.loads(stdout)
    assert not result["is_error"]
    assert "TUI-DRAFT-SENTINEL" not in result["content"]
    assert "''" in result["content"]


async def test_bash_drains_large_output_and_keeps_bounded_utf8_result(tmp_path: Path):
    code = (
        'import sys; sys.stdout.buffer.write(("中文"*200000).encode()); '
        'sys.stderr.write("err"*200000)'
    )
    result = await make_registry(tmp_path).execute(
        "bash", args(command=python_command(code)), timeout=10
    )
    assert not result.is_error
    assert "exit_code: 0" in result.content
    assert "[truncated]" in result.content
    assert len(result.content.encode("utf-8")) <= 30_000


async def test_bash_normal_exit_reaps_background_child_holding_output_pipe(tmp_path: Path):
    (tmp_path / "background.py").write_text("import time; time.sleep(60)", encoding="utf-8")
    if os.name == "nt":
        executable = sys.executable.replace("'", "''")
        command = (
            f"$p = Start-Process -FilePath '{executable}' -ArgumentList 'background.py' "
            "-NoNewWindow -PassThru; $p.Id | Set-Content 'pid.txt'; exit 0"
        )
    else:
        command = shlex.quote(sys.executable) + " background.py & echo $! > pid.txt; exit 0"
    result = await make_registry(tmp_path).execute("bash", args(command=command), timeout=2.5)
    assert not result.is_error
    assert "exit_code: 0" in result.content
    pid = int((tmp_path / "pid.txt").read_text().strip())
    assert not process_exists(pid)


def process_exists(pid: int) -> bool:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            status = wintypes.DWORD()
            return (
                bool(kernel.GetExitCodeProcess(handle, ctypes.byref(status)))
                and status.value == 259
            )
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.mark.parametrize("cancel", [False, True])
async def test_bash_timeout_and_cancel_reap_child_process_tree(tmp_path: Path, cancel: bool):
    script = tmp_path / "spawn.py"
    script.write_text(
        "import os,subprocess,sys,time\n"
        "from pathlib import Path\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "Path('pids.tmp').write_text(str(os.getpid()) + ',' + str(child.pid))\n"
        "Path('pids.tmp').replace('pids.txt')\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    command = python_command("exec(open('spawn.py', encoding='utf-8').read())")
    registry = make_registry(tmp_path)
    task = asyncio.create_task(registry.execute("bash", args(command=command), timeout=2.5))
    try:
        deadline = time.monotonic() + 2
        while not (tmp_path / "pids.txt").exists():
            assert time.monotonic() < deadline, "命令未启动"
            await asyncio.sleep(0.01)
        pids = [int(value) for value in (tmp_path / "pids.txt").read_text().split(",")]
        assert all(process_exists(pid) for pid in pids)
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            result = await task
            assert result.is_error and "超时" in result.content
        assert all(not process_exists(pid) for pid in pids)
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


async def test_long_file_io_keeps_event_loop_responsive_and_cancel_finishes_io(tmp_path: Path):
    registry = make_registry(tmp_path)
    content = "中文\n" * 2_000_000
    ticks = 0

    async def tick():
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.001)

    ticker = asyncio.create_task(tick())
    writing = asyncio.create_task(
        registry.execute("write_file", args(path="large.txt", content=content))
    )
    try:
        await asyncio.sleep(0.01)
        writing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await writing
        assert ticks >= 2
        path = tmp_path / "large.txt"
        snapshot = path.read_bytes() if path.exists() else None
        await asyncio.sleep(0.05)
        assert (path.read_bytes() if path.exists() else None) == snapshot
    finally:
        ticker.cancel()
        with pytest.raises(asyncio.CancelledError):
            await ticker


async def test_default_registry_keeps_captured_absolute_cwd(tmp_path: Path):
    original = Path.cwd()
    (tmp_path / "captured.txt").write_text("captured", encoding="utf-8")
    result = await make_registry(tmp_path).execute("read_file", args(path="captured.txt"))
    assert not result.is_error and "captured" in result.content
    assert Path.cwd() == original
