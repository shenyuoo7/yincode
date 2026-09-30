import asyncio
import importlib
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from datetime import date
from threading import Event

import pytest
from rich.cells import cell_len


def test_banner_is_readable_at_narrow_width():
    mod = importlib.import_module("yincode.prompt")
    banner = mod.render_banner("0.1.0", "E:/very/long/project/path", width=24)
    assert "YIN" in banner
    assert "yincode v0.1.0" in banner
    assert "E:/very/long/project/path" in banner.replace("\n", "")
    assert "Enter" in banner


def test_banner_wraps_chinese_paths_by_terminal_cells():
    mod = importlib.import_module("yincode.prompt")
    cwd = "E:/中文项目目录/很长的中文工作目录"
    banner = mod.render_banner("0.1.0", cwd, width=24)
    assert cwd in banner.replace("\n", "")
    assert all(cell_len(line) <= 24 for line in banner.splitlines())


def test_modules_sort_stably_skip_empty_and_allow_extension():
    from yincode.prompt import assemble_system
    from yincode.prompt.modules import Module

    mods = [
        Module("later", 30, "last"),
        Module("first", 10, "first"),
        Module("empty", 15, ""),
        Module("tie", 10, "second"),
        Module("extension", 20, "extra"),
    ]
    assert assemble_system(mods) == "first\n\nsecond\n\nextra\n\nlast"
    assert mods[0].name == "later"
    assert assemble_system([Module("empty", 1, "")]) == ""
    with pytest.raises(FrozenInstanceError):
        mods[0].content = "changed"


def test_stable_prompt_has_seven_modules_three_empty_slots_and_tool_rules():
    from yincode.prompt import build_system_prompt
    from yincode.prompt.modules import fixed_modules, optional_modules

    fixed = fixed_modules()
    assert len(fixed) == 7
    assert [m.priority for m in fixed] == [10, 20, 30, 40, 50, 60, 70]
    assert len(optional_modules()) == 3
    assert all(m.content == "" for m in optional_modules())
    stable = build_system_prompt()
    assert stable == build_system_prompt()
    assert stable.startswith("You are yincode")
    assert stable.count("\n\n") == 6
    assert "read_file" in stable and "glob" in stable and "grep" in stable
    assert "before editing" in stable.lower()
    assert "prefer" in stable.lower()
    for dynamic in [str(date.today()), str(os.getcwd()), sys.platform, "PowerShell"]:
        assert dynamic not in stable


def test_dedicated_tools_and_read_before_edit_are_reinforced_in_descriptions():
    from yincode.prompt import build_system_prompt
    from yincode.tool.bash import BashTool
    from yincode.tool.edit_file import EditFileTool

    stable = build_system_prompt()
    bash = BashTool()
    edit = EditFileTool()
    for name in ["read_file", "glob", "grep"]:
        assert name in stable
        assert name in bash.description
    assert "read_file" in edit.description
    assert "old_string" in edit.description
    assert "编辑前" in edit.description
    assert "优先" in bash.description


def test_reminders_are_temporary_context_not_user_questions_or_trust_elevation():
    from yincode.prompt import EXECUTE_DIRECTIVE, plan_reminder
    from yincode.prompt.reminder import system_reminder

    full, concise = plan_reminder(True), plan_reminder(False)
    assert len(full) > len(concise)
    for text in [full, concise]:
        assert text.startswith("<system-reminder>\n")
        assert text.endswith("\n</system-reminder>")
        assert "read-only" in text
        assert "/do" in text
        assert "not a user question" in text
        assert "file or tool output" in text
    assert "plan" in full and "step" in full
    assert "Execute" in EXECUTE_DIRECTIVE
    assert system_reminder("context") == "<system-reminder>\ncontext\n</system-reminder>"


def test_environment_render_omits_empty_fields():
    from yincode.prompt import Environment

    env = Environment("E:/workspace", "win32", "2026-10-01", "clean", "0.1", "model", "pwsh")
    rendered = env.render()
    for value in ["E:/workspace", "win32", "2026-10-01", "clean", "0.1", "model", "pwsh"]:
        assert value in rendered
    assert Environment("", "", "", "", "", "").render() == ""


async def test_environment_uses_explicit_cwd_and_only_git_counts(tmp_path, monkeypatch):
    from yincode.prompt import build_system_prompt, environment, gather_environment

    original_cwd = os.getcwd()
    observed = {}

    def run(args, **kwargs):
        observed.update(kwargs)
        assert args == ["git", "status", "--porcelain", "-z"]
        return subprocess.CompletedProcess(args, 0, " M .env.secret\0R  private-new\0private-old\0")

    monkeypatch.setattr(environment.subprocess, "run", run)
    monkeypatch.setenv("PROMPT_SECRET", "credential-never-send")
    before = build_system_prompt()
    env = await gather_environment("0.1", "test-model", cwd=tmp_path)
    assert env.working_dir == str(tmp_path.resolve())
    assert env.git_status == "2 changed files"
    assert observed["cwd"] == str(tmp_path.resolve())
    assert observed["stdin"] == subprocess.DEVNULL
    assert observed["timeout"] == 2.0
    assert os.getcwd() == original_cwd
    assert env.platform == sys.platform
    assert env.date == date.today().isoformat()
    assert env.shell
    assert build_system_prompt() == before
    for private in [
        ".env.secret",
        "private-new",
        "private-old",
        "credential-never-send",
        "PROMPT_SECRET",
    ]:
        assert private not in env.render()


@pytest.mark.parametrize(
    "result",
    [
        FileNotFoundError(),
        PermissionError(),
        subprocess.TimeoutExpired("git", 2),
        subprocess.CompletedProcess(["git"], 128, "", "not a repository"),
    ],
)
async def test_environment_git_errors_degrade_to_empty_status(tmp_path, monkeypatch, result):
    from yincode.prompt import environment, gather_environment

    def run(*args, **kwargs):
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(environment.subprocess, "run", run)
    env = await gather_environment("0.1", "model", cwd=tmp_path)
    assert env.git_status == ""
    assert env.working_dir == str(tmp_path)
    assert "Git status:" not in env.render()


async def test_environment_non_git_directory(tmp_path):
    from yincode.prompt import gather_environment

    env = await gather_environment("0.1", "model", cwd=tmp_path)
    assert env.git_status == ""


async def test_environment_defaults_to_process_cwd(tmp_path, monkeypatch):
    from yincode.prompt import environment, gather_environment

    monkeypatch.setattr(environment.Path, "cwd", lambda: tmp_path)
    env = await gather_environment("0.1", "model")
    assert env.working_dir == str(tmp_path)


async def test_environment_missing_process_cwd_degrades_without_git(monkeypatch):
    from yincode.prompt import environment, gather_environment

    def missing_cwd():
        raise OSError("removed directory")

    def forbidden_run(*args, **kwargs):
        raise AssertionError("No usable cwd must not start Git")

    monkeypatch.setattr(environment.Path, "cwd", missing_cwd)
    monkeypatch.setattr(environment.subprocess, "run", forbidden_run)
    env = await gather_environment("0.1", "model")
    assert env.working_dir == env.git_status == ""
    assert env.version == "0.1" and env.model == "model"


async def test_environment_clean_repository_summary(tmp_path, monkeypatch):
    from yincode.prompt import environment, gather_environment

    monkeypatch.setattr(
        environment.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, ""),
    )
    assert (await gather_environment("0.1", "model", cwd=tmp_path)).git_status == "clean"


async def test_environment_git_worker_does_not_block_and_is_joined_on_repeated_cancel(
    tmp_path, monkeypatch
):
    from yincode.prompt import environment, gather_environment

    started, release, finished = asyncio.Event(), Event(), Event()
    loop = asyncio.get_running_loop()

    def run(*args, **kwargs):
        loop.call_soon_threadsafe(started.set)
        try:
            assert release.wait(2)
            return subprocess.CompletedProcess(args, 0, "")
        finally:
            finished.set()

    monkeypatch.setattr(environment.subprocess, "run", run)
    task = asyncio.create_task(gather_environment("0.1", "model", cwd=tmp_path))
    try:
        async with asyncio.timeout(1):
            await started.wait()
        # 若同步调用阻塞主循环，这个等待点无法在工作线程释放前运行。
        assert not finished.is_set()
        task.cancel()
        await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set()


async def test_environment_cancel_waits_for_real_child_timeout_and_cleanup(tmp_path, monkeypatch):
    from yincode.prompt import environment, gather_environment

    run_process = subprocess.run
    started, finished = asyncio.Event(), Event()
    loop = asyncio.get_running_loop()

    def run(*args, **kwargs):
        loop.call_soon_threadsafe(started.set)
        try:
            # 保留产品 timeout/cwd/stdin 参数，替换外部 Git 为可控真实子过程。
            return run_process([sys.executable, "-c", "import time; time.sleep(10)"], **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(environment.subprocess, "run", run)
    task = asyncio.create_task(gather_environment("0.1", "model", cwd=tmp_path))
    async with asyncio.timeout(1):
        await started.wait()
    task.cancel()
    async with asyncio.timeout(4):
        with pytest.raises(asyncio.CancelledError):
            await task
    assert finished.is_set()


async def test_environment_cleanup_error_does_not_replace_cancellation(tmp_path, monkeypatch):
    from yincode.prompt import environment, gather_environment

    started, release = asyncio.Event(), Event()
    loop = asyncio.get_running_loop()

    def run(*args, **kwargs):
        loop.call_soon_threadsafe(started.set)
        release.wait(2)
        raise RuntimeError("unexpected worker failure during cancellation")

    monkeypatch.setattr(environment.subprocess, "run", run)
    task = asyncio.create_task(gather_environment("0.1", "model", cwd=tmp_path))
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
