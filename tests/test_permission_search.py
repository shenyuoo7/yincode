"""搜索工具不能读取搜索根外的链接目标。"""

import asyncio
import os
import subprocess
from pathlib import Path

import pytest


async def test_glob_rejects_parent_segments(tmp_path):
    from yincode.tool.glob_tool import GlobTool

    result = await GlobTool(tmp_path).execute('{"pattern":"../**"}')
    assert result.is_error


async def test_glob_does_not_traverse_windows_junction(tmp_path):
    from yincode.tool.glob_tool import GlobTool

    if os.name != "nt":
        pytest.skip("Windows 目录联接专用")
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("private fixture")
    await asyncio.to_thread(
        subprocess.run,
        ["cmd", "/c", "mklink", "/J", str(root / "junction"), str(outside)],
        check=True,
        capture_output=True,
    )
    result = await GlobTool(root).execute('{"pattern":"**/*"}')
    assert "secret.txt" not in result.content


def test_windows_junction_cannot_authorize_outside_new_file(tmp_path):
    from yincode.permission import new_engine
    from yincode.permission.sandbox import sandbox_ok

    if os.name != "nt":
        pytest.skip("Windows 目录联接专用")
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    # Junction 不需开发者模式，验证原生 Windows 的路径解析。
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(root / "junction"), str(outside)],
        check=True,
        capture_output=True,
    )
    engine, _ = new_engine(str(root))
    assert not sandbox_ok(engine, "junction/new/deep/file", internal="write_file")


def test_project_root_cannot_drift_to_external_junction(tmp_path):
    from yincode.llm import ToolCall
    from yincode.permission import Decision, Mode, new_engine

    if os.name != "nt":
        pytest.skip("Windows 目录联接专用")
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_text("outside fixture")
    engine, _ = new_engine(str(root))
    root.rename(tmp_path / "original")
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(root), str(outside)], check=True, capture_output=True
    )
    decision, _ = engine.check(
        Mode.DEFAULT,
        ToolCall("r", "read_file", '{"path":"secret"}'),
        True,
        registered=True,
        allowed=True,
    )
    assert decision is Decision.DENY


def test_grep_skips_external_file_links(tmp_path):
    from threading import Event

    from yincode.tool.grep_tool import GrepTool

    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("SENSITIVE_OUTSIDE_FIXTURE")
    (root / "safe.txt").write_text("SAFE_INSIDE_FIXTURE")
    try:
        (root / "link.txt").symlink_to(outside)
    except OSError:
        pytest.skip("当前账号不可创建文件符号链接")
    result = GrepTool(root)._execute({"pattern": "FIXTURE"}, Event())
    assert "SAFE_INSIDE" in result.content
    assert "SENSITIVE_OUTSIDE" not in result.content


def test_grep_rechecks_resolved_file_target(tmp_path, monkeypatch):
    from threading import Event

    from yincode.tool.grep_tool import GrepTool

    root = tmp_path / "root"
    root.mkdir()
    candidate = root / "link.txt"
    candidate.write_text("OUTSIDE_FIXTURE")
    outside = tmp_path / "secret"
    outside.write_text("OUTSIDE_FIXTURE")
    original_resolve = Path.resolve

    def changed_target(path, *args, **kwargs):
        if path == candidate:
            return outside
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", changed_target)
    result = GrepTool(root)._execute({"pattern": "FIXTURE"}, Event())
    assert "OUTSIDE_FIXTURE" not in result.content
