"""权限决定必须反映实际边界，不依赖模型服从提示。"""

import json
from pathlib import Path

import pytest
import yaml


def call(name, **args):
    from yincode.llm import ToolCall

    return ToolCall("id", name, json.dumps(args))


def engine_at(root, monkeypatch):
    from yincode.permission import new_engine

    monkeypatch.setattr(Path, "home", lambda: root / "user")
    engine, error = new_engine(str(root))
    assert error is None
    return engine


def test_modes_parse_and_matrix():
    from yincode.permission import Category, Decision, Mode, parse_mode
    from yincode.permission.engine import mode_fallback

    names = ["default", "acceptEdits", "plan", "bypassPermissions"]
    for mode, name in zip(Mode, names, strict=True):
        assert str(mode) == name
        assert parse_mode(name.upper()) == (mode, True)
        assert mode_fallback(mode, Category.READ) is Decision.ALLOW
        assert mode_fallback(mode, Category.EXEC) is (
            Decision.ALLOW if mode is Mode.BYPASS else Decision.ASK
        )
        assert mode_fallback(mode, Category.WRITE) is (
            Decision.ALLOW if mode in (Mode.ACCEPT_EDITS, Mode.BYPASS) else Decision.ASK
        )
    assert parse_mode("unknown") == (Mode.DEFAULT, False)


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf /",
        "rm -fr ~",
        "rm --recursive --force /",
        "rm -rf '$HOME'",
        "dd if=/dev/zero of=/dev/sda",
        ":(){ :|:& };:",
        "mkfs.ext4 /dev/sda",
        "echo x > /dev/nvme0n1",
        "chmod -R 777 /",
        "Remove-Item -LiteralPath C:\\ -Recurse -Force",
        "rm -Recurse -Force C:\\",
        "ri -r -fo $HOME",
        "Format-Volume -DriveLetter C",
        "Clear-Disk -Number 0",
    ],
)
def test_blacklist_is_hard_even_with_allow_and_bypass(tmp_path, monkeypatch, command):
    from yincode.permission import Decision, Mode
    from yincode.permission.rule import Rule, RuleSet

    engine = engine_at(tmp_path, monkeypatch)
    engine.local = RuleSet(allow=[Rule("Bash", "", True)])
    decision, reason = engine.check(
        Mode.BYPASS, call("bash", command=command), False, registered=True, allowed=True
    )
    assert decision is Decision.DENY
    assert "黑名单" in reason


@pytest.mark.parametrize("command", ["git status", "rm -rf ./build", "ls -la"])
def test_normal_commands_are_not_blacklisted(command):
    from yincode.permission.blacklist import hits_blacklist

    assert not hits_blacklist(command)


def test_sandbox_new_paths_escape_and_narrow_resources(tmp_path, monkeypatch):
    from yincode.permission import Decision, Mode

    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "resources"
    outside.mkdir()
    engine = engine_at(root, monkeypatch)
    for name, args in [
        ("read_file", {}),
        ("write_file", {"content": "x"}),
        ("edit_file", {"old_string": "x", "new_string": "y"}),
    ]:
        decision, _ = engine.check(
            Mode.BYPASS, call(name, path="../outside", **args), False, registered=True, allowed=True
        )
        assert decision is Decision.DENY
    assert (
        engine.check(
            Mode.BYPASS,
            call("write_file", path="a/b/new", content="x"),
            False,
            registered=True,
            allowed=True,
        )[0]
        is Decision.ALLOW
    )
    engine.resource_roots = (str(outside),)
    assert (
        engine.check(
            Mode.DEFAULT,
            call("read_file", path=str(outside / "x")),
            True,
            registered=True,
            allowed=True,
        )[0]
        is Decision.ALLOW
    )
    assert (
        engine.check(
            Mode.BYPASS,
            call("write_file", path=str(outside / "x"), content="x"),
            False,
            registered=True,
            allowed=True,
        )[0]
        is Decision.DENY
    )


def test_sandbox_resolves_links_and_dangling_links(tmp_path, monkeypatch):
    from yincode.permission.sandbox import sandbox_ok

    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    engine = engine_at(root, monkeypatch)
    try:
        (root / "link").symlink_to(outside, target_is_directory=True)
        (root / "dangling").symlink_to(outside / "missing", target_is_directory=True)
    except OSError:
        pytest.skip("当前账号不可创建符号链接")
    assert not sandbox_ok(engine, "link/new/file", internal="write_file")
    assert not sandbox_ok(engine, "dangling/new", internal="write_file")
    assert not sandbox_ok(engine, "link/../outside-file", internal="write_file")


def test_rule_globs_precedence_and_literal_escaping():
    from yincode.permission import Decision
    from yincode.permission.rule import Rule, RuleSet, match_pattern, parse_rule

    for text in ["Bash(git status)", "Write(src/**)", "Read", "Bash(echo (x))"]:
        assert parse_rule(text)[1]
    for text in ["", "Bash(", "Bogus(x)"]:
        assert not parse_rule(text)[1]
    assert match_pattern("git *", "git status --short", kind="command")
    assert not match_pattern("src/*", "src/a/b", kind="path")
    assert match_pattern("src/**", "src/a/b", kind="path")
    assert match_pattern("src/**/x.py", "src/x.py", kind="path")
    assert match_pattern(r"echo \*", "echo *", kind="command")
    assert not match_pattern(r"echo \*", "echo hello", kind="command")
    rules = RuleSet(allow=[Rule("Bash", "git *", True)], deny=[Rule("Bash", "git push", False)])
    assert rules.match("Bash", "git push") == (Decision.DENY, True)
    assert rules.match("Bash", "git status") == (Decision.ALLOW, True)


def test_settings_layers_default_modes_and_invalid_file(tmp_path, monkeypatch):
    from yincode.permission import Decision, Mode

    home = tmp_path / "user"
    monkeypatch.setattr(Path, "home", lambda: home)
    paths = [
        home / ".yincode/permissions.yaml",
        tmp_path / ".yincode/permissions.yaml",
        tmp_path / ".yincode/permissions.local.yaml",
    ]
    for path, mode, effect in zip(
        paths, ["bypassPermissions", "acceptEdits", "plan"], ["deny", "allow", "deny"], strict=True
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump({"default_mode": mode, "permissions": {effect: ["Bash(git status)"]}})
        )
    from yincode.permission import new_engine

    engine, error = new_engine(str(tmp_path))
    assert error is None and engine.start_mode() is Mode.PLAN
    assert (
        engine.check(
            Mode.DEFAULT, call("bash", command="git status"), False, registered=True, allowed=True
        )[0]
        is Decision.DENY
    )
    paths[2].write_text("permissions: [invalid]")
    engine, error = new_engine(str(tmp_path))
    assert error is None and engine.start_mode() is Mode.ACCEPT_EDITS
    assert (
        engine.check(
            Mode.DEFAULT, call("bash", command="git status"), False, registered=True, allowed=True
        )[0]
        is Decision.ALLOW
    )
    paths[1].write_text("permissions: {allow: [Bash], deny: wrong}")
    engine, _ = new_engine(str(tmp_path))
    assert engine.start_mode() is Mode.BYPASS
    assert (
        engine.check(
            Mode.DEFAULT, call("bash", command="git status"), False, registered=True, allowed=True
        )[0]
        is Decision.DENY
    )


@pytest.mark.parametrize(
    "name,args",
    [
        ("bash", {}),
        ("bash", {"command": 1}),
        ("bash", {"command": " "}),
        ("read_file", {}),
        ("read_file", {"path": 1}),
        ("write_file", {"path": "x"}),
        ("edit_file", {"path": "x", "old_string": "x"}),
        ("glob", {"pattern": "../*"}),
        ("glob", {"pattern": "C:/outside/*"}),
        ("grep", {"pattern": "x", "glob": 1}),
    ],
)
def test_invalid_parameters_are_denied_before_bypass(tmp_path, monkeypatch, name, args):
    from yincode.permission import Decision, Mode

    engine = engine_at(tmp_path, monkeypatch)
    assert (
        engine.check(Mode.BYPASS, call(name, **args), False, registered=True, allowed=True)[0]
        is Decision.DENY
    )


def test_registered_allowed_and_plan_are_mandatory(tmp_path, monkeypatch):
    from yincode.permission import Decision, Mode
    from yincode.permission.rule import Rule, RuleSet

    engine = engine_at(tmp_path, monkeypatch)
    engine.local = RuleSet(allow=[Rule("Write", "", True)])
    writing = call("write_file", path="x", content="hello")
    for mode, registered, allowed in [
        (Mode.BYPASS, False, True),
        (Mode.BYPASS, True, False),
        (Mode.PLAN, True, True),
    ]:
        assert (
            engine.check(mode, writing, False, registered=registered, allowed=allowed)[0]
            is Decision.DENY
        )
    assert (
        engine.check(Mode.DEFAULT, call("custom"), False, registered=True, allowed=True)[0]
        is Decision.ASK
    )


def test_permanent_rules_are_exact_reloaded_and_preserve_settings(tmp_path, monkeypatch):
    from yincode.permission import Decision, Mode

    engine = engine_at(tmp_path, monkeypatch)
    local = Path(engine.local_path)
    local.parent.mkdir(parents=True)
    local.write_text(
        "default_mode: default\nextra: preserve\npermissions:\n  deny: [Bash(git push)]\n"
    )
    command = call("bash", command=r"echo *(foo) C:\temp")
    engine.persist_local_allow(command)
    engine.persist_local_allow(command)
    saved = yaml.safe_load(local.read_text())
    assert saved["extra"] == "preserve"
    assert len(saved["permissions"]["allow"]) == 1
    again = engine_at(tmp_path, monkeypatch)
    assert (
        again.check(Mode.DEFAULT, command, False, registered=True, allowed=True)[0]
        is Decision.ALLOW
    )
    assert (
        again.check(
            Mode.DEFAULT,
            call("bash", command="echo ANY(foo) C:\\temp"),
            False,
            registered=True,
            allowed=True,
        )[0]
        is Decision.ASK
    )
    assert (
        again.check(
            Mode.DEFAULT, call("bash", command="git push"), False, registered=True, allowed=True
        )[0]
        is Decision.DENY
    )


def test_persistence_failure_does_not_update_memory(tmp_path, monkeypatch):
    import yincode.permission.persist as persist
    from yincode.permission import Decision, Mode

    engine = engine_at(tmp_path, monkeypatch)

    def failed_replace(*args):
        raise OSError("unwritable")

    monkeypatch.setattr(persist.os, "replace", failed_replace)
    command = call("bash", command="git status")
    with pytest.raises(OSError):
        engine.persist_local_allow(command)
    assert (
        engine.check(Mode.DEFAULT, command, False, registered=True, allowed=True)[0] is Decision.ASK
    )
    assert not list((tmp_path / ".yincode").glob("*.tmp"))


def test_unresolvable_root_returns_safe_engine(tmp_path, monkeypatch):
    from yincode.permission import Decision, Mode, new_engine

    engine, error = new_engine(str(tmp_path / "missing"))
    assert error is not None
    assert (
        engine.check(
            Mode.BYPASS, call("bash", command="echo ok"), False, registered=True, allowed=True
        )[0]
        is Decision.DENY
    )


def test_malformed_rule_downgrades_entire_layer(tmp_path, monkeypatch):
    from yincode.permission import Decision, Mode

    local = tmp_path / ".yincode/permissions.local.yaml"
    local.parent.mkdir()
    local.write_text('permissions:\n  allow: [Bash]\n  deny: ["Bash("]\n')
    engine = engine_at(tmp_path, monkeypatch)
    assert (
        engine.check(
            Mode.DEFAULT, call("bash", command="git status"), False, registered=True, allowed=True
        )[0]
        is Decision.ASK
    )
