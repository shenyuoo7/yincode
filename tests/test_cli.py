import asyncio
import importlib.util
import sys
from types import SimpleNamespace

import pytest
from rich.text import Text

from yincode import cli


def test_package_is_importable():
    assert importlib.util.find_spec("yincode") is not None


def test_missing_config_exits_with_readable_message(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as error:
        cli.main([])
    assert error.value.code == 1
    output = capsys.readouterr()
    assert "配置文件不存在" in output.err
    assert "Traceback" not in output.err


def test_invalid_config_exits_without_echoing_yaml_key(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".yincode").mkdir()
    (tmp_path / ".yincode/config.yaml").write_text(
        "providers: [api_key: private-value", encoding="utf-8"
    )
    with pytest.raises(SystemExit) as error:
        cli.main([])
    assert error.value.code == 1
    output = capsys.readouterr()
    assert "YAML" in output.err
    assert "private-value" not in output.err


def test_version_does_not_require_config(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as error:
        cli.main(["--version"])
    assert error.value.code == 0
    assert "yincode 0.1.0" in capsys.readouterr().out


def configure_project(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".yincode").mkdir()
    (tmp_path / ".yincode/config.yaml").write_text(
        "providers:\n  - name: local\n    protocol: openai\n"
        "    api_key: private-value\n    model: test-model\n",
        encoding="utf-8",
    )


def test_completed_transcript_is_replayed_once_after_ui(monkeypatch, tmp_path, capsys):
    configure_project(monkeypatch, tmp_path)

    class CompletedApp:
        def __init__(self, providers, *, cwd, registry):
            self.transcript = [Text("YIN snake"), Text("remember 42"), Text("okay 42")]

        def run(self):
            return None

    monkeypatch.setitem(sys.modules, "yincode.tui", SimpleNamespace(YinCodeApp=CompletedApp))
    cli.main([])
    output = capsys.readouterr()
    assert output.out.count("YIN snake") == 1
    assert output.out.count("remember 42") == 1
    assert output.out.count("okay 42") == 1


def test_constructor_failure_is_redacted_without_traceback(monkeypatch, tmp_path, capsys):
    configure_project(monkeypatch, tmp_path)

    class BrokenApp:
        def __init__(self, providers, *, cwd, registry):
            raise RuntimeError("cannot initialize private-value")

    monkeypatch.setitem(sys.modules, "yincode.tui", SimpleNamespace(YinCodeApp=BrokenApp))
    with pytest.raises(SystemExit) as error:
        cli.main([])
    assert error.value.code == 1
    output = capsys.readouterr()
    assert "private-value" not in output.err
    assert "[REDACTED]" in output.err
    assert "Traceback" not in output.err


def test_cli_tool_registry_reads_relative_to_the_project(monkeypatch, tmp_path, capsys):
    configure_project(monkeypatch, tmp_path)
    (tmp_path / "sample.txt").write_text("project fixture", encoding="utf-8")

    class ToolApp:
        def __init__(self, providers, *, cwd, registry):
            self.registry = registry
            self.transcript = []

        def run(self):
            result = asyncio.run(self.registry.execute("read_file", '{"path":"sample.txt"}'))
            self.transcript.append(Text(result.content))

    monkeypatch.setitem(sys.modules, "yincode.tui", SimpleNamespace(YinCodeApp=ToolApp))
    cli.main([])
    output = capsys.readouterr()
    assert "project fixture" in output.out
    assert "private-value" not in output.out + output.err
