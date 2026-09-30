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


@pytest.fixture
def smoke_module():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "examples" / "smoke.py"
    assert path.is_file(), "缺少可运行的纯文本 smoke 示例"
    spec = importlib.util.spec_from_file_location("yincode_smoke", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("plan", [False, True])
async def test_smoke_two_turns_print_cache_usage_and_close(
    monkeypatch, tmp_path, capsys, smoke_module, plan
):
    from yincode.config import Config, ProviderConfig
    from yincode.llm import StreamEvent, Usage

    class Provider:
        name = "fixture"
        model = "fixture-model"
        closed = False
        requests = []

        async def stream(self, req):
            self.requests.append(req)
            yield StreamEvent(text="private-")
            yield StreamEvent(text="value reply")
            yield StreamEvent(usage=Usage(10, 2, 3, 4))
            yield StreamEvent(done=True)

        async def aclose(self):
            self.closed = True

    provider = Provider()
    monkeypatch.setattr(smoke_module, "new_provider", lambda _: provider)
    config = Config([ProviderConfig("fixture", "openai", "private-value", "fixture-model")])
    await smoke_module.run(config, tmp_path, ["first", "second"], plan)
    output = capsys.readouterr().out
    assert "private-value" not in output and "[REDACTED]" in output
    assert "input=10 output=2 cache_write=3 cache_read=4" in output
    assert provider.closed and len(provider.requests) == 2
    assert all(len(req.tools) == (3 if plan else 6) for req in provider.requests)
    assert all(bool(req.reminder) is plan for req in provider.requests)
    assert [msg.role for msg in provider.requests[1].messages] == ["user", "assistant", "user"]
    assert str(tmp_path) in provider.requests[0].system.environment


async def test_smoke_cancellation_propagates_and_closes(monkeypatch, tmp_path, smoke_module):
    from yincode.config import Config, ProviderConfig
    from yincode.llm import StreamEvent

    class Provider:
        name = "fixture"
        model = "fixture-model"
        closed = False
        stream_closed = False

        async def stream(self, req):
            try:
                raise asyncio.CancelledError
                yield StreamEvent(done=True)
            finally:
                self.stream_closed = True

        async def aclose(self):
            self.closed = True

    provider = Provider()
    monkeypatch.setattr(smoke_module, "new_provider", lambda _: provider)
    config = Config([ProviderConfig("fixture", "openai", "private-value", "fixture-model")])
    with pytest.raises(asyncio.CancelledError):
        await smoke_module.run(config, tmp_path, ["first"], False)
    assert provider.closed and provider.stream_closed


def test_smoke_main_loads_config_and_redacts_startup_error(
    monkeypatch, tmp_path, capsys, smoke_module
):
    configure_project(monkeypatch, tmp_path)

    def broken_provider(config):
        raise RuntimeError("cannot initialize " + config.api_key)

    monkeypatch.setattr(smoke_module, "new_provider", broken_provider)
    assert smoke_module.main(["--cwd", str(tmp_path)]) == 1
    output = capsys.readouterr()
    assert "private-value" not in output.out + output.err
    assert "[REDACTED]" in output.err and "Traceback" not in output.err
