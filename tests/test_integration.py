"""配置、官方协议适配器和真实界面的完整离线联调。"""

import asyncio
import json
from io import StringIO

import httpx2
import pytest
import yaml
from rich.console import Console
from test_providers import PROTOCOLS, ResponseStream, payload, provider_with_http, tool_payload

from yincode.config import ProviderConfig, load
from yincode.tui import SessionState, YinCodeApp


@pytest.mark.parametrize("protocol", PROTOCOLS)
async def test_real_sdk_ui_failure_then_next_turn_has_valid_history(protocol, tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "providers": [
                    {
                        "name": "local",
                        "protocol": protocol,
                        "api_key": "integration-secret",
                        "model": "test-model",
                        "thinking": True,
                        "base_url": "https://compatible.example"
                        + ("/v1" if protocol == "openai" else ""),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    cfg = load(path).providers[0]
    requests = []
    response_stream = ResponseStream(payload(protocol))

    def respond(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx2.Response(
                401,
                json={
                    "error": {"type": "authentication_error", "message": "bad integration-secret"}
                },
            )
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, stream=response_stream
        )

    http = httpx2.AsyncClient(transport=httpx2.MockTransport(respond))
    provider = provider_with_http(cfg, http)
    app = YinCodeApp([cfg], provider_factory=lambda _: provider, cwd=tmp_path)
    async with app.run_test(size=(60, 20)) as pilot:
        await app.submit("remember 42")
        task = app._stream_task
        assert task is not None
        await asyncio.wait_for(task, 3)
        assert app.state is SessionState.IDLE
        assert app.conv.messages()[-1].role == "assistant"
        assert "integration-secret" not in app.conv.messages()[-1].content
        await app.submit("what number?")
        task = app._stream_task
        assert task is not None
        await asyncio.wait_for(task, 3)
        await pilot.pause()
        assert app.conv.messages()[-1].content == "你好世界"
        history = requests[1]["messages"]
        if protocol == "openai":
            history = history[1:]
        assert [item["role"] for item in history] == ["user", "assistant", "user"]
        assert history[0]["content"] == "remember 42"
        assert history[-1]["content"] == "what number?"
        assert "integration-secret" not in history[1]["content"]
        await app.action_quit()
    assert http.is_closed
    assert response_stream.closed


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("missing", [False, True], ids=["read", "missing"])
async def test_real_sdk_tool_result_flows_from_project_file_to_followup_and_replay(
    protocol, missing, tmp_path
):
    if not missing:
        (tmp_path / "sample.txt").write_text("project fixture", encoding="utf-8")
    cfg = ProviderConfig(
        "local", protocol, "integration-secret", "test-model", "https://compatible.example", True
    )
    requests = []
    streams = [
        ResponseStream(tool_payload(protocol, args='{"path":"sample.txt"}')),
        ResponseStream(payload(protocol)),
    ]

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=streams[len(requests) - 1],
        )

    http = httpx2.AsyncClient(transport=httpx2.MockTransport(respond))
    provider = provider_with_http(cfg, http)
    app = YinCodeApp([cfg], provider_factory=lambda _: provider, cwd=tmp_path)
    async with app.run_test(size=(70, 24)) as pilot:
        await app.submit("读项目文件")
        task = app._stream_task
        assert task is not None
        await asyncio.wait_for(task, 3)
        await pilot.pause()
        assert app.state is SessionState.IDLE
        messages = app.conv.messages()
        assert [message.role for message in messages] == ["user", "assistant", "tool", "assistant"]
        result = messages[2].tool_results[0]
        assert result.tool_call_id == "call-a" and result.is_error is missing
        assert messages[-1].content == "你好世界"
        assert len(requests) == 2 and all(len(body["tools"]) == 6 for body in requests)
        followup = requests[1]["messages"]
        if protocol == "openai":
            assert followup[-1] == {
                "role": "tool",
                "tool_call_id": "call-a",
                "content": result.content,
            }
            assert followup[-2]["tool_calls"][0]["id"] == "call-a"
        else:
            assert followup[-1]["content"] == [
                {
                    "type": "tool_result",
                    "tool_use_id": "call-a",
                    "content": result.content,
                    "is_error": missing,
                }
            ]
            assert "thinking" not in requests[0] and "thinking" not in requests[1]
        if not missing:
            assert "project fixture" in result.content
        if missing:
            assert any("red" in str(getattr(block, "style", "")) for block in app.transcript)
    buffer = StringIO()
    console = Console(file=buffer, color_system=None, width=80)
    for block in app.transcript:
        console.print(block)
    output = buffer.getvalue()
    assert output.index("正在读取") < output.index("read_file(") < output.index("你好世界")
    assert "integration-secret" not in output
    assert output.count("耗时：") == 1
    assert http.is_closed and all(stream.closed for stream in streams)
