"""配置、官方协议适配器和真实界面的完整离线联调。"""

import asyncio
import json

import httpx2
import pytest
import yaml
from test_providers import PROTOCOLS, ResponseStream, payload, provider_with_http

from yincode.config import load
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
