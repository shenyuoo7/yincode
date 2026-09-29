"""通过官方 SDK 与可控 SSE 响应验证适配器行为。"""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Literal

import anthropic
import httpx2
import openai
import pytest

from yincode.config import ProviderConfig
from yincode.llm import Message, Provider, StreamEvent, new_provider
from yincode.prompt import SYSTEM_PROMPT

ProtocolName = Literal["anthropic", "openai"]
PROTOCOLS: tuple[ProtocolName, ...] = ("anthropic", "openai")
TEST_KEY = "test-provider-secret"
HISTORY = [Message("user", "我叫小明"), Message("assistant", "你好"), Message("user", "我叫什么？")]


class ResponseStream(httpx2.AsyncByteStream):
    """保留真实 HTTP 流语义，只控制数据、等待和失败。"""

    def __init__(self, chunks: list[bytes], *, block: bool = False, fail: bool = False) -> None:
        self.chunks = chunks
        self.block = block
        self.fail = fail
        self.closed = False
        self.waiting = asyncio.Event()
        self.released = asyncio.Event()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for index, chunk in enumerate(self.chunks):
            if index == 1:
                if self.fail:
                    raise RuntimeError(f"流断开：{TEST_KEY}")
                if self.block:
                    self.waiting.set()
                    await self.released.wait()
            yield chunk

    async def aclose(self) -> None:
        self.closed = True
        self.released.set()


def sse(data: dict[str, object], event: str | None = None) -> bytes:
    prefix = f"event: {event}\n" if event else ""
    return (prefix + "data: " + json.dumps(data, ensure_ascii=False) + "\n\n").encode()


def payload(protocol: ProtocolName) -> list[bytes]:
    if protocol == "openai":

        def chunk(choices: list[dict[str, object]]) -> bytes:
            return sse(
                {
                    "id": "chat-test",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "test-model",
                    "choices": choices,
                }
            )

        return [
            chunk([])
            + chunk(
                [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "reasoning_content": "隐藏思考"},
                        "finish_reason": None,
                    }
                ]
            )
            + chunk([{"index": 0, "delta": {"content": "你好"}, "finish_reason": None}]),
            chunk([{"index": 0, "delta": {"content": "世界"}, "finish_reason": None}])
            + chunk([{"index": 0, "delta": {}, "finish_reason": "stop"}])
            + b"data: [DONE]\n\n",
        ]

    def event(kind: str, **data: object) -> bytes:
        return sse({"type": kind, **data}, kind)

    return [
        event(
            "message_start",
            message={
                "id": "msg-test",
                "type": "message",
                "role": "assistant",
                "model": "test-model",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 4, "output_tokens": 0},
            },
        )
        + event(
            "content_block_start",
            index=0,
            content_block={"type": "thinking", "thinking": "", "signature": ""},
        )
        + event(
            "content_block_delta", index=0, delta={"type": "thinking_delta", "thinking": "隐藏思考"}
        )
        + event(
            "content_block_delta",
            index=0,
            delta={"type": "signature_delta", "signature": "test-signature"},
        )
        + event("content_block_stop", index=0)
        + event("content_block_start", index=1, content_block={"type": "text", "text": ""})
        + event("content_block_delta", index=1, delta={"type": "text_delta", "text": "你好"}),
        event("content_block_delta", index=1, delta={"type": "text_delta", "text": "世界"})
        + event("content_block_stop", index=1)
        + event(
            "message_delta",
            delta={"stop_reason": "end_turn", "stop_sequence": None},
            usage={"output_tokens": 4},
        )
        + event("message_stop"),
    ]


def config(protocol: ProtocolName, *, thinking: bool = False) -> ProviderConfig:
    return ProviderConfig(
        "测试服务",
        protocol,
        TEST_KEY,
        "test-model",
        "https://compatible.example/"
        if protocol == "anthropic"
        else "https://compatible.example/v1/",
        thinking,
    )


def provider_with_http(cfg: ProviderConfig, http: httpx2.AsyncClient) -> Provider:
    if cfg.protocol == "anthropic":
        from yincode.llm.anthropic_provider import AnthropicProvider

        anthropic_client = anthropic.AsyncAnthropic(
            api_key=cfg.api_key, base_url=cfg.base_url, http_client=http, max_retries=0
        )
        return AnthropicProvider(cfg, client=anthropic_client)
    from yincode.llm.openai_provider import OpenAIProvider

    openai_client = openai.AsyncOpenAI(
        api_key=cfg.api_key, base_url=cfg.base_url, http_client=http, max_retries=0
    )
    return OpenAIProvider(cfg, client=openai_client)


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("thinking", [False, True])
async def test_request_contains_history_and_streams_only_body(
    protocol: ProtocolName, thinking: bool
) -> None:
    requests: list[httpx2.Request] = []
    response = ResponseStream(payload(protocol))

    def handle(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=response)

    http = httpx2.AsyncClient(transport=httpx2.MockTransport(handle))
    provider = provider_with_http(config(protocol, thinking=thinking), http)
    try:
        events = [event async for event in provider.stream(HISTORY)]
        assert events == [
            StreamEvent(text="你好"),
            StreamEvent(text="世界"),
            StreamEvent(done=True),
        ]
        assert response.closed
        assert len(requests) == 1
        body = json.loads(requests[0].content)
        assert body["model"] == "test-model"
        assert body["stream"] is True
        history = [
            {"role": "user", "content": "我叫小明"},
            {"role": "assistant", "content": "你好"},
            {"role": "user", "content": "我叫什么？"},
        ]
        if protocol == "anthropic":
            assert str(requests[0].url) == "https://compatible.example/v1/messages"
            assert body["system"] == SYSTEM_PROMPT
            assert body["messages"] == history
            assert body["max_tokens"] == 4096
            if thinking:
                assert body["thinking"] == {"type": "enabled", "budget_tokens": 2048}
            else:
                assert "thinking" not in body
        else:
            assert str(requests[0].url) == "https://compatible.example/v1/chat/completions"
            assert body["messages"] == [{"role": "system", "content": SYSTEM_PROMPT}, *history]
            assert "thinking" not in body
    finally:
        await provider.aclose()
    assert http.is_closed


@pytest.mark.parametrize("protocol", PROTOCOLS)
async def test_http_error_is_detached_redacted_and_next_request_can_succeed(
    protocol: ProtocolName,
) -> None:
    requests = 0

    def handle(request: httpx2.Request) -> httpx2.Response:
        nonlocal requests
        requests += 1
        if requests == 1:
            return httpx2.Response(
                429, json={"error": {"type": "rate_limit_error", "message": f"暂不可用 {TEST_KEY}"}}
            )
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=ResponseStream(payload(protocol)),
        )

    http = httpx2.AsyncClient(transport=httpx2.MockTransport(handle))
    provider = provider_with_http(config(protocol), http)
    try:
        events = [event async for event in provider.stream(HISTORY)]
        assert len(events) == 1
        error = events[0].err
        assert error is not None
        assert "[REDACTED]" in str(error)
        assert TEST_KEY not in str(error) + repr(error)
        assert error.__cause__ is None
        assert error.__context__ is None
        assert error.__traceback__ is None
        assert requests == 1
        assert [event async for event in provider.stream(HISTORY)][-1] == StreamEvent(done=True)
        assert requests == 2
    finally:
        await provider.aclose()


@pytest.mark.parametrize("protocol", PROTOCOLS)
async def test_stream_error_closes_response_and_emits_one_redacted_error(
    protocol: ProtocolName,
) -> None:
    response = ResponseStream(payload(protocol), fail=True)
    http = httpx2.AsyncClient(
        transport=httpx2.MockTransport(
            lambda request: httpx2.Response(
                200, headers={"content-type": "text/event-stream"}, stream=response
            )
        )
    )
    provider = provider_with_http(config(protocol), http)
    try:
        events = [event async for event in provider.stream(HISTORY)]
        assert events[0] == StreamEvent(text="你好")
        assert len(events) == 2
        assert events[1].err is not None
        assert TEST_KEY not in str(events[1].err) + repr(events[1].err)
        assert "[REDACTED]" in str(events[1].err)
        assert not any(event.done for event in events)
        assert response.closed
    finally:
        await provider.aclose()


@pytest.mark.parametrize("protocol", PROTOCOLS)
async def test_cancellation_propagates_without_terminal_event_and_closes_response(
    protocol: ProtocolName,
) -> None:
    response = ResponseStream(payload(protocol), block=True)
    http = httpx2.AsyncClient(
        transport=httpx2.MockTransport(
            lambda request: httpx2.Response(
                200, headers={"content-type": "text/event-stream"}, stream=response
            )
        )
    )
    provider = provider_with_http(config(protocol), http)
    events: list[StreamEvent] = []

    async def consume() -> None:
        async for event in provider.stream(HISTORY):
            events.append(event)

    task = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(response.waiting.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert events == [StreamEvent(text="你好")]
        assert response.closed
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await provider.aclose()
    assert http.is_closed


@pytest.mark.parametrize("protocol", PROTOCOLS)
async def test_closing_iterator_early_closes_response(protocol: ProtocolName) -> None:
    response = ResponseStream(payload(protocol), block=True)
    http = httpx2.AsyncClient(
        transport=httpx2.MockTransport(
            lambda request: httpx2.Response(
                200, headers={"content-type": "text/event-stream"}, stream=response
            )
        )
    )
    provider = provider_with_http(config(protocol), http)
    iterator = provider.stream(HISTORY)
    try:
        assert await anext(iterator) == StreamEvent(text="你好")
        await iterator.aclose()  # type: ignore[attr-defined]
        assert response.closed
    finally:
        await provider.aclose()


@pytest.mark.parametrize("protocol", PROTOCOLS)
async def test_provider_close_releases_active_response_and_client(protocol: ProtocolName) -> None:
    response = ResponseStream(payload(protocol), block=True)
    http = httpx2.AsyncClient(
        transport=httpx2.MockTransport(
            lambda request: httpx2.Response(
                200, headers={"content-type": "text/event-stream"}, stream=response
            )
        )
    )
    provider = provider_with_http(config(protocol), http)
    iterator = provider.stream(HISTORY)
    try:
        assert await anext(iterator) == StreamEvent(text="你好")
        await provider.aclose()
        assert response.closed
        assert http.is_closed
        await provider.aclose()
    finally:
        await iterator.aclose()  # type: ignore[attr-defined]
        await provider.aclose()


@pytest.mark.parametrize("protocol", PROTOCOLS)
async def test_default_client_uses_configured_endpoint_without_automatic_retries(
    protocol: ProtocolName,
) -> None:
    provider = new_provider(config(protocol))
    try:
        assert provider.name == "测试服务"
        assert provider.model == "test-model"
        client = provider._client  # type: ignore[attr-defined]
        assert str(client.base_url) == (
            "https://compatible.example/"
            if protocol == "anthropic"
            else "https://compatible.example/v1/"
        )
        assert client.max_retries == 0
    finally:
        await provider.aclose()
    assert client.is_closed()
