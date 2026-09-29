import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from yincode.conversation import Conversation
from yincode.llm import Message, StreamEvent, ToolCall, ToolDefinition


class ScriptProvider:
    name = "script"
    model = "test"

    def __init__(self, scripts: list[list[StreamEvent]]) -> None:
        self.scripts = scripts
        self.requests: list[list[Message]] = []
        self.definitions: list[list[ToolDefinition]] = []
        self.closed = 0

    async def stream(
        self, msgs: list[Message], tools: list[ToolDefinition]
    ) -> AsyncIterator[StreamEvent]:
        index = len(self.requests)
        self.requests.append(msgs)
        self.definitions.append(tools)
        try:
            for event in self.scripts[index]:
                yield event
        finally:
            self.closed += 1

    async def aclose(self) -> None:
        pass


def conversation() -> Conversation:
    conv = Conversation()
    conv.add_user("read the file")
    return conv


async def test_read_result_is_carried_into_continuation(tmp_path: Path):
    from yincode.agent import Agent, Phase
    from yincode.tool import new_default_registry

    (tmp_path / "sample.txt").write_text("unique fixture", encoding="utf-8")
    calls = [ToolCall("c1", "read_file", '{"path":"sample.txt"}')]
    provider = ScriptProvider(
        [
            [StreamEvent(text="reading"), StreamEvent(tool_calls=calls), StreamEvent(done=True)],
            [StreamEvent(text="fixture summary"), StreamEvent(done=True)],
        ]
    )
    conv = conversation()
    events = [event async for event in Agent(provider, new_default_registry(tmp_path)).run(conv)]
    assert [e.tool.phase for e in events if e.tool] == [Phase.START, Phase.END]
    assert [e.tool.tool_call_id for e in events if e.tool] == ["c1", "c1"]
    assert provider.requests[1][-1].tool_results[0].tool_call_id == "c1"
    assert "unique fixture" in provider.requests[1][-1].tool_results[0].content
    assert [m.role for m in conv.messages()] == ["user", "assistant", "tool", "assistant"]
    assert conv.messages()[-1].content == "fixture summary"
    assert len(provider.definitions[0]) == len(provider.definitions[1]) == 6
    assert provider.closed == 2
    assert events[-1].done


async def test_second_tool_batch_is_not_executed_or_saved(tmp_path: Path):
    from yincode.agent import Agent
    from yincode.tool import new_default_registry

    first = ToolCall("c1", "write_file", '{"path":"one.txt","content":"first"}')
    second = ToolCall("c2", "write_file", '{"path":"two.txt","content":"second"}')
    provider = ScriptProvider(
        [
            [StreamEvent(tool_calls=[first]), StreamEvent(done=True)],
            [StreamEvent(tool_calls=[second]), StreamEvent(done=True)],
        ]
    )
    conv = conversation()
    events = [e async for e in Agent(provider, new_default_registry(tmp_path)).run(conv)]
    assert (tmp_path / "one.txt").read_text() == "first"
    assert not (tmp_path / "two.txt").exists()
    assert "上限" in conv.messages()[-1].content
    assert "上限" in "".join(e.text for e in events)
    assert all(c.id != "c2" for m in conv.messages() for c in m.tool_calls)
    assert len(provider.requests) == 2


@pytest.mark.parametrize("continuation", [False, True])
async def test_stream_error_keeps_legal_tail_and_redacts(tmp_path: Path, continuation: bool):
    from yincode.agent import Agent
    from yincode.tool import new_default_registry

    scripts = [[StreamEvent(err=RuntimeError("test-secret"))]]
    if continuation:
        scripts.insert(
            0,
            [
                StreamEvent(tool_calls=[ToolCall("c1", "read_file", '{"path":"absent"}')]),
                StreamEvent(done=True),
            ],
        )
    conv = conversation()
    agent = Agent(
        ScriptProvider(scripts),
        new_default_registry(tmp_path),
        redactor=lambda s: s.replace("test-secret", "[redacted]"),
    )
    events = [e async for e in agent.run(conv)]
    assert events[-1].err is not None
    assert "test-secret" not in str(events[-1].err)
    assert conv.messages()[-1].role == "assistant"
    assert "失败" in conv.messages()[-1].content
    assert "test-secret" not in repr(conv.messages())


@pytest.mark.parametrize("raw", ["[]", '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}'])
async def test_invalid_call_never_enters_protocol_history(raw: str):
    from yincode.agent import Agent
    from yincode.tool import Registry

    provider = ScriptProvider(
        [[StreamEvent(tool_calls=[ToolCall("c1", "read_file", raw)]), StreamEvent(done=True)]]
    )
    conv = conversation()
    events = [e async for e in Agent(provider, Registry()).run(conv)]
    assert events[-1].err
    assert not any(m.tool_calls for m in conv.messages())
    assert conv.messages()[-1].role == "assistant"


async def test_cancellation_pairs_whole_batch_and_waits_for_tool_cleanup():
    from yincode.agent import Agent
    from yincode.tool import Registry, Result

    class BlockingTool:
        name = "block"
        description = "blocks"
        parameters = {"type": "object", "properties": {}}

        def __init__(self):
            self.started = asyncio.Event()
            self.cleaned = asyncio.Event()

        async def execute(self, args: str) -> Result:
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.cleaned.set()
            return Result("unreachable")

    tool = BlockingTool()
    registry = Registry()
    registry.register(tool)
    calls = [ToolCall("c1", "block", "{}"), ToolCall("c2", "block", "{}")]
    provider = ScriptProvider([[StreamEvent(tool_calls=calls), StreamEvent(done=True)]])
    conv = conversation()

    async def consume():
        async for _ in Agent(provider, registry).run(conv):
            pass

    task = asyncio.create_task(consume())
    await asyncio.wait_for(tool.started.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert tool.cleaned.is_set()
    results = conv.messages()[2].tool_results
    assert [r.tool_call_id for r in results] == ["c1", "c2"]
    assert all(r.is_error for r in results)
    assert conv.messages()[-1].role == "assistant"


async def test_aclose_at_start_event_pairs_batch_without_executing(tmp_path: Path):
    from yincode.agent import Agent, Phase
    from yincode.tool import new_default_registry

    call = ToolCall("c1", "write_file", '{"path":"never.txt","content":"x"}')
    provider = ScriptProvider([[StreamEvent(tool_calls=[call]), StreamEvent(done=True)]])
    conv = conversation()
    iterator = Agent(provider, new_default_registry(tmp_path)).run(conv)
    event = await anext(iterator)
    assert event.tool.phase is Phase.START
    await iterator.aclose()
    assert not (tmp_path / "never.txt").exists()
    assert conv.messages()[2].tool_results[0].is_error
    assert conv.messages()[-1].role == "assistant"


async def test_empty_reply_and_eof_are_handled_without_empty_assistant(tmp_path: Path):
    from yincode.agent import Agent
    from yincode.tool import new_default_registry

    for script in ([StreamEvent(done=True)], [StreamEvent(text="partial")]):
        conv = conversation()
        events = [
            e
            async for e in Agent(ScriptProvider([script]), new_default_registry(tmp_path)).run(conv)
        ]
        assert conv.messages()[-1].role == "assistant"
        assert conv.messages()[-1].content
        assert events[-1].done or events[-1].err


async def test_secret_split_across_text_chunks_is_redacted_before_emission(tmp_path: Path):
    from yincode.agent import Agent
    from yincode.tool import Registry

    provider = ScriptProvider(
        [[StreamEvent(text="reply key-"), StreamEvent(text="123 end"), StreamEvent(done=True)]]
    )
    conv = conversation()
    agent = Agent(
        provider,
        Registry(),
        redactor=lambda s: s.replace("key-123", "[redacted]"),
        secrets=("key-123",),
    )
    events = [e async for e in agent.run(conv)]
    assert "".join(e.text for e in events) == "reply [redacted] end"
    assert conv.messages()[-1].content == "reply [redacted] end"


async def test_empty_continuation_produces_visible_nonempty_tail():
    from yincode.agent import Agent
    from yincode.tool import Registry

    provider = ScriptProvider(
        [
            [StreamEvent(tool_calls=[ToolCall("c1", "missing", "{}")]), StreamEvent(done=True)],
            [StreamEvent(done=True)],
        ]
    )
    conv = conversation()
    events = [e async for e in Agent(provider, Registry()).run(conv)]
    assert conv.messages()[2].tool_results[0].is_error
    assert conv.messages()[-1].content
    assert conv.messages()[-1].content == "".join(e.text for e in events)
    assert events[-1].done
