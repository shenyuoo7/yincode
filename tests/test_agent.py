import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from yincode.conversation import Conversation
from yincode.llm import Message, Request, StreamEvent, ToolCall, ToolDefinition, Usage
from yincode.permission import Mode


class ScriptProvider:
    name = "script"
    model = "test"

    def __init__(self, scripts: list[list[StreamEvent]]) -> None:
        self.scripts = scripts
        self.requests: list[list[Message]] = []
        self.definitions: list[list[ToolDefinition]] = []
        self.assembled: list[Request] = []
        self.closed = 0

    async def stream(self, req: Request) -> AsyncIterator[StreamEvent]:
        index = len(self.requests)
        self.requests.append(req.messages)
        self.definitions.append(req.tools)
        self.assembled.append(req)
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
    events = [
        event
        async for event in Agent(provider, new_default_registry(tmp_path)).run(conv, Mode.BYPASS)
    ]
    assert [e.tool.phase for e in events if e.tool] == [Phase.START, Phase.END]
    assert [e.tool.tool_call_id for e in events if e.tool] == ["c1", "c1"]
    assert provider.requests[1][-1].tool_results[0].tool_call_id == "c1"
    assert "unique fixture" in provider.requests[1][-1].tool_results[0].content
    assert [m.role for m in conv.messages()] == ["user", "assistant", "tool", "assistant"]
    assert conv.messages()[-1].content == "fixture summary"
    assert len(provider.definitions[0]) == len(provider.definitions[1]) == 6
    assert provider.closed == 2
    assert events[-1].done


async def test_second_tool_batch_is_executed_without_user_prompt(tmp_path: Path):
    from yincode.agent import Agent
    from yincode.tool import new_default_registry

    first = ToolCall("c1", "write_file", '{"path":"one.txt","content":"first"}')
    second = ToolCall("c2", "write_file", '{"path":"two.txt","content":"second"}')
    provider = ScriptProvider(
        [
            [StreamEvent(tool_calls=[first]), StreamEvent(done=True)],
            [StreamEvent(tool_calls=[second]), StreamEvent(done=True)],
            [StreamEvent(text="done"), StreamEvent(done=True)],
        ]
    )
    conv = conversation()
    events = [
        e async for e in Agent(provider, new_default_registry(tmp_path)).run(conv, Mode.BYPASS)
    ]
    assert (tmp_path / "one.txt").read_text() == "first"
    assert (tmp_path / "two.txt").read_text() == "second"
    assert conv.messages()[-1].content == "done"
    assert [c.id for m in conv.messages() for c in m.tool_calls] == ["c1", "c2"]
    assert len(provider.requests) == 3
    assert [e.iter for e in events if e.iter] == [1, 2, 3]


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
    events = [e async for e in agent.run(conv, Mode.BYPASS)]
    assert events[-1].err is not None
    assert "test-secret" not in str(events[-1].err)
    assert conv.messages()[-1].role == "assistant"
    assert "失败" in conv.messages()[-1].content
    assert "test-secret" not in repr(conv.messages())


@pytest.mark.parametrize("raw", ["[]", '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}'])
async def test_invalid_call_has_safe_history_and_paired_denial(raw: str):
    from yincode.agent import Agent
    from yincode.tool import Registry

    provider = ScriptProvider(
        [
            [StreamEvent(tool_calls=[ToolCall("c1", "read_file", raw)]), StreamEvent(done=True)],
            [StreamEvent(text="已调整"), StreamEvent(done=True)],
        ]
    )
    conv = conversation()
    events = [e async for e in Agent(provider, Registry()).run(conv, Mode.BYPASS)]
    assert events[-1].done
    assert conv.messages()[1].tool_calls[0].input == "{}"
    assert conv.messages()[2].tool_results[0].is_error
    assert conv.messages()[-1].role == "assistant"


async def test_cancellation_pairs_whole_batch_and_waits_for_tool_cleanup():
    from yincode.agent import Agent
    from yincode.tool import Registry, Result

    class BlockingTool:
        name = "block"
        description = "blocks"
        parameters = {"type": "object", "properties": {}}
        read_only = False

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
        async for _ in Agent(provider, registry).run(conv, Mode.BYPASS):
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
    iterator = Agent(provider, new_default_registry(tmp_path)).run(conv, Mode.BYPASS)
    event = await anext(iterator)
    assert event.iter == 1
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
            async for e in Agent(ScriptProvider([script]), new_default_registry(tmp_path)).run(
                conv, Mode.BYPASS
            )
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
    events = [e async for e in agent.run(conv, Mode.BYPASS)]
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
    events = [e async for e in Agent(provider, Registry()).run(conv, Mode.BYPASS)]
    assert conv.messages()[2].tool_results[0].is_error
    assert conv.messages()[-1].content
    assert conv.messages()[-1].content == "".join(e.text for e in events)
    assert events[-1].done


async def test_plan_mode_filters_and_blocks_malicious_write(tmp_path: Path):
    from yincode.agent import Agent, Mode
    from yincode.prompt import plan_reminder
    from yincode.tool import new_default_registry

    call = ToolCall("evil", "write_file", '{"path":"plan.txt","content":"bad"}')
    provider = ScriptProvider(
        [
            [StreamEvent(tool_calls=[call]), StreamEvent(done=True)],
            [StreamEvent(text="plan only"), StreamEvent(done=True)],
        ]
    )
    conv = conversation()
    events = [e async for e in Agent(provider, new_default_registry(tmp_path)).run(conv, Mode.PLAN)]
    assert [req.reminder for req in provider.assembled] == [
        plan_reminder(True),
        plan_reminder(False),
    ]
    assert all(
        {tool.name for tool in defs} == {"read_file", "glob", "grep"}
        for defs in provider.definitions
    )
    assert not (tmp_path / "plan.txt").exists()
    assert conv.messages()[2].tool_results[0].is_error
    assert events[-1].done


async def test_usage_events_and_unknown_tool_limit():
    from yincode.agent import MAX_UNKNOWN_RUN, NOTICE_UNKNOWN_TOOLS, Agent
    from yincode.tool import Registry

    provider = ScriptProvider(
        [
            [
                StreamEvent(tool_calls=[ToolCall(f"c{i}", "missing", "{}")]),
                StreamEvent(usage=Usage(i + 1, 2)),
                StreamEvent(done=True),
            ]
            for i in range(MAX_UNKNOWN_RUN)
        ]
    )
    conv = conversation()
    events = [e async for e in Agent(provider, Registry()).run(conv, Mode.BYPASS)]
    assert len(provider.requests) == MAX_UNKNOWN_RUN
    assert [e.usage for e in events if e.usage] == [Usage(i + 1, 2) for i in range(MAX_UNKNOWN_RUN)]
    assert [e.notice for e in events if e.notice] == [NOTICE_UNKNOWN_TOOLS]
    assert conv.last_role() == "assistant"
    assert all(m.tool_results for m in conv.messages() if m.role == "tool")


async def test_cancel_while_provider_is_silent_closes_stream():
    from yincode.agent import NOTICE_CANCELLED, Agent
    from yincode.tool import Registry

    class SilentProvider(ScriptProvider):
        def __init__(self):
            super().__init__([])
            self.started = asyncio.Event()
            self.closed_stream = asyncio.Event()

        async def stream(self, req):
            self.started.set()
            try:
                await asyncio.Event().wait()
                yield StreamEvent(done=True)
            finally:
                self.closed_stream.set()

    provider = SilentProvider()
    conv = conversation()
    cancel = asyncio.Event()
    task = asyncio.create_task(
        collect(Agent(provider, Registry()).run(conv, Mode.BYPASS, cancel=cancel))
    )
    await asyncio.wait_for(provider.started.wait(), 2)
    cancel.set()
    events = await asyncio.wait_for(task, 2)
    assert provider.closed_stream.is_set()
    assert [e.notice for e in events if e.notice] == [NOTICE_CANCELLED]
    assert events[-1].done
    assert conv.last_role() == "assistant"


async def collect(iterator):
    return [event async for event in iterator]


async def test_consecutive_read_only_calls_overlap_before_write():
    from yincode.agent import Agent, Phase
    from yincode.tool import Registry, Result

    class Probe:
        description = "probe"
        parameters = {"type": "object", "properties": {}}

        def __init__(self, name: str, read_only: bool):
            self.name = name
            self.read_only = read_only
            self.active = 0
            self.peak = 0
            self.completed = 0
            self.write_after_reads = False
            self.both_started = asyncio.Event()

        async def execute(self, args: str) -> Result:
            if not self.read_only:
                self.write_after_reads = reader.completed == 2
                return Result("written")
            self.active += 1
            self.peak = max(self.peak, self.active)
            if self.active == 2:
                self.both_started.set()
            await asyncio.wait_for(self.both_started.wait(), 2)
            self.active -= 1
            self.completed += 1
            return Result(args)

    reader = Probe("reader", True)
    writer = Probe("writer", False)
    registry = Registry()
    registry.register(reader)
    registry.register(writer)
    calls = [
        ToolCall("r1", "reader", "{}"),
        ToolCall("r2", "reader", "{}"),
        ToolCall("w1", "writer", "{}"),
    ]
    provider = ScriptProvider(
        [
            [StreamEvent(tool_calls=calls), StreamEvent(done=True)],
            [StreamEvent(text="done"), StreamEvent(done=True)],
        ]
    )
    conv = conversation()
    events = await collect(Agent(provider, registry).run(conv, Mode.BYPASS))
    assert reader.peak == 2
    assert writer.write_after_reads
    assert [result.tool_call_id for result in conv.messages()[2].tool_results] == ["r1", "r2", "w1"]
    assert [event.tool.phase for event in events if event.tool] == [Phase.START] * 3 + [
        Phase.END
    ] * 3


async def test_max_iterations_and_unknown_counter_reset():
    from yincode.agent import MAX_ITERATIONS, NOTICE_MAX_ITER, Agent
    from yincode.tool import Registry

    provider = ScriptProvider(
        [
            [StreamEvent(tool_calls=[ToolCall(f"c{i}", "missing", "{}")]), StreamEvent(done=True)]
            for i in range(MAX_ITERATIONS)
        ]
    )
    # 一个已注册工具夹在未知工具之间，使未知计数归零，但总轮数仍应受上限控制。
    from yincode.tool import Result

    class Known:
        name = "known"
        description = "known"
        parameters = {"type": "object", "properties": {}}
        read_only = True

        async def execute(self, args: str) -> Result:
            return Result("ok")

    registry = Registry()
    registry.register(Known())
    for index in range(0, MAX_ITERATIONS, 3):
        provider.scripts[index] = [
            StreamEvent(tool_calls=[ToolCall(f"c{index}", "known", "{}")]),
            StreamEvent(done=True),
        ]
    conv = conversation()
    events = await collect(Agent(provider, registry).run(conv, Mode.BYPASS))
    assert len(provider.requests) == MAX_ITERATIONS
    assert [e.notice for e in events if e.notice] == [NOTICE_MAX_ITER]
    assert conv.last_role() == "assistant"


async def test_cancel_event_during_tool_pairs_results():
    from yincode.agent import NOTICE_CANCELLED, Agent
    from yincode.tool import Registry, Result

    class BlockingTool:
        name = "block"
        description = "block"
        parameters = {"type": "object", "properties": {}}
        read_only = True

        def __init__(self):
            self.started = asyncio.Event()
            self.cleaned = asyncio.Event()

        async def execute(self, args: str) -> Result:
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.cleaned.set()
            return Result("never")

    blocker = BlockingTool()
    registry = Registry()
    registry.register(blocker)
    provider = ScriptProvider(
        [[StreamEvent(tool_calls=[ToolCall("c1", "block", "{}")]), StreamEvent(done=True)]]
    )
    conv = conversation()
    cancel = asyncio.Event()
    task = asyncio.create_task(
        collect(Agent(provider, registry).run(conv, Mode.BYPASS, cancel=cancel))
    )
    await asyncio.wait_for(blocker.started.wait(), 2)
    cancel.set()
    events = await asyncio.wait_for(task, 2)
    assert blocker.cleaned.is_set()
    assert conv.messages()[2].tool_results[0].is_error
    assert conv.last_role() == "assistant"
    assert [e.notice for e in events if e.notice] == [NOTICE_CANCELLED]


async def test_cancel_keeps_completed_result_in_read_only_batch():
    from yincode.agent import Agent
    from yincode.tool import Registry, Result

    class MixedSpeedTool:
        name = "probe"
        description = "probe"
        parameters = {"type": "object", "properties": {"kind": {"type": "string"}}}
        read_only = True

        def __init__(self):
            self.slow_started = asyncio.Event()
            self.fast_done = asyncio.Event()
            self.slow_cleaned = asyncio.Event()

        async def execute(self, args: str) -> Result:
            if "fast" in args:
                await self.slow_started.wait()
                self.fast_done.set()
                return Result("fast result")
            self.slow_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.slow_cleaned.set()
            return Result("never")

    tool = MixedSpeedTool()
    registry = Registry()
    registry.register(tool)
    provider = ScriptProvider(
        [
            [
                StreamEvent(
                    tool_calls=[
                        ToolCall("fast", "probe", '{"kind":"fast"}'),
                        ToolCall("slow", "probe", '{"kind":"slow"}'),
                    ]
                ),
                StreamEvent(done=True),
            ]
        ]
    )
    conv = conversation()
    cancel = asyncio.Event()
    task = asyncio.create_task(
        collect(Agent(provider, registry).run(conv, Mode.BYPASS, cancel=cancel))
    )
    await asyncio.wait_for(tool.fast_done.wait(), 2)
    cancel.set()
    await asyncio.wait_for(task, 2)
    assert tool.slow_cleaned.is_set()
    results = conv.messages()[2].tool_results
    assert [(result.tool_call_id, result.content, result.is_error) for result in results] == [
        ("fast", "fast result", False),
        ("slow", "工具执行已取消，未完成。", True),
    ]


async def test_plan_reminder_cadence_stable_environment_and_cache_usage(tmp_path: Path):
    from yincode.agent import Agent, Mode
    from yincode.prompt import build_system_prompt, plan_reminder
    from yincode.tool import new_default_registry

    (tmp_path / "sample.txt").write_text("sample", encoding="utf-8")
    scripts = [
        [
            StreamEvent(tool_calls=[ToolCall(f"c{i}", "read_file", '{"path":"sample.txt"}')]),
            StreamEvent(done=True),
        ]
        for i in range(9)
    ] + [[StreamEvent(usage=Usage(1, 2, 3, 4)), StreamEvent(text="plan"), StreamEvent(done=True)]]
    provider = ScriptProvider(scripts)
    conv = conversation()
    events = await collect(
        Agent(provider, new_default_registry(tmp_path), "test-version", cwd=tmp_path).run(
            conv, Mode.PLAN
        )
    )
    assert len(provider.assembled) == 10
    assert [req.reminder for req in provider.assembled] == [
        plan_reminder(i in (1, 5, 9)) for i in range(1, 11)
    ]
    assert {req.system.stable for req in provider.assembled} == {build_system_prompt()}
    assert all(
        str(tmp_path) in req.system.environment and "test-version" in req.system.environment
        for req in provider.assembled
    )
    assert "<system-reminder>" not in repr(conv.messages())
    assert [event.usage for event in events if event.usage] == [Usage(1, 2, 3, 4)]
    normal = ScriptProvider([[StreamEvent(text="normal"), StreamEvent(done=True)]])
    await collect(Agent(normal, new_default_registry(tmp_path), cwd=tmp_path).run(conversation()))
    assert normal.assembled[0].system.stable == provider.assembled[0].system.stable
    assert normal.assembled[0].reminder == ""


async def test_cancellation_during_environment_collection_waits_for_cleanup(monkeypatch):
    from yincode import prompt
    from yincode.agent import NOTICE_CANCELLED, Agent
    from yincode.tool import Registry

    started = asyncio.Event()
    cleaned = asyncio.Event()

    async def delayed_environment(version, model, *, cwd=None):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    monkeypatch.setattr(prompt, "gather_environment", delayed_environment)
    provider = ScriptProvider([])
    conv = conversation()
    cancel = asyncio.Event()
    task = asyncio.create_task(
        collect(Agent(provider, Registry()).run(conv, Mode.BYPASS, cancel=cancel))
    )
    await asyncio.wait_for(started.wait(), 2)
    cancel.set()
    events = await asyncio.wait_for(task, 2)
    assert cleaned.is_set()
    assert provider.requests == []
    assert conv.messages()[-1].content == NOTICE_CANCELLED
    assert events[-1].done
