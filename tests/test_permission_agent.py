"""通过真实文件和会话结果验证权限编排。"""

import asyncio
import json
import os
import subprocess
from pathlib import Path

import pytest

from yincode.agent import Agent
from yincode.conversation import Conversation
from yincode.llm import Request, StreamEvent, ToolCall
from yincode.tool import new_default_registry


class Provider:
    name = "fixture"
    model = "fixture"

    def __init__(self, calls):
        self.calls = calls
        self.requests = []

    async def stream(self, request: Request):
        self.requests.append(request)
        if len(self.requests) == 1:
            yield StreamEvent(tool_calls=self.calls)
        else:
            yield StreamEvent(text="完成")
        yield StreamEvent(done=True)


def prepare(tmp_path, calls):
    from yincode.permission import new_engine

    engine, _ = new_engine(str(tmp_path))
    provider = Provider(calls)
    agent = Agent(provider, new_default_registry(tmp_path), cwd=tmp_path, engine=engine)
    conv = Conversation()
    conv.add_user("实施")
    return agent, conv, provider, engine


@pytest.mark.parametrize("choice", ["ALLOW_ONCE", "ALLOW_FOREVER", "DENY_ONCE"])
async def test_default_asks_and_continues_after_choice(tmp_path, choice):
    from yincode.permission import Outcome

    writing = ToolCall("w", "write_file", '{"path":"new.txt","content":"fixture"}')
    agent, conv, provider, engine = prepare(tmp_path, [writing])
    approvals = []
    async for event in agent.run(conv):
        if event.approval is not None:
            approvals.append(event.approval)
            assert not (tmp_path / "new.txt").exists()
            event.approval.respond.set_result(Outcome[choice])
    assert len(approvals) == 1
    assert (tmp_path / "new.txt").exists() is (choice != "DENY_ONCE")
    local_path = Path(engine.local_path)
    assert await asyncio.to_thread(local_path.exists) is (choice == "ALLOW_FOREVER")
    assert conv.messages()[2].tool_results[0].is_error is (choice == "DENY_ONCE")
    assert len(provider.requests) == 2 and conv.messages()[-1].content == "完成"


async def test_mixed_denials_preserve_ids_and_do_not_execute(tmp_path):
    from yincode.permission import Mode

    (tmp_path / "inside.txt").write_text("inside fixture")
    calls = [
        ToolCall("outside", "read_file", '{"path":"../secret"}'),
        ToolCall("inside", "read_file", '{"path":"inside.txt"}'),
        ToolCall("danger", "bash", '{"command":"rm -rf /"}'),
        ToolCall("valid", "write_file", '{"path":"valid.txt","content":"ok"}'),
    ]
    agent, conv, provider, _ = prepare(tmp_path, calls)
    events = [event async for event in agent.run(conv, Mode.BYPASS)]
    results = conv.messages()[2].tool_results
    assert [item.tool_call_id for item in results] == ["outside", "inside", "danger", "valid"]
    assert [item.is_error for item in results] == [True, False, True, False]
    assert "inside fixture" in results[1].content
    assert (tmp_path / "valid.txt").read_text() == "ok"
    assert all(event.approval is None for event in events)
    assert len(provider.requests) == 2


@pytest.mark.parametrize("raw", ["{bad", "[]", '{"x":NaN}', '{"path":1}', "{}"])
async def test_invalid_call_is_paired_error_and_loop_continues(tmp_path, raw):
    from yincode.permission import Mode

    agent, conv, provider, _ = prepare(tmp_path, [ToolCall("bad", "write_file", raw)])
    events = [event async for event in agent.run(conv, Mode.BYPASS)]
    assert not any(event.err for event in events)
    assert len(provider.requests) == 2
    assert conv.messages()[2].tool_results[0].is_error
    # 即使拒绝原始 JSON，也必须把合法的工具调用参数交回提供者。
    json.loads(conv.messages()[1].tool_calls[0].input)


@pytest.mark.parametrize("external", [False, True])
async def test_approval_cancel_keeps_legal_history_and_cleans_future(tmp_path, external):
    writing = ToolCall("w", "write_file", '{"path":"new.txt","content":"fixture"}')
    agent, conv, _, _ = prepare(tmp_path, [writing])
    cancel = asyncio.Event()
    seen = asyncio.Event()
    approvals = []
    baseline = set(asyncio.all_tasks())

    async def collect():
        async for event in agent.run(conv, cancel=cancel):
            if event.approval is not None:
                approvals.append(event.approval)
                seen.set()

    task = asyncio.create_task(collect())
    await asyncio.wait_for(seen.wait(), 3)
    if external:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        cancel.set()
        await asyncio.wait_for(task, 3)
    assert approvals[0].respond.done()
    assert not (tmp_path / "new.txt").exists()
    assert [item.role for item in conv.messages()] == ["user", "assistant", "tool", "assistant"]
    assert conv.messages()[2].tool_results[0].is_error
    assert set(asyncio.all_tasks()) <= baseline


async def test_persist_failure_warns_and_still_executes_once(tmp_path, monkeypatch):
    from yincode.permission import Outcome

    agent, conv, _, engine = prepare(
        tmp_path, [ToolCall("w", "write_file", '{"path":"new.txt","content":"fixture"}')]
    )

    def failed(*args):
        raise OSError("unwritable")

    monkeypatch.setattr(type(engine), "persist_local_allow", failed)
    notices = []
    async for event in agent.run(conv):
        if event.approval:
            event.approval.respond.set_result(Outcome.ALLOW_FOREVER)
        if event.notice:
            notices.append(event.notice)
    assert any("未保存" in text for text in notices)
    assert (tmp_path / "new.txt").read_text() == "fixture"


async def test_plan_cannot_be_overridden_by_allow_rule(tmp_path):
    from yincode.permission import Mode
    from yincode.permission.rule import Rule, RuleSet

    agent, conv, provider, engine = prepare(
        tmp_path, [ToolCall("w", "write_file", '{"path":"new.txt","content":"fixture"}')]
    )
    engine.local = RuleSet(allow=[Rule("Write", "", True)])
    events = [event async for event in agent.run(conv, Mode.PLAN)]
    assert not (tmp_path / "new.txt").exists()
    assert all(event.approval is None for event in events)
    assert {tool.name for tool in provider.requests[0].tools} == {"read_file", "glob", "grep"}
    assert provider.requests[0].reminder
    assert conv.messages()[2].tool_results[0].is_error


async def test_approval_rechecks_path_before_execution(tmp_path):
    from yincode.permission import Outcome

    if os.name != "nt":
        pytest.skip("Windows 目录联接专用")
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = root / "approved"
    target.mkdir()
    agent, conv, _, _ = prepare(
        root, [ToolCall("w", "write_file", '{"path":"approved/new.txt","content":"fixture"}')]
    )
    async for event in agent.run(conv):
        if event.approval:
            await asyncio.to_thread(target.rmdir)
            await asyncio.to_thread(
                subprocess.run,
                ["cmd", "/c", "mklink", "/J", str(target), str(outside)],
                check=True,
                capture_output=True,
            )
            event.approval.respond.set_result(Outcome.ALLOW_ONCE)
    assert not (outside / "new.txt").exists()
    assert conv.messages()[2].tool_results[0].is_error


def test_agent_rejects_tools_bound_to_different_directory(tmp_path):
    from yincode.permission import new_engine

    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    engine, _ = new_engine(str(root))
    with pytest.raises(ValueError, match="工作目录"):
        Agent(Provider([]), new_default_registry(outside), cwd=root, engine=engine)


async def test_denied_read_does_not_serialize_remaining_read_batch(tmp_path):
    from yincode.permission import new_engine
    from yincode.tool import Registry, Result

    class Reader:
        name = "read_file"
        description = "fixture"
        parameters = {"type": "object", "properties": {"path": {"type": "string"}}}
        read_only = True
        active = 0
        peak = 0

        async def execute(self, args):
            self.active += 1
            self.peak = max(self.peak, self.active)
            if self.active == 2:
                barrier.set()
            await asyncio.wait_for(barrier.wait(), 2)
            self.active -= 1
            return Result("read fixture")

    barrier = asyncio.Event()
    reader = Reader()
    registry = Registry(cwd=tmp_path)
    registry.register(reader)
    calls = [
        ToolCall("outside", "read_file", '{"path":"../outside"}'),
        ToolCall("first", "read_file", '{"path":"one"}'),
        ToolCall("second", "read_file", '{"path":"two"}'),
    ]
    provider = Provider(calls)
    engine, _ = new_engine(str(tmp_path))
    agent = Agent(provider, registry, cwd=tmp_path, engine=engine)
    conv = Conversation()
    conv.add_user("read fixtures")
    events = [event async for event in agent.run(conv)]
    assert reader.peak == 2
    assert all(event.approval is None for event in events)
    results = conv.messages()[2].tool_results
    assert [result.tool_call_id for result in results] == ["outside", "first", "second"]
    assert [result.is_error for result in results] == [True, False, False]


async def test_registered_tool_without_read_only_metadata_requires_approval(tmp_path):
    from yincode.permission import Mode, Outcome
    from yincode.tool import Registry, Result

    class Undeclared:
        name = "custom"
        description = "fixture"
        parameters = {"type": "object"}
        executions = 0

        async def execute(self, args):
            self.executions += 1
            return Result("fixture")

    tool = Undeclared()
    registry = Registry(cwd=tmp_path)
    registry.register(tool)
    assert not registry.is_read_only("custom")
    assert registry.read_only_definitions() == []
    for mode in (Mode.DEFAULT, Mode.ACCEPT_EDITS, Mode.BYPASS):
        provider = Provider([ToolCall("c", "custom", "{}")])
        agent = Agent(provider, registry, cwd=tmp_path)
        conv = Conversation()
        conv.add_user("custom fixture")
        approvals = []
        async for event in agent.run(conv, mode):
            if event.approval:
                approvals.append(event.approval)
                event.approval.respond.set_result(Outcome.DENY_ONCE)
        assert bool(approvals) is (mode is not Mode.BYPASS)
    assert tool.executions == 1
