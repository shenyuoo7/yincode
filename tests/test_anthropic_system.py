"""验证稳定系统块的缓存边界与请求副本。"""

from copy import deepcopy

import pytest

from yincode.llm import Message, Request, System, ToolCall, ToolResult


@pytest.mark.parametrize(
    "stable,environment", [("稳定", "环境"), ("", "环境"), ("稳定", ""), ("", "")]
)
def test_system_blocks_only_cache_stable_prefix(stable: str, environment: str) -> None:
    from yincode.llm.anthropic_provider import build_anthropic_system

    system = System(stable, environment)
    expected = []
    if stable:
        expected.append({"type": "text", "text": stable, "cache_control": {"type": "ephemeral"}})
    if environment:
        expected.append({"type": "text", "text": environment})
    assert build_anthropic_system(system) == expected
    assert build_anthropic_system(system) == build_anthropic_system(system)


def test_environment_changes_leave_stable_bytes_unchanged() -> None:
    from yincode.llm.anthropic_provider import build_anthropic_system

    first = build_anthropic_system(System("稳定\n指令", "日期 A"))
    second = build_anthropic_system(System("稳定\n指令", "日期 B"))
    assert first[0]["text"].encode() == second[0]["text"].encode()
    first[0]["text"] = "改副本"
    assert build_anthropic_system(System("稳定\n指令", "日期 B")) == second


@pytest.mark.parametrize("role", ["user", "assistant", "tool", "empty"])
def test_reminder_request_keeps_original_nested_history(role: str) -> None:
    from yincode.llm.anthropic_provider import _append_reminder_anthropic, _to_anthropic_messages
    from yincode.llm.openai_provider import _to_openai_messages

    history = [Message("user", "任务")]
    if role in ("assistant", "tool"):
        history.append(Message("assistant", tool_calls=[ToolCall("call-a", "glob", "{}")]))
    if role == "tool":
        history.append(Message("tool", tool_results=[ToolResult("call-a", "内容", True)]))
    if role == "empty":
        history = []
    req = Request(history, system=System("稳定", "环境"), reminder="临时提醒")
    before = deepcopy(req)
    messages = _to_anthropic_messages(req.messages)
    old_messages = deepcopy(messages)
    new_messages = _append_reminder_anthropic(messages, req.reminder)
    assert messages == old_messages
    assert new_messages[-1]["role"] == "user"
    assert new_messages[-1]["content"][-1] == {"type": "text", "text": "临时提醒"}
    if role == "tool":
        assert new_messages[-1]["content"][0] == {
            "type": "tool_result",
            "tool_use_id": "call-a",
            "content": "内容",
            "is_error": True,
        }
        assert len(new_messages) == len(messages)
    elif role == "user":
        assert new_messages[-1]["content"][0] == {"type": "text", "text": "任务"}
    else:
        assert len(new_messages) == len(messages) + 1
    openai_messages = _to_openai_messages(req)
    assert openai_messages[0] == {"role": "system", "content": "稳定\n\n环境"}
    assert openai_messages[-1] == {"role": "user", "content": "临时提醒"}
    if role == "tool":
        assert openai_messages[-2] == {"role": "tool", "tool_call_id": "call-a", "content": "内容"}
    assert req == before
