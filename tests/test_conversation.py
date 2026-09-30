import importlib

import pytest


def test_conversation_retains_order_and_returns_safe_snapshot():
    mod = importlib.import_module("yincode.conversation")
    conv = mod.Conversation()
    conv.add_user("remember 42")
    conv.add_assistant("okay")
    conv.add_user("what number?")
    snapshot = conv.messages()
    assert [(m.role, m.content) for m in snapshot] == [
        ("user", "remember 42"),
        ("assistant", "okay"),
        ("user", "what number?"),
    ]
    snapshot.clear()
    assert len(conv.messages()) == 3


def test_stream_event_rejects_conflicting_payloads():
    mod = importlib.import_module("yincode.llm")
    with pytest.raises(ValueError):
        mod.StreamEvent(text="text", done=True)
    with pytest.raises(ValueError):
        mod.StreamEvent(done=True, err=ValueError("failure"))
    assert mod.StreamEvent(usage=mod.Usage(3, 5)).usage.input_tokens == 3
    with pytest.raises(ValueError):
        mod.StreamEvent(usage=mod.Usage(1, 2), done=True)


def test_last_role_tracks_conversation_tail():
    from yincode.conversation import Conversation
    from yincode.llm import ToolResult

    conv = Conversation()
    assert conv.last_role() == ""
    conv.add_user("hello")
    assert conv.last_role() == "user"
    conv.add_tool_results([ToolResult("c1", "ok")])
    assert conv.last_role() == "tool"
    conv.add_assistant("done")
    assert conv.last_role() == "assistant"


def test_request_defaults_are_independent_and_cache_usage_is_available():
    from yincode.llm import Message, Request, System, Usage

    first = Request()
    first.messages.append(Message("user", "hello"))
    assert Request().messages == []
    assert first.system == System()
    usage = Usage(1, 2, cache_write=3, cache_read=4)
    assert (usage.cache_write, usage.cache_read) == (3, 4)


def test_messages_cannot_be_mutated_through_snapshot():
    mod = importlib.import_module("yincode.conversation")
    conv = mod.Conversation()
    conv.add_user("original")
    with pytest.raises(AttributeError):
        conv.messages()[0].content = "changed"


def test_tool_history_is_paired_and_snapshot_lists_are_isolated():
    from yincode.conversation import Conversation
    from yincode.llm import ToolCall, ToolResult

    conv = Conversation()
    calls = [ToolCall("c1", "read_file", '{"path":"demo.txt"}')]
    results = [ToolResult("c1", "1\thello")]
    conv.add_user("read")
    conv.add_assistant_with_tool_calls("", calls)
    conv.add_tool_results(results)
    conv.add_assistant("hello")
    calls.clear()
    results.clear()
    snapshot = conv.messages()
    assert [m.role for m in snapshot] == ["user", "assistant", "tool", "assistant"]
    assert snapshot[1].tool_calls[0].id == snapshot[2].tool_results[0].tool_call_id
    snapshot[1].tool_calls.clear()
    snapshot[2].tool_results.clear()
    assert len(conv.messages()[1].tool_calls) == len(conv.messages()[2].tool_results) == 1


def test_tool_events_are_exclusive_and_have_independent_defaults():
    from yincode.llm import Message, StreamEvent, ToolCall

    call = ToolCall("c1", "read_file", "{}")
    for kwargs in ({"text": "x"}, {"done": True}, {"err": ValueError("x")}):
        with pytest.raises(ValueError):
            StreamEvent(tool_calls=[call], **kwargs)
    first = Message("tool")
    first.tool_calls.append(call)
    assert Message("tool").tool_calls == []
