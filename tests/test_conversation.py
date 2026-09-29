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


def test_messages_cannot_be_mutated_through_snapshot():
    mod = importlib.import_module("yincode.conversation")
    conv = mod.Conversation()
    conv.add_user("original")
    with pytest.raises(AttributeError):
        conv.messages()[0].content = "changed"
