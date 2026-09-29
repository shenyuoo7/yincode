from dataclasses import replace

from yincode.llm import Message, ToolCall, ToolResult


class Conversation:
    """仅在当前进程内保存会话，返回不可变消息的列表副本。"""

    def __init__(self) -> None:
        self._messages: list[Message] = []

    def add_user(self, text: str) -> None:
        self._messages.append(Message("user", text))

    def add_assistant(self, text: str) -> None:
        self._messages.append(Message("assistant", text))

    def add_assistant_with_tool_calls(self, text: str, calls: list[ToolCall]) -> None:
        self._messages.append(Message("assistant", text, tool_calls=list(calls)))

    def add_tool_results(self, results: list[ToolResult]) -> None:
        self._messages.append(Message("tool", tool_results=list(results)))

    def messages(self) -> list[Message]:
        return [
            replace(m, tool_calls=list(m.tool_calls), tool_results=list(m.tool_results))
            for m in self._messages
        ]
