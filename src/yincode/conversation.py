from yincode.llm import Message


class Conversation:
    """仅在当前进程内保存会话，返回不可变消息的列表副本。"""

    def __init__(self) -> None:
        self._messages: list[Message] = []

    def add_user(self, text: str) -> None:
        self._messages.append(Message("user", text))

    def add_assistant(self, text: str) -> None:
        self._messages.append(Message("assistant", text))

    def messages(self) -> list[Message]:
        return list(self._messages)
