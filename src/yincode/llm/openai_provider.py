"""使用官方 OpenAI SDK 提供正文流。"""

import asyncio
from collections.abc import AsyncIterator

from openai import AsyncOpenAI, AsyncStream, Omit, omit
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionToolParam,
)

from yincode.config import ProviderConfig, redact
from yincode.llm import Message, StreamEvent, ToolCall, ToolDefinition, Usage
from yincode.llm._json import object_from_json
from yincode.prompt import SYSTEM_PROMPT


def _check_call(call_id: str, name: str, raw: str) -> None:
    if not isinstance(call_id, str) or not call_id.strip():
        raise RuntimeError("工具调用缺少有效 id")
    if not isinstance(name, str) or not name.strip():
        raise RuntimeError("工具调用缺少有效名称")
    object_from_json(raw)


def _to_openai_messages(
    msgs: list[Message], system_suffix: str = ""
) -> list[ChatCompletionMessageParam]:
    system = SYSTEM_PROMPT + "\n\n" + system_suffix if system_suffix else SYSTEM_PROMPT
    messages: list[ChatCompletionMessageParam] = [{"role": "system", "content": system}]
    for message in msgs:
        if message.role == "tool":
            messages.extend(
                {
                    "role": "tool",
                    "tool_call_id": result.tool_call_id,
                    "content": result.content,
                }
                for result in message.tool_results
            )
        elif message.role == "user":
            messages.append({"role": "user", "content": message.content})
        elif message.tool_calls:
            tool_calls: list[ChatCompletionMessageToolCallParam] = []
            for call in message.tool_calls:
                raw = call.input or "{}"
                _check_call(call.id, call.name, raw)
                tool_calls.append(
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": raw},
                    }
                )
            messages.append(
                {"role": "assistant", "content": message.content or None, "tool_calls": tool_calls}
            )
        else:
            messages.append({"role": "assistant", "content": message.content})
    return messages


class OpenAIProvider:
    def __init__(self, cfg: ProviderConfig, *, client: AsyncOpenAI | None = None) -> None:
        self._cfg = cfg
        if client is None:
            self._client = AsyncOpenAI(
                api_key=cfg.api_key,
                base_url=cfg.base_url or "https://api.openai.com/v1",
                organization="",
                project="",
                admin_api_key="",
                webhook_secret="",
                max_retries=0,
            )
            # 显式空值阻止环境身份回填，清理 SDK 无条件合入的环境请求头。
            self._client.organization = None
            self._client.project = None
            self._client.admin_api_key = None
            self._client._custom_headers = {}
            self._client._ambient_authorizations = frozenset()
        else:
            self._client = client
        self._streams: set[AsyncStream[ChatCompletionChunk]] = set()

    @property
    def name(self) -> str:
        return self._cfg.name

    @property
    def model(self) -> str:
        return self._cfg.model

    async def stream(
        self, msgs: list[Message], tools: list[ToolDefinition], system_suffix: str = ""
    ) -> AsyncIterator[StreamEvent]:
        completed = False
        finish_reason: str | None = None
        call_buffers: dict[int, dict[str, str]] = {}
        calls: list[ToolCall] = []
        usage: Usage | None = None
        try:
            messages = _to_openai_messages(msgs, system_suffix)
            tool_params: list[ChatCompletionToolParam] | Omit = (
                [
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.input_schema,
                        },
                    }
                    for tool in tools
                ]
                if tools
                else omit
            )
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
                tools=tool_params,
            )
            async with stream:
                self._streams.add(stream)
                try:
                    async for chunk in stream:
                        if chunk.usage is not None:
                            usage = Usage(
                                input_tokens=chunk.usage.prompt_tokens,
                                output_tokens=chunk.usage.completion_tokens,
                            )
                        if not chunk.choices:
                            continue
                        choice = chunk.choices[0]
                        if choice.finish_reason:
                            completed = True
                            finish_reason = choice.finish_reason
                        text = choice.delta.content
                        if text:
                            yield StreamEvent(text=text)
                        for part in choice.delta.tool_calls or []:
                            if not isinstance(part.index, int) or part.index < 0:
                                raise RuntimeError("工具调用缺少有效序号")
                            buffer = call_buffers.setdefault(
                                part.index, {"id": "", "name": "", "args": ""}
                            )
                            if part.id:
                                buffer["id"] += part.id
                            if part.function:
                                if part.function.name:
                                    buffer["name"] += part.function.name
                                if part.function.arguments:
                                    buffer["args"] += part.function.arguments
                finally:
                    self._streams.discard(stream)
            if not completed:
                raise RuntimeError("响应流提前结束，未收到完整回复")
            if call_buffers and finish_reason not in ("tool_calls", "stop"):
                raise RuntimeError("工具调用未完整结束")
            if finish_reason == "tool_calls" and not call_buffers:
                raise RuntimeError("工具回复缺少完整调用")
            for index in sorted(call_buffers):
                buffer = call_buffers[index]
                raw = buffer["args"] or "{}"
                _check_call(buffer["id"], buffer["name"], raw)
                calls.append(ToolCall(buffer["id"], buffer["name"], raw))
            if len({call.id for call in calls}) != len(calls):
                raise RuntimeError("工具调用 id 重复，无法配对结果")
        except asyncio.CancelledError:
            raise
        except ValueError:
            yield StreamEvent(err=RuntimeError("工具调用参数无效，必须是完整的 JSON 对象"))
        except Exception as error:
            # 错误事件只携带脱敏后的文本，不保留包含凭证的 SDK 异常对象。
            yield StreamEvent(err=RuntimeError(redact(str(error), [self._cfg.api_key])))
        else:
            if calls:
                yield StreamEvent(tool_calls=calls)
            if usage is not None:
                yield StreamEvent(usage=usage)
            yield StreamEvent(done=True)

    async def aclose(self) -> None:
        try:
            for stream in tuple(self._streams):
                await stream.close()
        finally:
            await self._client.close()
