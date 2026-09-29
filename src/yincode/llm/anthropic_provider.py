"""使用官方 Anthropic SDK 提供正文流。"""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from anthropic import AsyncAnthropic, Omit, omit
from anthropic.lib.streaming import AsyncMessageStream
from anthropic.types import (
    MessageParam,
    TextBlockParam,
    ThinkingConfigParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)

from yincode.config import ProviderConfig, redact
from yincode.llm import Message, StreamEvent, ToolCall, ToolDefinition
from yincode.llm._json import object_from_json
from yincode.prompt import SYSTEM_PROMPT


def _json_object(raw: str) -> dict[str, Any]:
    return object_from_json(raw)


def _check_identity(call_id: str, name: str) -> None:
    if not isinstance(call_id, str) or not call_id.strip():
        raise RuntimeError("工具调用缺少有效 id")
    if not isinstance(name, str) or not name.strip():
        raise RuntimeError("工具调用缺少有效名称")


def _to_anthropic_messages(msgs: list[Message]) -> list[MessageParam]:
    messages: list[MessageParam] = []
    for message in msgs:
        if message.role == "tool":
            results: list[ToolResultBlockParam] = [
                {
                    "type": "tool_result",
                    "tool_use_id": result.tool_call_id,
                    "content": result.content,
                    "is_error": result.is_error,
                }
                for result in message.tool_results
            ]
            messages.append({"role": "user", "content": results})
        elif message.role == "assistant" and message.tool_calls:
            content: list[TextBlockParam | ToolUseBlockParam] = []
            if message.content:
                content.append({"type": "text", "text": message.content})
            for call in message.tool_calls:
                _check_identity(call.id, call.name)
                content.append(
                    {
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": _json_object(call.input),
                    }
                )
            messages.append({"role": "assistant", "content": content})
        else:
            messages.append({"role": message.role, "content": message.content})
    return messages


class AnthropicProvider:
    def __init__(self, cfg: ProviderConfig, *, client: AsyncAnthropic | None = None) -> None:
        self._cfg = cfg
        if client is None:
            self._client = AsyncAnthropic(
                api_key=cfg.api_key,
                base_url=cfg.base_url or "https://api.anthropic.com",
                webhook_key="",
                max_retries=0,
            )
            # SDK 会合入环境请求头，只清理本产品创建的客户端。
            self._client._custom_headers = {}
        else:
            self._client = client
        self._streams: set[AsyncMessageStream[None]] = set()

    @property
    def name(self) -> str:
        return self._cfg.name

    @property
    def model(self) -> str:
        return self._cfg.model

    async def stream(
        self, msgs: list[Message], tools: list[ToolDefinition]
    ) -> AsyncIterator[StreamEvent]:
        completed = False
        calls: list[ToolCall] = []
        json_inputs: dict[int, str] = {}
        closed_tools: set[int] = set()
        try:
            messages = _to_anthropic_messages(msgs)
            tool_params: list[ToolParam] | Omit = (
                [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.input_schema,
                    }
                    for tool in tools
                ]
                if tools
                else omit
            )
            has_tools = bool(tools) or any(
                message.tool_calls or message.tool_results for message in msgs
            )
            thinking: ThinkingConfigParam | Omit = (
                {"type": "enabled", "budget_tokens": 2048}
                if self._cfg.thinking and not has_tools
                else omit
            )
            async with self._client.messages.stream(
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=messages,
                thinking=thinking,
                tools=tool_params,
            ) as stream:
                self._streams.add(stream)
                try:
                    async for event in stream:
                        if event.type == "message_stop":
                            completed = True
                        if (
                            event.type == "content_block_start"
                            and event.content_block.type == "tool_use"
                        ):
                            json_inputs[event.index] = ""
                        if (
                            event.type == "content_block_delta"
                            and event.delta.type == "input_json_delta"
                        ):
                            if event.index in json_inputs:
                                json_inputs[event.index] += event.delta.partial_json
                        if event.type == "content_block_stop" and event.index in json_inputs:
                            closed_tools.add(event.index)
                        if (
                            event.type == "content_block_delta"
                            and event.delta.type == "text_delta"
                            and event.delta.text
                        ):
                            yield StreamEvent(text=event.delta.text)
                    if not completed:
                        raise RuntimeError("响应流提前结束，未收到完整回复")
                    final_message = await stream.get_final_message()
                    for index, block in enumerate(final_message.content):
                        if block.type != "tool_use":
                            continue
                        if index not in closed_tools or final_message.stop_reason != "tool_use":
                            raise RuntimeError("工具调用未完整结束")
                        _check_identity(block.id, block.name)
                        # SDK 用宽容模式累加 JSON；严格验证原串，防止截断参数成为有效快照。
                        raw = json_inputs[index]
                        if raw:
                            _json_object(raw)
                        if not isinstance(block.input, dict):
                            raise ValueError("工具参数必须是 JSON 对象")
                        calls.append(
                            ToolCall(
                                block.id,
                                block.name,
                                json.dumps(block.input, ensure_ascii=False, allow_nan=False),
                            )
                        )
                    if final_message.stop_reason == "tool_use" and not calls:
                        raise RuntimeError("工具回复缺少完整调用")
                    if len({call.id for call in calls}) != len(calls):
                        raise RuntimeError("工具调用 id 重复，无法配对结果")
                finally:
                    self._streams.discard(stream)
        except asyncio.CancelledError:
            raise
        except ValueError:
            # SDK 参数解析异常可能包含模型原串；只输出固定、可读的校验错误。
            yield StreamEvent(err=RuntimeError("工具调用参数无效，必须是完整的 JSON 对象"))
        except Exception as error:
            # 创建独立异常，避免向界面传递 SDK 的请求、响应和原始异常链。
            yield StreamEvent(err=RuntimeError(redact(str(error), [self._cfg.api_key])))
        else:
            if calls:
                yield StreamEvent(tool_calls=calls)
            yield StreamEvent(done=True)

    async def aclose(self) -> None:
        try:
            for stream in tuple(self._streams):
                await stream.close()
        finally:
            await self._client.close()
