"""请求一批工具，回传结果后续答一次。"""

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from yincode.conversation import Conversation
from yincode.llm import Provider, StreamEvent, ToolCall, ToolDefinition, ToolResult
from yincode.llm._json import object_from_json
from yincode.tool import Registry


class Phase(Enum):
    START = "start"
    END = "end"


@dataclass(frozen=True, slots=True)
class ToolEvent:
    name: str
    tool_call_id: str
    args: str = ""
    phase: Phase = Phase.START
    result: str = ""
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class Event:
    text: str = ""
    tool: ToolEvent | None = None
    done: bool = False
    err: Exception | None = None

    def __post_init__(self) -> None:
        if sum((bool(self.text), self.tool is not None, self.done, self.err is not None)) > 1:
            raise ValueError("Agent 事件只能包含一种有效载荷")


class Agent:
    def __init__(
        self,
        provider: Provider,
        registry: Registry,
        *,
        redactor: Callable[[str], str] | None = None,
        secrets: tuple[str, ...] = (),
    ) -> None:
        self._provider = provider
        self._registry = registry
        self._redact = redactor or (lambda text: text)
        self._secrets = tuple(secret for secret in secrets if secret)

    def _stream_text(self, pending: str, text: str) -> tuple[str, str]:
        """只暂存可能是密钥前缀的尾部，防止跨分片泄漏。"""
        combined = self._redact(pending + text)
        keep = 0
        for secret in self._secrets:
            for length in range(min(len(secret) - 1, len(combined)), keep, -1):
                if combined.endswith(secret[:length]):
                    keep = length
                    break
        if keep:
            return combined[:-keep], combined[-keep:]
        return combined, ""

    def _redact_values(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._redact(value)
        if isinstance(value, dict):
            return {self._redact(k): self._redact_values(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._redact_values(v) for v in value]
        return value

    def _validate_calls(self, calls: list[ToolCall]) -> list[ToolCall]:
        normalized: list[ToolCall] = []
        ids: set[str] = set()
        for call in calls:
            if not call.id.strip() or not call.name.strip() or call.id in ids:
                raise RuntimeError("工具调用的 id 或名称不完整，或调用 id 重复")
            try:
                object_from_json(call.input)
            except (ValueError, TypeError):
                raise RuntimeError("工具调用参数不是完整 JSON 对象") from None
            ids.add(call.id)
            normalized.append(ToolCall(call.id, call.name, call.input or "{}"))
        return normalized

    async def _stream_once(
        self, conv: Conversation, definitions: list[ToolDefinition]
    ) -> AsyncIterator[StreamEvent]:
        iterator = self._provider.stream(conv.messages(), definitions)
        completed = False
        try:
            async for event in iterator:
                if event.err is not None:
                    raise event.err
                yield event
                if event.done:
                    completed = True
                    break
        finally:
            close = getattr(iterator, "aclose", None)
            if close is not None:
                await close()
        if not completed:
            raise RuntimeError("响应流提前结束，未收到完成事件，请重试")

    async def run(self, conv: Conversation) -> AsyncIterator[Event]:
        calls: list[ToolCall] = []
        history_calls: list[ToolCall] = []
        results: list[ToolResult] = []
        batch_saved = False
        results_saved = False
        final_saved = False
        iterator: AsyncIterator[StreamEvent] | None = None
        try:
            definitions = self._registry.definitions()
            preamble = ""
            pending = ""
            iterator = self._stream_once(conv, definitions)
            async for event in iterator:
                if event.text:
                    text, pending = self._stream_text(pending, event.text)
                    preamble += text
                    if text:
                        yield Event(text=text)
                calls.extend(event.tool_calls)
            if pending:
                text = self._redact(pending)
                preamble += text
                yield Event(text=text)
            preamble = self._redact(preamble)
            calls = self._validate_calls(calls)
            if not calls:
                if not preamble:
                    preamble = "[本轮未收到文本回复，请重试。]"
                    yield Event(text=preamble)
                conv.add_assistant(preamble)
                final_saved = True
                yield Event(done=True)
                return

            history_calls = [
                ToolCall(
                    self._redact(c.id),
                    self._redact(c.name),
                    json.dumps(
                        self._redact_values(object_from_json(c.input)),
                        ensure_ascii=False,
                        allow_nan=False,
                    ),
                )
                for c in calls
            ]
            conv.add_assistant_with_tool_calls(preamble, history_calls)
            batch_saved = True
            for call, public in zip(calls, history_calls, strict=True):
                yield Event(tool=ToolEvent(public.name, public.id, public.input))
                result = await self._registry.execute(call.name, call.input)
                content = self._redact(result.content)
                results.append(ToolResult(public.id, content, result.is_error))
                yield Event(
                    tool=ToolEvent(
                        public.name, public.id, public.input, Phase.END, content, result.is_error
                    )
                )
            conv.add_tool_results(results)
            results_saved = True

            final = ""
            pending = ""
            more_calls = False
            iterator = self._stream_once(conv, definitions)
            async for event in iterator:
                if event.text:
                    text, pending = self._stream_text(pending, event.text)
                    final += text
                    if text:
                        yield Event(text=text)
                more_calls = more_calls or bool(event.tool_calls)
            if pending:
                text = self._redact(pending)
                final += text
                yield Event(text=text)
            final = self._redact(final)
            if more_calls:
                notice = "[已达到单轮工具执行上限，请发送下一条消息继续。]"
                suffix = ("\n\n" if final else "") + notice
                final += suffix
                yield Event(text=suffix)
            elif not final:
                final = "[工具结果已回传，但未收到文本回复，请重试。]"
                yield Event(text=final)
            conv.add_assistant(final)
            final_saved = True
            yield Event(done=True)
        except (asyncio.CancelledError, GeneratorExit):
            raise
        except Exception as error:
            if batch_saved and not results_saved:
                completed_ids = {r.tool_call_id for r in results}
                results.extend(
                    ToolResult(c.id, "工具执行中断，未完成。", True)
                    for c in history_calls
                    if c.id not in completed_ids
                )
                conv.add_tool_results(results)
                results_saved = True
            message = self._redact(str(error))
            conv.add_assistant(f"[请求失败：{message}]")
            final_saved = True
            yield Event(err=RuntimeError(message))
        finally:
            # 生成器可能在 START/text/done 处被关闭；先同步补齐历史，再关闭流。
            if batch_saved and not results_saved:
                completed_ids = {r.tool_call_id for r in results}
                results.extend(
                    ToolResult(c.id, "工具执行已取消，未完成。", True)
                    for c in history_calls
                    if c.id not in completed_ids
                )
                conv.add_tool_results(results)
            if not final_saved:
                conv.add_assistant("[请求已取消。]")
            close = getattr(iterator, "aclose", None)
            if close is not None:
                await close()
