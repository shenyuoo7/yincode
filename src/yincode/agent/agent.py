"""ReAct 多轮工具编排与合法会话收尾。"""

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any

from yincode.conversation import Conversation
from yincode.llm import Provider, StreamEvent, ToolCall, ToolDefinition, ToolResult, Usage
from yincode.llm._json import object_from_json
from yincode.prompt import PLAN_MODE_REMINDER
from yincode.tool import Registry

MAX_ITERATIONS = 25
MAX_UNKNOWN_RUN = 3
NOTICE_MAX_ITER = "（已达最大迭代轮数 25，自动停止；可继续发消息推进。）"
NOTICE_UNKNOWN_TOOLS = "（连续多轮只请求到未注册的工具，自动停止。）"
NOTICE_STREAM_ERR = "（请求出错，本轮已中断。）"
NOTICE_CANCELLED = "（已取消。）"


class Mode(IntEnum):
    NORMAL = 0
    PLAN = 1


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
    usage: Usage | None = None
    iter: int = 0
    notice: str = ""
    done: bool = False
    err: Exception | None = None

    @property
    def iteration(self) -> int:
        return self.iter

    def __post_init__(self) -> None:
        if (
            sum(
                (
                    bool(self.text),
                    self.tool is not None,
                    self.usage is not None,
                    bool(self.iter),
                    bool(self.notice),
                    self.done,
                    self.err is not None,
                )
            )
            > 1
        ):
            raise ValueError("Agent 事件只能包含一种有效载荷")


@dataclass(slots=True)
class StreamState:
    text: str = ""
    calls: list[ToolCall] = field(default_factory=list)
    completed: bool = False
    cancelled: bool = False


async def _next_event(iterator: AsyncIterator[StreamEvent]) -> StreamEvent:
    return await anext(iterator)


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
        """暂存可能是密钥前缀的尾部，防止跨分片泄漏。"""
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

    def _public_calls(self, calls: list[ToolCall]) -> list[ToolCall]:
        return [
            ToolCall(
                self._redact(call.id),
                self._redact(call.name),
                json.dumps(
                    self._redact_values(object_from_json(call.input)),
                    ensure_ascii=False,
                    allow_nan=False,
                ),
            )
            for call in calls
        ]

    async def _stream_once(
        self,
        conv: Conversation,
        definitions: list[ToolDefinition],
        suffix: str,
        cancel: asyncio.Event,
        state: StreamState,
    ) -> AsyncIterator[Event]:
        iterator = self._provider.stream(conv.messages(), definitions, suffix)
        pending = ""
        next_task: asyncio.Task[StreamEvent] | None = None
        cancel_task: asyncio.Task[bool] | None = None
        try:
            while not cancel.is_set():
                next_task = asyncio.create_task(_next_event(iterator))
                cancel_task = asyncio.create_task(cancel.wait())
                done, _ = await asyncio.wait(
                    {next_task, cancel_task}, return_when=asyncio.FIRST_COMPLETED
                )
                if cancel_task in done or cancel.is_set():
                    state.cancelled = True
                    break
                cancel_task.cancel()
                await asyncio.gather(cancel_task, return_exceptions=True)
                cancel_task = None
                try:
                    event = next_task.result()
                except StopAsyncIteration:
                    break
                next_task = None
                if event.err is not None:
                    raise event.err
                if event.text:
                    visible, pending = self._stream_text(pending, event.text)
                    state.text += visible
                    if visible:
                        yield Event(text=visible)
                if event.tool_calls:
                    state.calls.extend(event.tool_calls)
                if event.usage is not None:
                    yield Event(usage=event.usage)
                if event.done:
                    state.completed = True
                    break
            if cancel.is_set():
                state.cancelled = True
            if state.completed and pending:
                visible = self._redact(pending)
                state.text += visible
                yield Event(text=visible)
            if not state.cancelled and not state.completed:
                raise RuntimeError("响应流提前结束，未收到完成事件，请重试")
        finally:
            for task in (next_task, cancel_task):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(
                *(task for task in (next_task, cancel_task) if task is not None),
                return_exceptions=True,
            )
            close = getattr(iterator, "aclose", None)
            if close is not None:
                await close()

    async def _run_group(
        self,
        calls: list[ToolCall],
        indexes: list[int],
        allowed_names: set[str],
        cancel: asyncio.Event,
        slots: list[ToolResult | None],
    ) -> bool:
        """执行一个连续只读批或单个副作用工具，并等待取消清理。"""
        tasks = [
            asyncio.create_task(
                self._registry.execute(
                    calls[index].name, calls[index].input, allowed_names=allowed_names
                )
            )
            for index in indexes
        ]
        cancel_task = asyncio.create_task(cancel.wait())
        try:
            done, _ = await asyncio.wait({*tasks, cancel_task}, return_when=asyncio.FIRST_COMPLETED)
            if cancel_task in done or cancel.is_set():
                return False
            # 首个工具结束不代表整批结束；继续等待全部工具或取消。
            pending: set[asyncio.Task[Any]] = {task for task in tasks if not task.done()}
            while pending:
                done, pending = await asyncio.wait(
                    pending | {cancel_task}, return_when=asyncio.FIRST_COMPLETED
                )
                if cancel_task in done or cancel.is_set():
                    return False
                pending.discard(cancel_task)
            for index, task in zip(indexes, tasks, strict=True):
                result = task.result()
                slots[index] = ToolResult(
                    calls[index].id, self._redact(result.content), result.is_error
                )
            return True
        finally:
            cancel_task.cancel()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, cancel_task, return_exceptions=True)
            # 取消只读批时保留已经完成的调用；仅未完成项由上层补取消结果。
            for index, task in zip(indexes, tasks, strict=True):
                if slots[index] is not None or task.cancelled():
                    continue
                try:
                    result = task.result()
                except Exception:
                    continue
                slots[index] = ToolResult(
                    calls[index].id, self._redact(result.content), result.is_error
                )

    async def run(
        self,
        conv: Conversation,
        mode: Mode = Mode.NORMAL,
        cancel: asyncio.Event | None = None,
    ) -> AsyncIterator[Event]:
        cancel = cancel or asyncio.Event()
        definitions = (
            self._registry.read_only_definitions()
            if mode is Mode.PLAN
            else self._registry.definitions()
        )
        allowed_names = {definition.name for definition in definitions}
        suffix = PLAN_MODE_REMINDER if mode is Mode.PLAN else ""
        history_calls: list[ToolCall] = []
        slots: list[ToolResult | None] = []
        batch_saved = False
        final_saved = False
        unknown_run = 0
        try:
            for iteration in range(1, MAX_ITERATIONS + 1):
                if cancel.is_set():
                    conv.add_assistant(NOTICE_CANCELLED)
                    final_saved = True
                    yield Event(notice=NOTICE_CANCELLED)
                    yield Event(done=True)
                    return
                yield Event(iter=iteration)
                state = StreamState()
                async for event in self._stream_once(conv, definitions, suffix, cancel, state):
                    yield event
                if state.cancelled:
                    conv.add_assistant(NOTICE_CANCELLED)
                    final_saved = True
                    yield Event(notice=NOTICE_CANCELLED)
                    yield Event(done=True)
                    return
                calls = self._validate_calls(state.calls)
                if not calls:
                    final = self._redact(state.text)
                    if not final:
                        final = "[本轮未收到文本回复，请重试。]"
                        yield Event(text=final)
                    conv.add_assistant(final)
                    final_saved = True
                    yield Event(done=True)
                    return

                history_calls = self._public_calls(calls)
                conv.add_assistant_with_tool_calls(self._redact(state.text), history_calls)
                slots = [None] * len(calls)
                batch_saved = True
                unknown_run = (
                    unknown_run + 1
                    if all(self._registry.get(call.name) is None for call in calls)
                    else 0
                )
                for call in history_calls:
                    yield Event(tool=ToolEvent(call.name, call.id, call.input))

                index = 0
                while index < len(calls) and not cancel.is_set():
                    end = index + 1
                    if self._registry.is_read_only(calls[index].name):
                        while end < len(calls) and self._registry.is_read_only(calls[end].name):
                            end += 1
                    group = list(range(index, end))
                    completed = await self._run_group(calls, group, allowed_names, cancel, slots)
                    if not completed:
                        break
                    index = end
                for i, public in enumerate(history_calls):
                    if slots[i] is None:
                        slots[i] = ToolResult(public.id, "工具执行已取消，未完成。", True)
                    else:
                        previous = slots[i]
                        assert previous is not None
                        slots[i] = ToolResult(public.id, previous.content, previous.is_error)
                    result = slots[i]
                    assert result is not None
                    yield Event(
                        tool=ToolEvent(
                            public.name,
                            public.id,
                            public.input,
                            Phase.END,
                            result.content,
                            result.is_error,
                        )
                    )
                conv.add_tool_results([result for result in slots if result is not None])
                batch_saved = False
                if cancel.is_set():
                    conv.add_assistant(NOTICE_CANCELLED)
                    final_saved = True
                    yield Event(notice=NOTICE_CANCELLED)
                    yield Event(done=True)
                    return
                if unknown_run >= MAX_UNKNOWN_RUN:
                    conv.add_assistant(NOTICE_UNKNOWN_TOOLS)
                    final_saved = True
                    yield Event(notice=NOTICE_UNKNOWN_TOOLS)
                    yield Event(done=True)
                    return

            conv.add_assistant(NOTICE_MAX_ITER)
            final_saved = True
            yield Event(notice=NOTICE_MAX_ITER)
            yield Event(done=True)
        except (asyncio.CancelledError, GeneratorExit):
            raise
        except Exception as error:
            message = self._redact(str(error))
            if batch_saved:
                self._finish_batch(conv, history_calls, slots)
                batch_saved = False
            conv.add_assistant(f"[请求失败：{message}]")
            final_saved = True
            yield Event(err=RuntimeError(message))
        finally:
            if batch_saved:
                self._finish_batch(conv, history_calls, slots)
            if not final_saved:
                conv.add_assistant("[请求已取消。]")

    @staticmethod
    def _finish_batch(
        conv: Conversation, calls: list[ToolCall], slots: list[ToolResult | None]
    ) -> None:
        results = [
            result if result is not None else ToolResult(call.id, "工具执行已取消，未完成。", True)
            for call, result in zip(calls, slots, strict=True)
        ]
        conv.add_tool_results(results)
