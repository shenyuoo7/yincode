"""ReAct 多轮工具编排与合法会话收尾。"""

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from yincode import __version__, prompt
from yincode.conversation import Conversation
from yincode.llm import Provider, Request, StreamEvent, System, ToolCall, ToolResult, Usage
from yincode.llm._json import object_from_json
from yincode.permission import Decision, Engine, Mode, Outcome, new_engine
from yincode.tool import Registry, Result

MAX_ITERATIONS = 25
MAX_UNKNOWN_RUN = 3
PLAN_REMINDER_INTERVAL = 4
NOTICE_MAX_ITER = "（已达最大迭代轮数 25，自动停止；可继续发消息推进。）"
NOTICE_UNKNOWN_TOOLS = "（连续多轮只请求到未注册的工具，自动停止。）"
NOTICE_STREAM_ERR = "（请求出错，本轮已中断。）"
NOTICE_CANCELLED = "（已取消。）"


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    name: str
    args: str
    reason: str
    respond: asyncio.Future[Outcome]


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
    approval: ApprovalRequest | None = None

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
                    self.approval is not None,
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
        version: str = __version__,
        *,
        engine: Engine | None = None,
        cwd: str | Path | None = None,
        redactor: Callable[[str], str] | None = None,
        secrets: tuple[str, ...] = (),
    ) -> None:
        self._provider = provider
        self._registry = registry
        self._version = version
        self._cwd = Path(cwd if cwd is not None else registry.cwd).resolve()
        self.engine = engine if engine is not None else new_engine(str(self._cwd))[0]
        if not self.engine.deny_all and self._cwd != Path(self.engine.root):
            raise ValueError("权限引擎与会话工作目录必须一致")
        for definition in registry.definitions():
            tool = registry.get(definition.name)
            directory = getattr(tool, "cwd", None)
            if directory is not None and Path(directory).resolve() != self._cwd:
                raise ValueError("工具与会话工作目录必须一致")
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
            ids.add(call.id)
            normalized.append(ToolCall(call.id, call.name, call.input or "{}"))
        return normalized

    def _public_calls(self, calls: list[ToolCall]) -> list[ToolCall]:
        def public_input(raw: str) -> str:
            try:
                value = object_from_json(raw)
            except (ValueError, TypeError):
                # 拒绝非法原始参数时，历史仍需能被 SDK 序列化并配对错误结果。
                value = {}
            return json.dumps(self._redact_values(value), ensure_ascii=False, allow_nan=False)

        return [
            ToolCall(
                self._redact(call.id),
                self._redact(call.name),
                public_input(call.input),
            )
            for call in calls
        ]

    async def _stream_once(
        self,
        req: Request,
        cancel: asyncio.Event,
        state: StreamState,
    ) -> AsyncIterator[Event]:
        iterator = self._provider.stream(req)
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
        mode: Mode,
    ) -> bool:
        """执行一个连续只读批或单个副作用工具，并等待取消清理。"""
        if not indexes:
            return True

        async def execute_checked(index: int) -> Result:
            call = calls[index]
            # 审批期间路径可能被改动；执行开始前重验强制边界与拒绝规则。
            decision, reason = self.engine.check(
                mode,
                call,
                self._registry.is_read_only(call.name),
                registered=self._registry.get(call.name) is not None,
                allowed=call.name in allowed_names,
            )
            if decision is Decision.DENY:
                return Result(reason, is_error=True)
            return await self._registry.execute(call.name, call.input, allowed_names=allowed_names)

        tasks = [asyncio.create_task(execute_checked(index)) for index in indexes]
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
        mode: Mode = Mode.DEFAULT,
        cancel: asyncio.Event | None = None,
    ) -> AsyncIterator[Event]:
        cancel = cancel or asyncio.Event()
        definitions = (
            self._registry.read_only_definitions()
            if mode is Mode.PLAN
            else self._registry.definitions()
        )
        allowed_names = {definition.name for definition in definitions}
        history_calls: list[ToolCall] = []
        slots: list[ToolResult | None] = []
        batch_saved = False
        final_saved = False
        unknown_run = 0
        approval_future: asyncio.Future[Outcome] | None = None
        try:
            environment = await self._environment(cancel)
            system = System(
                stable=prompt.build_system_prompt(),
                environment=environment or "",
            )
            for iteration in range(1, MAX_ITERATIONS + 1):
                if cancel.is_set():
                    conv.add_assistant(NOTICE_CANCELLED)
                    final_saved = True
                    yield Event(notice=NOTICE_CANCELLED)
                    yield Event(done=True)
                    return
                yield Event(iter=iteration)
                state = StreamState()
                reminder = (
                    prompt.plan_reminder((iteration - 1) % PLAN_REMINDER_INTERVAL == 0)
                    if mode is Mode.PLAN
                    else ""
                )
                req = Request(conv.messages(), definitions, system, reminder)
                async for event in self._stream_once(req, cancel, state):
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
                    permitted: list[int] = []
                    for position in group:
                        call = calls[position]
                        decision, reason = self.engine.check(
                            mode,
                            call,
                            self._registry.is_read_only(call.name),
                            registered=self._registry.get(call.name) is not None,
                            allowed=call.name in allowed_names,
                        )
                        if decision is Decision.ASK:
                            approval_future = asyncio.get_running_loop().create_future()
                            yield Event(
                                approval=ApprovalRequest(
                                    self._redact(call.name),
                                    self._redact(call.input),
                                    self._redact(reason),
                                    approval_future,
                                )
                            )
                            outcome = await self._await_approval(approval_future, cancel)
                            approval_future = None
                            if cancel.is_set():
                                break
                            if outcome is Outcome.DENY_ONCE:
                                decision, reason = Decision.DENY, "用户拒绝本次工具调用"
                            else:
                                decision = Decision.ALLOW
                                if outcome is Outcome.ALLOW_FOREVER:
                                    try:
                                        await self._persist_allow(call)
                                    except Exception:
                                        yield Event(
                                            notice="本次已允许，永久规则未保存；后续仍需确认。"
                                        )
                        if decision is Decision.DENY:
                            slots[position] = ToolResult(call.id, self._redact(reason), True)
                        else:
                            permitted.append(position)
                    if cancel.is_set():
                        break
                    completed = await self._run_group(
                        calls, permitted, allowed_names, cancel, slots, mode
                    )
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
            if approval_future is not None and not approval_future.done():
                approval_future.cancel()
            if batch_saved:
                self._finish_batch(conv, history_calls, slots)
            if not final_saved:
                conv.add_assistant("[请求已取消。]")

    @staticmethod
    async def _await_approval(respond: asyncio.Future[Outcome], cancel: asyncio.Event) -> Outcome:
        waiter = asyncio.create_task(cancel.wait())
        try:
            waiting: set[asyncio.Future[Any]] = {respond, waiter}
            await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
            if cancel.is_set():
                return Outcome.DENY_ONCE
            return respond.result()
        finally:
            if not respond.done():
                respond.cancel()
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)

    async def _persist_allow(self, call: ToolCall) -> None:
        task = asyncio.create_task(asyncio.to_thread(self.engine.persist_local_allow, call))
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            # 原子写入线程拥有磁盘变更，必须等其收尾再传播取消。
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not task.cancelled():
                task.exception()
            raise

    async def _environment(self, cancel: asyncio.Event) -> str | None:
        """环境采集也受本轮取消控制，收尾后再允许下一轮运行。"""
        if cancel.is_set():
            return None
        task = asyncio.create_task(
            prompt.gather_environment(self._version, self._provider.model, cwd=self._cwd)
        )
        waiter = asyncio.create_task(cancel.wait())
        try:
            done, _ = await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
            if waiter in done or cancel.is_set():
                return None
            return self._redact(task.result().render())
        finally:
            waiter.cancel()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, waiter, return_exceptions=True)

    @staticmethod
    def _finish_batch(
        conv: Conversation, calls: list[ToolCall], slots: list[ToolResult | None]
    ) -> None:
        results = [
            result if result is not None else ToolResult(call.id, "工具执行已取消，未完成。", True)
            for call, result in zip(calls, slots, strict=True)
        ]
        conv.add_tool_results(results)
