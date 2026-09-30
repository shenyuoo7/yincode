"""单会话终端聊天，负责输入、选择及客户端生命周期。"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import cast

from rich.console import Group, RenderableType
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer
from textual.screen import Screen
from textual.timer import Timer
from textual.widgets import OptionList, RichLog, Static, TextArea

from yincode import __version__
from yincode.agent import Agent, Mode, Phase, ToolEvent
from yincode.config import ProviderConfig, redact
from yincode.conversation import Conversation
from yincode.llm import Provider, new_provider
from yincode.prompt import EXECUTE_DIRECTIVE, render_banner_text
from yincode.tool import Registry, new_default_registry

from .select import provider_options
from .stream import consume_stream
from .view import (
    error_block,
    notice_block,
    render_markdown,
    status_bar,
    streaming_block,
    tool_line,
    tool_result_summary,
    tool_streaming_block,
    user_block,
)


class SessionState(Enum):
    SELECTING = auto()
    IDLE = auto()
    STREAMING = auto()


@dataclass(frozen=True, slots=True)
class ToolDisplay:
    """按调用 id 保存展示参数，同名工具也能准确配对。"""

    tool_call_id: str
    name: str
    args: str


class MessageInput(TextArea):
    """Enter 发消息，Alt+Enter 插入换行。"""

    async def on_key(self, event: events.Key) -> None:
        if event.key not in ("enter", "alt+enter"):
            return
        # 字符与控制键必须在同一组件队列处理，避免优先绑定抢先读取旧正文。
        event.stop()
        event.prevent_default()
        if event.key == "enter":
            await self.action_submit_message()
        else:
            self.action_insert_newline()

    async def action_submit_message(self) -> None:
        # 接受、清空与状态切换必须完成后，才处理下一个草稿字符。
        await cast(YinCodeApp, self.app).submit(self.text)

    def action_insert_newline(self) -> None:
        self.insert("\n")


class YinCodeApp(App[None]):
    CSS = """
    Screen { layout: vertical; }
    #log { width: 1fr; height: 1fr; min-height: 1; }
    #providers { height: 1fr; }
    #streaming-panel {
        width: 1fr; height: 1fr; max-height: 40%; min-height: 1;
        overflow-x: hidden; overflow-y: auto;
    }
    #streaming { width: 1fr; height: auto; }
    #input-wrapper {
        height: auto; border-top: solid $primary;
    }
    #input-prefix { width: 2; height: 1; margin-top: 1; }
    #input {
        width: 1fr; height: 3; min-height: 3; max-height: 7;
        border: none; padding: 1 0; scrollbar-size-horizontal: 0;
    }
    #statusbar { width: 1fr; height: auto; min-height: 1; max-height: 2; }
    """
    BINDINGS = [
        Binding("ctrl+c", "cancel_or_quit", "取消/退出", show=False, priority=True),
        Binding("escape", "cancel_turn", "取消本轮", show=False, priority=True),
    ]

    def __init__(
        self,
        providers: list[ProviderConfig],
        *,
        provider_factory: Callable[[ProviderConfig], Provider] | None = None,
        cwd: str | Path | None = None,
        registry: Registry | None = None,
    ) -> None:
        if not providers:
            raise ValueError("至少需要一个模型配置")
        super().__init__()
        self.providers = list(providers)
        self._provider_factory = provider_factory or new_provider
        self.cwd = Path(cwd if cwd is not None else Path.cwd()).resolve()
        self._tool_registry = (
            registry if registry is not None else new_default_registry(cwd=self.cwd)
        )
        self.state = SessionState.SELECTING
        self.provider: Provider | None = None
        self.agent: Agent | None = None
        self.conv = Conversation()
        self.mode = Mode.NORMAL
        self._has_plan = False
        self.iter = 0
        self.usage_in = 0
        self.usage_out = 0
        self.cur_reply = ""
        self.cur_tools: list[ToolDisplay] = []
        self.turn_cancel: asyncio.Event | None = None
        self.turn_start = 0.0
        self._stream_task: asyncio.Task[None] | None = None
        self._timer: Timer | None = None
        self._pending_stream_follow: float | None = None
        self.transcript: list[RenderableType] = []
        self._history_width = 0
        self._resources_closing = False
        self._resources_closed = False
        self._close_lock = asyncio.Lock()

    def compose(self) -> ComposeResult:
        yield RichLog(id="log", wrap=True, markup=False, min_width=1)
        yield provider_options(self.providers)
        with ScrollableContainer(id="streaming-panel"):
            yield Static("", id="streaming", markup=False)
        with Horizontal(id="input-wrapper"):
            yield Static(Text("❯"), id="input-prefix")
            yield MessageInput(id="input", placeholder="Send a message...")
        yield Static("", id="statusbar", markup=False)

    def on_mount(self) -> None:
        self.screen.screen_layout_refresh_signal.subscribe(
            self, self._follow_stream_after_layout, immediate=True
        )
        self.screen.screen_layout_refresh_signal.subscribe(
            self, self._resize_history_after_layout, immediate=True
        )
        self.query_one("#streaming-panel").display = False
        self.transcript.append(render_banner_text(__version__, str(self.cwd), self.size.width))
        if len(self.providers) == 1:
            self._select_provider(0)
        else:
            self.query_one("#log").display = False
            self.query_one("#input-wrapper").display = False
            self.query_one("#statusbar", Static).update("选择模型：↑ / ↓ 移动，Enter 确认")
            self.query_one("#providers", OptionList).focus()
        self.call_after_refresh(self._redraw_history)

    def _select_provider(self, index: int) -> None:
        try:
            self.provider = self._provider_factory(self.providers[index])
            self.agent = Agent(
                self.provider,
                self._tool_registry,
                __version__,
                cwd=self.cwd,
                redactor=self._redact,
                secrets=tuple(cfg.api_key for cfg in self.providers),
            )
        except Exception as error:
            message = self._redact(str(error))
            self.query_one("#log").display = True
            self._append_history(error_block("模型初始化失败：" + message))
            self.query_one("#statusbar", Static).update("请检查配置或选择其他模型，Ctrl+C 退出")
            if len(self.providers) == 1:
                self.query_one("#providers").display = False
                self.query_one("#input-wrapper").display = True
                self.query_one("#input", MessageInput).focus()
            self.call_after_refresh(self._redraw_history)
            return
        self.state = SessionState.IDLE
        self.query_one("#providers").display = False
        self.query_one("#log").display = True
        self.query_one("#input-wrapper").display = True
        self._refresh_status()
        self.query_one("#input", MessageInput).focus()
        self.call_after_refresh(self._redraw_history)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if self.state is SessionState.SELECTING and not self._resources_closing:
            self._select_provider(event.option_index)

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        event.text_area.styles.height = min(7, event.text_area.document.line_count + 2)

    async def submit(self, text: str) -> None:
        command = text.strip()
        if command == "/exit":
            await self.action_quit()
            return
        if not command or self.state is not SessionState.IDLE or self._resources_closing:
            return
        if command == "/plan":
            self.mode = Mode.PLAN
            self._has_plan = False
            self.query_one("#input", MessageInput).clear()
            self._append_history(notice_block("已进入计划模式（只读工具）"))
            self._refresh_status()
            return
        if command == "/do":
            if not self._has_plan:
                self.query_one("#input", MessageInput).clear()
                self._append_history(notice_block("请先在 /plan 模式中生成计划"))
                return
            self.mode = Mode.NORMAL
            self._has_plan = False
            self.conv.add_user(EXECUTE_DIRECTIVE)
        else:
            self.conv.add_user(text)
        self._append_history(user_block(text))
        self.query_one("#input", MessageInput).clear()
        self.cur_reply = ""
        self.cur_tools.clear()
        self.iter = 0
        self.turn_cancel = asyncio.Event()
        self._refresh_status()
        self.turn_start = time.monotonic()
        self.state = SessionState.STREAMING
        self.query_one("#streaming-panel").display = True
        self._refresh_streaming_view()
        self._timer = self.set_interval(0.1, self._tick)
        self._stream_task = asyncio.create_task(self._consume_stream(), name="yincode-stream")

    def _redact(self, text: str) -> str:
        return redact(text, (cfg.api_key for cfg in self.providers))

    async def _consume_stream(self) -> None:
        await consume_stream(self)

    def _tick(self) -> None:
        if self.state is SessionState.STREAMING and not self._resources_closing:
            self._refresh_streaming_view()

    def _refresh_streaming_view(self, *, follow_output: bool = False) -> None:
        panel = self.query_one("#streaming-panel", ScrollableContainer)
        was_at_end = panel.is_vertical_scroll_end
        previous_scroll = panel.scroll_y
        if self._pending_stream_follow != previous_scroll:
            self._pending_stream_follow = None
        if follow_output and was_at_end:
            self._pending_stream_follow = previous_scroll
        streaming = self.query_one("#streaming", Static)
        elapsed = time.monotonic() - self.turn_start
        block: RenderableType
        if self.cur_tools:
            block = Group(
                *(tool_streaming_block(tool.name, tool.args, elapsed) for tool in self.cur_tools)
            )
        else:
            block = streaming_block(self.cur_reply, elapsed, self.iter)
        streaming.update(block)

    def _handle_tool_event(self, event: ToolEvent) -> None:
        if event.phase is Phase.START:
            if self.cur_reply:
                self._append_history(render_markdown(self._redact(self.cur_reply)))
                self.cur_reply = ""
            display = ToolDisplay(
                event.tool_call_id, self._redact(event.name), self._redact(event.args)
            )
            self.cur_tools.append(display)
        else:
            finished_display = next(
                (tool for tool in self.cur_tools if tool.tool_call_id == event.tool_call_id), None
            )
            if finished_display is None:
                finished_display = ToolDisplay(
                    event.tool_call_id, self._redact(event.name), self._redact(event.args)
                )
            else:
                self.cur_tools.remove(finished_display)
            self._append_history(tool_line(finished_display.name, finished_display.args))
            self._append_history(tool_result_summary(self._redact(event.result), event.is_error))
        self._pending_stream_follow = None
        self._refresh_streaming_view(follow_output=True)

    def _follow_stream_after_layout(self, screen: Screen) -> None:
        previous_scroll = self._pending_stream_follow
        if (
            previous_scroll is None
            or self.state is not SessionState.STREAMING
            or self._resources_closing
        ):
            self._pending_stream_follow = None
            return
        panel = screen.query_one("#streaming-panel", ScrollableContainer)
        if panel.scroll_y != previous_scroll:
            self._pending_stream_follow = None
            return
        streaming = screen.query_one("#streaming", Static)
        size = streaming.content_size
        # 滚动重排也会发信号；正文高度尚未应用时，继续等它自己的布局。
        if (
            not size.width
            or streaming.get_content_height(panel.size, screen.size, size.width) != size.height
        ):
            return
        # 同高度正文也消费一次，避免意图遗留到之后的普通窗口缩放。
        self._pending_stream_follow = None
        panel.scroll_end(animate=False, immediate=True, x_axis=False)

    def _finish_with_assistant(self, reply: str) -> None:
        if (
            self.mode is Mode.PLAN
            and reply.strip()
            and not (self.turn_cancel is not None and self.turn_cancel.is_set())
        ):
            self._has_plan = True
        self._append_history(
            render_markdown(self._redact(reply), time.monotonic() - self.turn_start)
        )
        self._finish_turn()

    def _finish_with_error(self, error: Exception) -> None:
        message = self._redact(str(error))
        self._append_history(error_block(message, time.monotonic() - self.turn_start))
        self._finish_turn()

    def _finish_turn(self) -> None:
        self._stop_timer()
        self._pending_stream_follow = None
        self.state = SessionState.IDLE
        self.cur_reply = ""
        self.cur_tools.clear()
        self.iter = 0
        self.turn_cancel = None
        self._stream_task = None
        self.query_one("#streaming", Static).update("")
        self.query_one("#streaming-panel").display = False
        self.query_one("#input", MessageInput).focus()

    def _stop_timer(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def _refresh_status(self) -> None:
        if self.provider is not None:
            self.query_one("#statusbar", Static).update(
                status_bar(
                    self.provider.name,
                    self.provider.model,
                    plan=self.mode is Mode.PLAN,
                    usage_in=self.usage_in,
                    usage_out=self.usage_out,
                )
            )

    def _append_history(self, block: RenderableType) -> None:
        self.transcript.append(block)
        self.query_one("#log", RichLog).write(block)

    def _resize_history_after_layout(self, screen: Screen) -> None:
        if self._resources_closing:
            return
        # App 的 Resize 消息可能早于子组件重排，等日志区实际宽度应用后再重画。
        log = screen.query_one("#log", RichLog)
        if log.content_size.width != self._history_width:
            self._redraw_history()

    def _redraw_history(self) -> None:
        if not self.transcript or self._resources_closing:
            return
        log = self.query_one("#log", RichLog)
        if not log.display or log.content_size.width == 0:
            return
        self._history_width = log.content_size.width
        # RichLog 保存的是已渲染行；尺寸变化后从原始块重新排版。
        self.transcript[0] = render_banner_text(__version__, str(self.cwd), log.content_size.width)
        log.clear()
        for block in self.transcript:
            log.write(block)

    async def _close_resources(self) -> None:
        async with self._close_lock:
            if self._resources_closed:
                return
            self._resources_closing = True
            self._pending_stream_follow = None
            self._stop_timer()
            task = self._stream_task
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    # 只接收子任务的取消；退出调用方自身被取消时仍要传播。
                    current = asyncio.current_task()
                    if current is not None and current.cancelling():
                        raise
                finally:
                    self._stream_task = None
            self.cur_tools.clear()
            self.turn_cancel = None
            if self.provider is not None:
                try:
                    await self.provider.aclose()
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    # 退出后由 CLI 重放，避免关闭失败造成密钥进入 traceback。
                    self.transcript.append(error_block(self._redact(str(error))))
            self._resources_closed = True

    async def action_quit(self) -> None:
        await self._close_resources()
        self.exit()

    async def action_cancel_or_quit(self) -> None:
        if self.state is SessionState.STREAMING:
            self.action_cancel_turn()
        else:
            await self.action_quit()

    def action_cancel_turn(self) -> None:
        if self.state is SessionState.STREAMING and self.turn_cancel is not None:
            self.turn_cancel.set()

    async def on_unmount(self) -> None:
        await self._close_resources()

    async def _shutdown(self) -> None:
        # Textual 会先卸载组件再发送 App Unmount，必须先停止流与计时器。
        await self._close_resources()
        await super()._shutdown()
