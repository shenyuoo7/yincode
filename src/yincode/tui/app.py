"""单会话终端聊天，负责输入、选择及客户端生命周期。"""

import asyncio
import time
from collections.abc import Callable
from enum import Enum, auto
from pathlib import Path

from rich.console import RenderableType
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer
from textual.message import Message
from textual.screen import Screen
from textual.timer import Timer
from textual.widgets import OptionList, RichLog, Static, TextArea

from yincode import __version__
from yincode.config import ProviderConfig, redact
from yincode.conversation import Conversation
from yincode.llm import Provider, new_provider
from yincode.prompt import render_banner

from .select import provider_options
from .stream import consume_stream
from .view import error_block, render_markdown, status_bar, streaming_block, user_block


class SessionState(Enum):
    SELECTING = auto()
    IDLE = auto()
    STREAMING = auto()


class MessageInput(TextArea):
    """Enter 发消息，Alt+Enter 插入换行。"""

    class Submitted(Message):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    def on_key(self, event: events.Key) -> None:
        if event.key not in ("enter", "alt+enter"):
            return
        # 字符与控制键必须在同一组件队列处理，避免优先绑定抢先读取旧正文。
        event.stop()
        event.prevent_default()
        if event.key == "enter":
            self.action_submit_message()
        else:
            self.action_insert_newline()

    def action_submit_message(self) -> None:
        self.post_message(self.Submitted(self.text))

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
    BINDINGS = [Binding("ctrl+c", "quit", "退出", show=False, priority=True)]

    def __init__(
        self,
        providers: list[ProviderConfig],
        *,
        provider_factory: Callable[[ProviderConfig], Provider] | None = None,
        cwd: str | Path | None = None,
    ) -> None:
        if not providers:
            raise ValueError("至少需要一个模型配置")
        super().__init__()
        self.providers = list(providers)
        self._provider_factory = provider_factory or new_provider
        self.cwd = Path(cwd if cwd is not None else Path.cwd()).resolve()
        self.state = SessionState.SELECTING
        self.provider: Provider | None = None
        self.conv = Conversation()
        self.cur_reply = ""
        self.turn_start = 0.0
        self._stream_task: asyncio.Task[None] | None = None
        self._timer: Timer | None = None
        self._pending_stream_follow: float | None = None
        self.transcript: list[RenderableType] = []
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
        self.query_one("#streaming-panel").display = False
        self.transcript.append(Text(render_banner(__version__, str(self.cwd), self.size.width)))
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
        except Exception as error:
            message = redact(str(error), (cfg.api_key for cfg in self.providers))
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
        self.query_one("#statusbar", Static).update(
            status_bar(self.provider.name, self.provider.model)
        )
        self.query_one("#input", MessageInput).focus()
        self.call_after_refresh(self._redraw_history)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if self.state is SessionState.SELECTING and not self._resources_closing:
            self._select_provider(event.option_index)

    async def on_message_input_submitted(self, event: MessageInput.Submitted) -> None:
        await self.submit(event.text)

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        event.text_area.styles.height = min(7, event.text_area.document.line_count + 2)

    async def submit(self, text: str) -> None:
        if text.strip() == "/exit":
            await self.action_quit()
            return
        if not text.strip() or self.state is not SessionState.IDLE or self._resources_closing:
            return
        self.conv.add_user(text)
        self._append_history(user_block(text))
        self.query_one("#input", MessageInput).clear()
        self.cur_reply = ""
        self.turn_start = time.monotonic()
        self.state = SessionState.STREAMING
        self.query_one("#streaming-panel").display = True
        self._refresh_streaming_view()
        self._timer = self.set_interval(0.1, self._tick)
        self._stream_task = asyncio.create_task(self._consume_stream(), name="yincode-stream")

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
        streaming.update(streaming_block(self.cur_reply, time.monotonic() - self.turn_start))

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
        self.conv.add_assistant(reply)
        self._append_history(render_markdown(reply, time.monotonic() - self.turn_start))
        self._finish_turn()

    def _finish_with_error(self, error: Exception) -> None:
        message = redact(str(error), (cfg.api_key for cfg in self.providers))
        self.conv.add_assistant(f"[请求失败：{message}]")
        self._append_history(error_block(message, time.monotonic() - self.turn_start))
        self._finish_turn()

    def _finish_turn(self) -> None:
        self._stop_timer()
        self._pending_stream_follow = None
        self.state = SessionState.IDLE
        self.cur_reply = ""
        self.query_one("#streaming", Static).update("")
        self.query_one("#streaming-panel").display = False
        self.query_one("#input", MessageInput).focus()

    def _stop_timer(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def _append_history(self, block: RenderableType) -> None:
        self.transcript.append(block)
        self.query_one("#log", RichLog).write(block)

    def on_resize(self, event: events.Resize) -> None:
        if self.transcript and not self._resources_closing:
            self.call_after_refresh(self._redraw_history)

    def _redraw_history(self) -> None:
        if not self.transcript or self._resources_closing:
            return
        log = self.query_one("#log", RichLog)
        if not log.display or log.content_size.width == 0:
            return
        # RichLog 保存的是已渲染行；尺寸变化后从原始块重新排版。
        self.transcript[0] = Text(render_banner(__version__, str(self.cwd), log.content_size.width))
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
            if self.provider is not None:
                try:
                    await self.provider.aclose()
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    # 退出后由 CLI 重放，避免关闭失败造成密钥进入 traceback。
                    self.transcript.append(
                        error_block(redact(str(error), (cfg.api_key for cfg in self.providers)))
                    )
            self._resources_closed = True

    async def action_quit(self) -> None:
        await self._close_resources()
        self.exit()

    async def on_unmount(self) -> None:
        await self._close_resources()

    async def _shutdown(self) -> None:
        # Textual 会先卸载组件再发送 App Unmount，必须先停止流与计时器。
        await self._close_resources()
        await super()._shutdown()
