import asyncio
from collections.abc import AsyncIterator
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console, RenderableType
from rich.markdown import Markdown
from textual import events
from textual.containers import ScrollableContainer
from textual.geometry import Size
from textual.widgets import OptionList, RichLog, Static, TextArea

from yincode.config import ProviderConfig
from yincode.llm import Message, StreamEvent
from yincode.tui import SessionState, YinCodeApp
from yincode.tui.view import streaming_block


class FakeProvider:
    """只替代网络边界，流的等待、取消和关闭均实际执行。"""

    def __init__(self, cfg: ProviderConfig) -> None:
        self.name = cfg.name
        self.model = cfg.model
        self.events: asyncio.Queue[StreamEvent | Exception | None] = asyncio.Queue()
        self.requests: list[list[Message]] = []
        self.stream_closed = 0
        self.client_closed = 0
        self.cancelled = False
        self.close_error: Exception | None = None

    async def stream(self, msgs: list[Message]) -> AsyncIterator[StreamEvent]:
        self.requests.append(msgs)
        try:
            while True:
                event = await self.events.get()
                if event is None:
                    return
                if isinstance(event, Exception):
                    raise event
                yield event
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        finally:
            self.stream_closed += 1

    async def aclose(self) -> None:
        self.client_closed += 1
        if self.close_error is not None:
            raise self.close_error


def cfg(name: str = "本地模型", key: str = "unit-test-secret") -> ProviderConfig:
    return ProviderConfig(name=name, protocol="openai", api_key=key, model="test-model")


def render(blocks: list[RenderableType], width: int = 80) -> str:
    buffer = StringIO()
    console = Console(file=buffer, width=width, color_system=None)
    for block in blocks:
        console.print(block)
    return buffer.getvalue()


def static_text(app: YinCodeApp, selector: str) -> str:
    return render([app.query_one(selector, Static).content])  # type: ignore[list-item]


def setup_app(tmp_path: Path) -> tuple[YinCodeApp, FakeProvider]:
    config = cfg()
    fake = FakeProvider(config)
    return YinCodeApp([config], provider_factory=lambda _: fake, cwd=tmp_path), fake


async def wait_for_state(app: YinCodeApp, expected: SessionState) -> None:
    task = app._stream_task
    if task is not None:
        await asyncio.wait_for(asyncio.shield(task), timeout=3)
    assert app.state is expected


def test_wait_indicator_animates_before_the_first_second_or_text() -> None:
    first = render([streaming_block("", 0.0)])
    second = render([streaming_block("", 0.15)])
    assert "Imagining… (0s)" in first and "Imagining… (0s)" in second
    assert first != second


@pytest.mark.asyncio
async def test_single_provider_banner_input_and_status(tmp_path: Path) -> None:
    app, fake = setup_app(tmp_path)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        assert app.state is SessionState.IDLE
        assert app.provider is fake
        assert app.focused is app.query_one("#input", TextArea)
        assert app.query_one("#input", TextArea).placeholder == "Send a message..."
        text = render(app.transcript)
        assert "yincode v0.1.0" in text
        assert str(tmp_path) in text.replace("\n", "")
        assert "snake" in text
        assert len(app.transcript) == 1
        assert "❯" in static_text(app, "#input-prefix")
        status = static_text(app, "#statusbar")
        assert "本地模型" in status and "test-model" in status
    assert fake.client_closed == 1


@pytest.mark.asyncio
async def test_alt_enter_and_enter_send_complete_multiline_context(tmp_path: Path) -> None:
    app, fake = setup_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.press("h", "i", "alt+enter", "x")
        assert app.query_one("#input", TextArea).text == "hi\nx"
        await pilot.press("enter")
        assert app.query_one("#input", TextArea).text == ""
        assert app.state is SessionState.STREAMING
        assert fake.requests[0] == [Message("user", "hi\nx")]
        await fake.events.put(StreamEvent(text="**答案**\n\n- 条目\n\n```python\nprint(1)\n```"))
        await fake.events.put(StreamEvent(done=True))
        await wait_for_state(app, SessionState.IDLE)
        assert app.conv.messages()[-1].role == "assistant"
        assert any(
            isinstance(child, Markdown)
            for block in app.transcript
            for child in getattr(block, "renderables", [])
        )
        completed = render(app.transcript)
        assert "答案" in completed and "print(1)" in completed and "耗时" in completed
        assert "You:" not in completed and "yincode:" not in completed
        await app.submit("追问")
        await pilot.pause()
        assert [message.role for message in fake.requests[1]] == ["user", "assistant", "user"]
        assert fake.stream_closed == 1


@pytest.mark.asyncio
async def test_stream_is_plain_text_timed_and_rejects_concurrent_turns(tmp_path: Path) -> None:
    app, fake = setup_app(tmp_path)
    async with app.run_test() as pilot:
        await app.submit("问题")
        assert "Imagining… (0s)" in static_text(app, "#streaming")
        await fake.events.put(StreamEvent(text="**literal** [red]正文[/red]"))
        await pilot.pause()
        assert "**literal** [red]正文[/red]" in static_text(app, "#streaming")
        app.turn_start -= 2
        await pilot.pause(0.2)
        assert "Imagining… (2s)" in static_text(app, "#streaming")
        await app.submit("不应发送")
        assert app.conv.messages() == [Message("user", "问题")]
        assert len(fake.requests) == 1
        assert not app.query_one("#input", TextArea).disabled
        await fake.events.put(StreamEvent(done=True))
        await wait_for_state(app, SessionState.IDLE)
        assert app.query_one("#streaming", Static).content == ""


@pytest.mark.asyncio
async def test_whitespace_is_ignored_without_a_request(tmp_path: Path) -> None:
    app, fake = setup_app(tmp_path)
    async with app.run_test() as pilot:
        await app.submit(" \n\t ")
        await pilot.pause()
        assert app.state is SessionState.IDLE
        assert app.conv.messages() == []
        assert fake.requests == []
        assert len(app.transcript) == 1


@pytest.mark.asyncio
async def test_selection_uses_arrow_enter_without_creating_unused_clients(tmp_path: Path) -> None:
    configs = [cfg("第一个"), cfg("第二个")]
    created: list[FakeProvider] = []

    def factory(config: ProviderConfig) -> FakeProvider:
        fake = FakeProvider(config)
        created.append(fake)
        return fake

    app = YinCodeApp(configs, provider_factory=factory, cwd=tmp_path)
    async with app.run_test() as pilot:
        assert app.state is SessionState.SELECTING
        assert app.focused is app.query_one("#providers", OptionList)
        assert created == []
        await pilot.press("down", "enter")
        assert app.state is SessionState.IDLE
        assert len(created) == 1 and app.provider is created[0]
        assert created[0].name == "第二个"
        assert "第二个" in static_text(app, "#statusbar")
        assert app.focused is app.query_one("#input", TextArea)
        assert not app.query_one("#providers", OptionList).display
    assert created[0].client_closed == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("via_exception", [False, True])
async def test_errors_are_redacted_red_and_allow_followup(
    tmp_path: Path, via_exception: bool
) -> None:
    config = cfg()
    other = cfg("备用", key="other-test-secret")
    fake = FakeProvider(config)
    app = YinCodeApp([config, other], provider_factory=lambda _: fake, cwd=tmp_path)
    async with app.run_test() as pilot:
        await pilot.press("enter")
        await app.submit("触发错误")
        error = RuntimeError("失败 unit-test-secret other-test-secret [blue]文字[/blue]")
        await fake.events.put(error if via_exception else StreamEvent(err=error))
        await wait_for_state(app, SessionState.IDLE)
        output = render(app.transcript)
        assert "unit-test-secret" not in output and "other-test-secret" not in output
        assert "[REDACTED]" in output
        assert "[blue]文字[/blue]" in output
        assert any("red" in str(getattr(block, "style", "")) for block in app.transcript)
        assert [m.role for m in app.conv.messages()] == ["user", "assistant"]
        assert "失败" in app.conv.messages()[1].content
        await app.submit("继续")
        await fake.events.put(StreamEvent(text="恢复"))
        await fake.events.put(StreamEvent(done=True))
        await wait_for_state(app, SessionState.IDLE)
        assert app.conv.messages()[-1] == Message("assistant", "恢复")
        assert fake.stream_closed == 2


@pytest.mark.asyncio
async def test_unfinished_eof_becomes_error_without_hanging(tmp_path: Path) -> None:
    app, fake = setup_app(tmp_path)
    async with app.run_test():
        await app.submit("请求")
        await fake.events.put(StreamEvent(text="半段"))
        await fake.events.put(None)
        await wait_for_state(app, SessionState.IDLE)
        assert len(app.conv.messages()) == 2
        assert "结束" in render(app.transcript)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["exit", "ctrl+c", "teardown"])
async def test_exit_awaits_stream_cancellation_and_client_close(
    tmp_path: Path, method: str
) -> None:
    app, fake = setup_app(tmp_path)
    async with app.run_test() as pilot:
        await app.submit("慢请求")
        await pilot.pause()
        stream_task = app._stream_task
        assert stream_task is not None
        if method == "exit":
            app.query_one("#input", TextArea).load_text("/exit")
            await pilot.press("enter")
        elif method == "ctrl+c":
            await pilot.press("ctrl+c")
    assert stream_task.done() and stream_task.cancelled()
    assert fake.cancelled and fake.stream_closed == 1
    assert fake.client_closed == 1
    assert app._stream_task is None
    await app.action_quit()
    assert fake.client_closed == 1


@pytest.mark.asyncio
async def test_ctrl_c_also_exits_provider_selection(tmp_path: Path) -> None:
    app = YinCodeApp([cfg("一"), cfg("二")], cwd=tmp_path)
    async with app.run_test() as pilot:
        await pilot.press("ctrl+c")
    assert app.provider is None


@pytest.mark.asyncio
async def test_long_stream_and_resize_preserve_visible_input_and_banner(tmp_path: Path) -> None:
    app, fake = setup_app(tmp_path)
    async with app.run_test(size=(80, 24)) as pilot:
        await app.submit("长回复")
        await fake.events.put(StreamEvent(text="长行 " * 150 + "\n内容" * 80))
        await pilot.pause()
        await pilot.resize_terminal(30, 16)
        await pilot.pause()
        input_widget = app.query_one("#input", TextArea)
        assert input_widget.region.height >= 1
        assert input_widget.region.bottom <= 15
        assert app.query_one("#statusbar", Static).region.bottom <= 16
        assert app.query_one("#log", RichLog).region.height >= 1
        assert app.query_one("#streaming", Static).region.width <= 30
        assert len([block for block in app.transcript if "snake" in render([block])]) == 1


@pytest.mark.asyncio
async def test_small_height_with_multiline_input_does_not_hide_controls(tmp_path: Path) -> None:
    app, fake = setup_app(tmp_path)
    async with app.run_test(size=(30, 12)) as pilot:
        await app.submit("长回复")
        await fake.events.put(StreamEvent(text="输出\n" * 80))
        app.query_one("#input", TextArea).load_text("草稿\n" * 6)
        await pilot.pause()
        assert app.query_one("#input", TextArea).region.bottom <= 11
        assert app.query_one("#statusbar", Static).region.bottom <= 12
        assert app.query_one("#log", RichLog).region.height >= 1


@pytest.mark.asyncio
async def test_client_close_failure_still_exits_without_secret_traceback(tmp_path: Path) -> None:
    app, fake = setup_app(tmp_path)
    fake.close_error = RuntimeError("关闭失败 unit-test-secret")
    async with app.run_test() as pilot:
        await pilot.press("ctrl+c")
    assert not app.is_running
    assert fake.client_closed == 1
    assert "关闭失败" in render(app.transcript)
    assert "unit-test-secret" not in render(app.transcript)


@pytest.mark.asyncio
async def test_provider_construction_failure_is_displayed_without_a_traceback(
    tmp_path: Path,
) -> None:
    def factory(_: ProviderConfig) -> FakeProvider:
        raise RuntimeError("初始化失败 unit-test-secret")

    app = YinCodeApp([cfg()], provider_factory=factory, cwd=tmp_path)
    async with app.run_test() as pilot:
        assert "初始化失败" in render(app.transcript)
        assert "unit-test-secret" not in render(app.transcript)
        assert app.provider is None
        assert app._exception is None
        app.query_one("#input", TextArea).load_text("/exit")
        await pilot.press("enter")


@pytest.mark.asyncio
async def test_cancelling_quit_propagates_instead_of_becoming_a_normal_exit(tmp_path: Path) -> None:
    class SlowCleanupProvider(FakeProvider):
        def __init__(self, config: ProviderConfig) -> None:
            super().__init__(config)
            self.cleanup_started = asyncio.Event()

        async def stream(self, msgs: list[Message]) -> AsyncIterator[StreamEvent]:
            self.requests.append(msgs)
            try:
                await asyncio.Event().wait()
                yield StreamEvent(done=True)
            finally:
                self.cleanup_started.set()
                await asyncio.Event().wait()

    fake = SlowCleanupProvider(cfg())
    app = YinCodeApp([cfg()], provider_factory=lambda _: fake, cwd=tmp_path)
    async with app.run_test() as pilot:
        await app.submit("请求")
        await pilot.pause()
        quit_task = asyncio.create_task(app.action_quit())
        await asyncio.wait_for(fake.cleanup_started.wait(), timeout=2)
        quit_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await quit_task
    assert fake.client_closed == 1


@pytest.mark.asyncio
async def test_teardown_stops_producers_before_widgets_are_removed(tmp_path: Path) -> None:
    class SlowWidgetShutdownApp(YinCodeApp):
        async def _close_all(self) -> None:
            await super()._close_all()
            # 固定放大 Textual 的「子组件已卸载，App Unmount 尚未发出」窗口。
            await asyncio.sleep(0.2)

    fake = FakeProvider(cfg())
    app = SlowWidgetShutdownApp([cfg()], provider_factory=lambda _: fake, cwd=tmp_path)
    async with app.run_test() as pilot:
        await app.submit("未结束的请求")
        await pilot.pause()
    assert app._exception is None
    assert fake.client_closed == 1
    assert fake.stream_closed == 1


@pytest.mark.asyncio
async def test_timer_refresh_preserves_user_scroll_position(tmp_path: Path) -> None:
    app, fake = setup_app(tmp_path)
    async with app.run_test() as pilot:
        await app.submit("长回复")
        await fake.events.put(StreamEvent(text="正文\n" * 100))
        await pilot.pause()
        panel = app.query_one("#streaming-panel", ScrollableContainer)
        assert panel.is_vertical_scroll_end
        panel.scroll_home(animate=False)
        await pilot.pause()
        assert panel.scroll_y == 0
        # 整个等待期间没有新正文，只有持续计时刷新。
        await pilot.pause(0.25)
        assert panel.scroll_y == 0


@pytest.mark.asyncio
async def test_increment_preserves_scroll_and_follow_resumes_at_bottom(tmp_path: Path) -> None:
    app, fake = setup_app(tmp_path)
    async with app.run_test() as pilot:
        await app.submit("长回复")
        await fake.events.put(StreamEvent(text="正文\n" * 100))
        await pilot.pause()
        panel = app.query_one("#streaming-panel", ScrollableContainer)
        panel.scroll_to(y=25, animate=False)
        await pilot.pause()
        assert panel.scroll_y == 25
        await fake.events.put(StreamEvent(text="新增\n" * 30))
        await pilot.pause()
        assert panel.scroll_y == 25
        assert not panel.is_vertical_scroll_end
        panel.scroll_end(animate=False)
        await pilot.pause()
        assert panel.is_vertical_scroll_end
        previous_bottom = panel.scroll_y
        await fake.events.put(StreamEvent(text="继续\n" * 30))
        await pilot.pause()
        assert panel.scroll_y > previous_bottom
        assert panel.is_vertical_scroll_end


@pytest.mark.asyncio
async def test_queued_deltas_follow_the_new_virtual_height(tmp_path: Path) -> None:
    app, fake = setup_app(tmp_path)
    async with app.run_test() as pilot:
        await app.submit("长回复")
        # 多个增量在一次布局刷新之前排队，不能依赖更新前的 max_scroll_y。
        for _ in range(10):
            await fake.events.put(StreamEvent(text="正文\n" * 10))
        await pilot.pause()
        panel = app.query_one("#streaming-panel", ScrollableContainer)
        assert panel.max_scroll_y > 90
        assert panel.is_vertical_scroll_end


@pytest.mark.asyncio
async def test_user_scroll_cancels_a_follow_waiting_for_layout(tmp_path: Path) -> None:
    app, fake = setup_app(tmp_path)
    async with app.run_test() as pilot:
        await app.submit("长回复")
        await fake.events.put(StreamEvent(text="正文\n" * 100))
        await pilot.pause()
        panel = app.query_one("#streaming-panel", ScrollableContainer)
        assert panel.is_vertical_scroll_end
        app.cur_reply += "新增\n" * 30
        app._refresh_streaming_view(follow_output=True)
        # 用户在布局回调执行前离开底部，已经排队的跟随也必须取消。
        panel.scroll_home(animate=False, immediate=True)
        await pilot.pause()
        assert panel.scroll_y == 0


def post_rapid_keys(app: YinCodeApp, *keys: str) -> None:
    """模拟终端单批到达的按键，不在字符之间等待组件处理。"""
    for key in keys:
        app.post_message(events.Key(key, key if len(key) == 1 else None))


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["last", "末尾字"])
async def test_rapid_keys_submit_the_complete_text(tmp_path: Path, text: str) -> None:
    app, fake = setup_app(tmp_path)
    async with app.run_test() as pilot:
        post_rapid_keys(app, *text, "enter")
        await pilot.pause()
        assert app.conv.messages() == [Message("user", text)]
        assert fake.requests == [[Message("user", text)]]
        assert app.query_one("#input", TextArea).text == ""


@pytest.mark.asyncio
async def test_rapid_exit_keys_quit_without_starting_a_request(tmp_path: Path) -> None:
    app, fake = setup_app(tmp_path)
    async with app.run_test() as pilot:
        post_rapid_keys(app, *"/exit", "enter")
        await pilot.pause()
        assert fake.client_closed == 1
        assert app.conv.messages() == []
        assert fake.requests == []


@pytest.mark.asyncio
async def test_rapid_alt_enter_keeps_newline_between_characters(tmp_path: Path) -> None:
    app, fake = setup_app(tmp_path)
    async with app.run_test() as pilot:
        post_rapid_keys(app, "a", "b", "alt+enter", "c", "d")
        await pilot.pause()
        assert app.query_one("#input", TextArea).text == "ab\ncd"
        assert fake.requests == []


@pytest.mark.asyncio
async def test_rapid_multiline_keys_submit_the_complete_ordered_text(tmp_path: Path) -> None:
    app, fake = setup_app(tmp_path)
    async with app.run_test() as pilot:
        post_rapid_keys(app, "a", "b", "alt+enter", "c", "d", "enter")
        await pilot.pause()
        assert app.conv.messages() == [Message("user", "ab\ncd")]
        assert fake.requests == [[Message("user", "ab\ncd")]]
        assert app.query_one("#input", TextArea).text == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", ["las", "/exi"])
async def test_rapid_last_character_is_processed_before_enter(tmp_path: Path, prefix: str) -> None:
    app, fake = setup_app(tmp_path)
    async with app.run_test() as pilot:
        input_widget = app.query_one("#input", TextArea)
        input_widget.load_text(prefix)
        input_widget.move_cursor((0, len(prefix)))
        await pilot.pause()
        post_rapid_keys(app, "t", "enter")
        await pilot.pause()
        if prefix == "/exi":
            assert fake.client_closed == 1
            assert fake.requests == []
            assert app.conv.messages() == []
        else:
            assert fake.requests == [[Message("user", "last")]]
            assert app.conv.messages() == [Message("user", "last")]


@pytest.mark.asyncio
async def test_follow_waits_for_the_actual_virtual_size_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, fake = setup_app(tmp_path)
    async with app.run_test() as pilot:
        await app.submit("长回复")
        await fake.events.put(StreamEvent(text="正文\n" * 100))
        await pilot.pause()
        panel = app.query_one("#streaming-panel", ScrollableContainer)
        assert panel.is_vertical_scroll_end
        previous_bottom = panel.scroll_y
        original_layout = app.screen._refresh_layout

        def hold_layout(size: Size | None = None, scroll: bool = False) -> None:
            pass

        # 控制布局与正文更新分离，让「刷新后回调」先读取旧虚拟尺寸。
        monkeypatch.setattr(app.screen, "_refresh_layout", hold_layout)
        await fake.events.put(StreamEvent(text="新增\n" * 30))
        await pilot.pause()
        assert app.cur_reply.count("\n") == 130
        assert panel.max_scroll_y == previous_bottom
        original_layout()
        await pilot.pause()
        assert panel.max_scroll_y > previous_bottom
        assert panel.is_vertical_scroll_end


@pytest.mark.asyncio
async def test_same_height_increment_does_not_leave_follow_for_window_resize(
    tmp_path: Path,
) -> None:
    app, fake = setup_app(tmp_path)
    async with app.run_test() as pilot:
        await app.submit("长回复")
        await fake.events.put(StreamEvent(text="正文\n" * 100))
        await pilot.pause()
        panel = app.query_one("#streaming-panel", ScrollableContainer)
        streaming = app.query_one("#streaming", Static)
        assert panel.is_vertical_scroll_end
        previous_height = streaming.content_size.height
        previous_bottom = panel.scroll_y
        await fake.events.put(StreamEvent(text="尾"))
        await pilot.pause()
        assert streaming.content_size.height == previous_height
        assert panel.scroll_y == previous_bottom
        await pilot.resize_terminal(80, 18)
        await pilot.pause()
        assert panel.max_scroll_y > previous_bottom
        assert panel.scroll_y == previous_bottom
