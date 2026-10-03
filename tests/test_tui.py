import asyncio
import json
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
from yincode.llm import Message, Request, StreamEvent, ToolCall, ToolDefinition, Usage
from yincode.permission import Mode
from yincode.prompt import EXECUTE_DIRECTIVE, build_system_prompt, plan_reminder, render_banner_text
from yincode.tui import SessionState, YinCodeApp, view
from yincode.tui.view import streaming_block


@pytest.fixture(autouse=True)
def deterministic_ui_environment(monkeypatch):
    """界面测试隔离 Git 外调时延；真实采集由环境与 SDK 集成测试覆盖。"""
    from yincode import prompt
    from yincode.prompt.environment import Environment

    async def gather(version, model, *, cwd=None):
        return Environment(str(cwd), "win32", "2026-10-01", "", version, model)

    monkeypatch.setattr(prompt, "gather_environment", gather)


def test_crlf_tool_output_has_clean_lines_and_lone_carriage_is_visible():
    summary = view.tool_result_summary("first\r\nsecond\r\n", False)
    assert "first\n    second" in summary.plain
    assert "\\x0d" not in summary.plain
    assert "\\x0d" in view.tool_result_summary("before\rafter", False).plain


class FakeProvider:
    """只替代网络边界，流的等待、取消和关闭均实际执行。"""

    def __init__(self, cfg: ProviderConfig) -> None:
        self.name = cfg.name
        self.model = cfg.model
        self.events: asyncio.Queue[StreamEvent | Exception | None] = asyncio.Queue()
        self.requests: list[list[Message]] = []
        self.tool_requests: list[list[ToolDefinition]] = []
        self.model_requests: list[Request] = []
        self.stream_closed = 0
        self.client_closed = 0
        self.cancelled = False
        self.close_error: Exception | None = None

    async def stream(self, req: Request) -> AsyncIterator[StreamEvent]:
        self.requests.append(req.messages)
        self.tool_requests.append(req.tools)
        self.model_requests.append(req)
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


async def test_permission_modes_cycle_and_persist_across_turns(tmp_path):
    app, fake = setup_app(tmp_path)
    async with app.run_test() as pilot:
        assert app.mode is Mode.DEFAULT
        for mode, label in [
            (Mode.ACCEPT_EDITS, "ACCEPT EDITS"),
            (Mode.PLAN, "PLAN"),
            (Mode.BYPASS, "BYPASS"),
            (Mode.DEFAULT, "DEFAULT"),
        ]:
            await pilot.press("shift+tab")
            assert app.mode is mode
            assert label in static_text(app, "#statusbar")
        await pilot.press("shift+tab")
        await app.submit("hello")
        await fake.events.put(StreamEvent(text="ok"))
        await fake.events.put(StreamEvent(done=True))
        await wait_for_state(app, SessionState.IDLE)
        assert app.mode is Mode.ACCEPT_EDITS


async def queue_approval(app, fake, pilot):
    await app.submit("写入文件")
    await fake.events.put(
        StreamEvent(
            tool_calls=[
                ToolCall(
                    "w",
                    "write_file",
                    '{"path":"approved.txt","content":"unit-test-secret fixture"}',
                )
            ]
        )
    )
    await fake.events.put(StreamEvent(done=True))
    for _ in range(30):
        await pilot.pause(0.01)
        if app.state is SessionState.APPROVING:
            return
    raise AssertionError("未进入审批状态")


@pytest.mark.parametrize(
    "keys,exists,permanent",
    [
        (["1"], True, False),
        (["down", "enter"], True, True),
        (["3"], False, False),
        (["up", "enter"], False, False),
        (["down", "down", "up", "2"], True, True),
    ],
)
async def test_permission_approval_menu_controls_real_write(tmp_path, keys, exists, permanent):
    app, fake = setup_app(tmp_path)
    async with app.run_test(size=(80, 30)) as pilot:
        await queue_approval(app, fake, pilot)
        assert app.approve_cursor == 0
        display = static_text(app, "#streaming")
        assert "允许本次" in display and "永久允许" in display and "拒绝本次" in display
        assert "unit-test-secret" not in display
        await pilot.press("shift+tab")
        assert app.mode is Mode.DEFAULT
        await pilot.press(*keys)
        await fake.events.put(StreamEvent(text="完成"))
        await fake.events.put(StreamEvent(done=True))
        await wait_for_state(app, SessionState.IDLE)
        assert (tmp_path / "approved.txt").exists() is exists
        local_path = Path(app.engine.local_path)
        assert await asyncio.to_thread(local_path.exists) is permanent
        assert app.pending is None
        assert app.query_one("#input", TextArea).text == ""


@pytest.mark.parametrize("key", ["escape", "ctrl+c"])
async def test_permission_approval_cancel_recovers_without_exit(tmp_path, key):
    from yincode.permission import Outcome

    app, fake = setup_app(tmp_path)
    async with app.run_test() as pilot:
        await queue_approval(app, fake, pilot)
        pending = app.pending
        await pilot.press(key)
        await wait_for_state(app, SessionState.IDLE)
        assert pending.respond.result() is Outcome.DENY_ONCE
        assert not (tmp_path / "approved.txt").exists()
        assert not app._resources_closing
        await app.submit("继续")
        await fake.events.put(StreamEvent(text="可以继续"))
        await fake.events.put(StreamEvent(done=True))
        await wait_for_state(app, SessionState.IDLE)
        assert app.conv.messages()[-1].content == "可以继续"


async def test_permission_start_mode_plan_applies_reminder(tmp_path):
    (tmp_path / ".yincode").mkdir()
    (tmp_path / ".yincode/permissions.local.yaml").write_text("default_mode: plan")
    app, fake = setup_app(tmp_path)
    async with app.run_test():
        assert app.mode is Mode.PLAN
        await app.submit("调研")
        await fake.events.put(StreamEvent(text="计划"))
        await fake.events.put(StreamEvent(done=True))
        await wait_for_state(app, SessionState.IDLE)
        assert fake.model_requests[0].reminder
        assert {definition.name for definition in fake.tool_requests[0]} == {
            "read_file",
            "glob",
            "grep",
        }


@pytest.mark.asyncio
async def test_plan_requires_a_completed_plan_before_do_starts_execution(tmp_path: Path) -> None:
    app, fake = setup_app(tmp_path)
    async with app.run_test() as pilot:
        await app.submit("/do")
        assert fake.requests == [] and app.conv.messages() == []
        await app.submit("/plan")
        assert app.mode is Mode.PLAN
        assert "PLAN" in static_text(app, "#statusbar")
        await app.submit("/do")
        assert fake.requests == [] and app.mode is Mode.PLAN
        await app.submit("制定方案")
        await pilot.pause()
        assert fake.model_requests[0].reminder == plan_reminder(True)
        assert fake.model_requests[0].system.stable == build_system_prompt()
        assert str(tmp_path) in fake.model_requests[0].system.environment
        assert {tool.name for tool in fake.tool_requests[0]} == {
            "read_file",
            "glob",
            "grep",
        }
        await fake.events.put(StreamEvent(text="先阅读，再修改"))
        await fake.events.put(StreamEvent(done=True))
        await wait_for_state(app, SessionState.IDLE)
        await app.submit("/do")
        await pilot.pause()
        assert app.mode is Mode.DEFAULT
        assert "PLAN" not in static_text(app, "#statusbar")
        assert app.conv.messages()[-1] == Message("user", EXECUTE_DIRECTIVE)
        assert fake.model_requests[-1].reminder == ""
        assert fake.model_requests[-1].system.stable == fake.model_requests[0].system.stable
        assert len(fake.tool_requests[-1]) == 6
        await fake.events.put(StreamEvent(text="已执行"))
        await fake.events.put(StreamEvent(done=True))
        await wait_for_state(app, SessionState.IDLE)


@pytest.mark.asyncio
async def test_usage_and_iteration_are_visible_and_accumulate_across_turns(tmp_path: Path) -> None:
    app, fake = setup_app(tmp_path)
    async with app.run_test() as pilot:
        await app.submit("第一轮")
        await pilot.pause()
        assert "第 1 轮" in static_text(app, "#streaming")
        await fake.events.put(StreamEvent(usage=Usage(1234, 56)))
        await fake.events.put(StreamEvent(text="回复"))
        await pilot.pause()
        assert "↑1.2k ↓56 tok" in static_text(app, "#statusbar")
        await fake.events.put(StreamEvent(done=True))
        await wait_for_state(app, SessionState.IDLE)
        assert app.iter == 0 and app.usage_in == 1234 and app.usage_out == 56
        await app.submit("第二轮")
        await fake.events.put(StreamEvent(usage=Usage(10, 4)))
        await fake.events.put(StreamEvent(text="完成"))
        await fake.events.put(StreamEvent(done=True))
        await wait_for_state(app, SessionState.IDLE)
        assert app.usage_in == 1244 and app.usage_out == 60
        assert "↑1.2k ↓60 tok" in static_text(app, "#statusbar")


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["escape", "ctrl+c"])
async def test_turn_cancel_returns_idle_and_next_turn_can_continue(
    tmp_path: Path, key: str
) -> None:
    app, fake = setup_app(tmp_path)
    async with app.run_test() as pilot:
        await app.submit("等待中")
        await pilot.pause()
        current_cancel = app.turn_cancel
        assert current_cancel is not None
        await pilot.press(key)
        await wait_for_state(app, SessionState.IDLE)
        assert current_cancel.is_set()
        assert app.turn_cancel is None and app.is_running
        assert app.conv.messages()[-1].role == "assistant"
        await app.submit("继续")
        await fake.events.put(StreamEvent(text="恢复成功"))
        await fake.events.put(StreamEvent(done=True))
        await wait_for_state(app, SessionState.IDLE)
        assert app.conv.messages()[-1] == Message("assistant", "恢复成功")


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
        assert len(app.transcript) == 1
        assert "❯" in static_text(app, "#input-prefix")
        status = static_text(app, "#statusbar")
        assert "DEFAULT" in status and "本地模型" not in status and "test-model" in status
    assert fake.client_closed == 1


@pytest.mark.asyncio
async def test_banner_follows_actual_width_when_shrinking_and_expanding(tmp_path: Path) -> None:
    app, _ = setup_app(tmp_path)
    async with app.run_test(size=(100, 30)) as pilot:
        for width in (30, 80, 24, 100):
            await pilot.resize_terminal(width, 24)
            await pilot.pause()
            log = app.query_one("#log", RichLog)
            expected = render_banner_text("0.1.0", str(tmp_path), log.content_size.width)
            assert len(app.transcript) == 1
            assert app.transcript[0] == expected


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
        assert "Imagining… (2s · 第 1 轮)" in static_text(app, "#streaming")
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
        assert "DEFAULT" in static_text(app, "#statusbar")
        assert "第二个" not in static_text(app, "#statusbar")
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
@pytest.mark.parametrize("method", ["exit", "teardown"])
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
        assert len([block for block in app.transcript if "yincode v" in render([block])]) == 1


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

        async def stream(self, req: Request) -> AsyncIterator[StreamEvent]:
            self.requests.append(req.messages)
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


@pytest.mark.asyncio
async def test_rapid_submit_clears_before_following_draft_characters(tmp_path: Path) -> None:
    app, fake = setup_app(tmp_path)
    async with app.run_test() as pilot:
        post_rapid_keys(app, "a", "enter", "b")
        await pilot.pause()
        assert app.conv.messages() == [Message("user", "a")]
        assert fake.requests == [[Message("user", "a")]]
        assert app.query_one("#input", TextArea).text == "b"


@pytest.mark.asyncio
async def test_rapid_streaming_rejection_preserves_the_new_draft(tmp_path: Path) -> None:
    app, fake = setup_app(tmp_path)
    async with app.run_test() as pilot:
        post_rapid_keys(app, "a", "enter", "b", "enter", "c")
        await pilot.pause()
        assert app.state is SessionState.STREAMING
        assert app.conv.messages() == [Message("user", "a")]
        assert fake.requests == [[Message("user", "a")]]
        assert app.query_one("#input", TextArea).text == "bc"


@pytest.mark.asyncio
async def test_rapid_blank_rejection_preserves_following_draft(tmp_path: Path) -> None:
    app, fake = setup_app(tmp_path)
    async with app.run_test() as pilot:
        post_rapid_keys(app, " ", "enter", "b")
        await pilot.pause()
        assert app.state is SessionState.IDLE
        assert app.conv.messages() == []
        assert fake.requests == []
        assert app.query_one("#input", TextArea).text == " b"


@pytest.mark.parametrize(
    "result", ["\n".join(f"第{i}行" for i in range(20)), "蛇" * 10000], ids=["lines", "bytes"]
)
def test_tool_summary_has_eight_line_and_utf8_byte_limits(result: str) -> None:
    from yincode.tui.view import tool_result_summary

    block = tool_result_summary(result, False)
    text = block.plain
    assert len(text.splitlines()) <= 8
    assert len(text.encode("utf-8")) <= 4096
    assert "[truncated]" in text
    assert "�" not in text


class WaitingTool:
    """执行真实异步等待，让 Pilot 验证界面仍消费输入与刷新。"""

    name = "read_file"
    description = "读取内存夹具"
    parameters = {"type": "object", "properties": {"path": {"type": "string"}}}
    read_only = True

    def __init__(self, result: str | None = None, is_error: bool = False) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cleaned = False
        self.result = result
        self.is_error = is_error

    async def execute(self, args: str):
        from yincode.tool import Result

        self.started.set()
        try:
            await self.release.wait()
            return Result(self.result or f"已读取 {json.loads(args)['path']}", self.is_error)
        finally:
            self.cleaned = True


@pytest.mark.asyncio
async def test_concurrent_tools_have_two_running_rows_then_ordered_history(tmp_path: Path) -> None:
    from yincode.tool import Registry

    tool = WaitingTool()
    registry = Registry()
    registry.register(tool)
    fake = FakeProvider(cfg())
    app = YinCodeApp([cfg()], provider_factory=lambda _: fake, cwd=tmp_path, registry=registry)
    async with app.run_test() as pilot:
        await app.submit("读取两个文件")
        await fake.events.put(
            StreamEvent(
                tool_calls=[
                    ToolCall("first", "read_file", '{"path":"one.txt"}'),
                    ToolCall("second", "read_file", '{"path":"two.txt"}'),
                ]
            )
        )
        await fake.events.put(StreamEvent(done=True))
        await asyncio.wait_for(tool.started.wait(), 2)
        await pilot.pause()
        dynamic = static_text(app, "#streaming")
        assert dynamic.count("Running…") == 2
        assert dynamic.index("one.txt") < dynamic.index("two.txt")
        assert [display.tool_call_id for display in app.cur_tools] == ["first", "second"]
        tool.release.set()
        await fake.events.put(StreamEvent(text="已完成"))
        await fake.events.put(StreamEvent(done=True))
        await wait_for_state(app, SessionState.IDLE)
        history = render(app.transcript)
        assert history.index("one.txt") < history.index("two.txt") < history.index("已完成")
        assert app.cur_tools == []


@pytest.mark.asyncio
async def test_slow_tool_keeps_input_and_running_timer_responsive(tmp_path: Path) -> None:
    from yincode.tool import Registry

    tool = WaitingTool()
    registry = Registry()
    registry.register(tool)
    fake = FakeProvider(cfg())
    app = YinCodeApp([cfg()], provider_factory=lambda _: fake, cwd=tmp_path, registry=registry)
    async with app.run_test() as pilot:
        await app.submit("读取")
        await fake.events.put(
            StreamEvent(tool_calls=[ToolCall("read-1", "read_file", '{"path":"first.txt"}')])
        )
        await fake.events.put(StreamEvent(done=True))
        await asyncio.wait_for(tool.started.wait(), 2)
        await pilot.pause()
        first = static_text(app, "#streaming")
        assert "read_file" in first and "first.txt" in first and "Running… (0s)" in first
        app.turn_start -= 2
        await pilot.press("草", "稿", "enter", "续")
        await pilot.pause(0.2)
        assert "Running… (2s)" in static_text(app, "#streaming")
        assert app.query_one("#input", TextArea).text == "草稿续"
        assert len(fake.requests) == 1
        tool.release.set()
        await fake.events.put(StreamEvent(text="最终答复"))
        await fake.events.put(StreamEvent(done=True))
        await wait_for_state(app, SessionState.IDLE)
        assert app.query_one("#input", TextArea).text == "草稿续"
        assert app.cur_tools == []
        assert tool.cleaned


@pytest.mark.asyncio
async def test_two_same_named_tools_replay_preamble_results_then_final_once(tmp_path: Path) -> None:
    from yincode.tool import Registry

    tool = WaitingTool()
    tool.release.set()
    registry = Registry()
    registry.register(tool)
    fake = FakeProvider(cfg())
    app = YinCodeApp([cfg()], provider_factory=lambda _: fake, cwd=tmp_path, registry=registry)
    async with app.run_test():
        await app.submit("读两个文件")
        for event in (
            StreamEvent(text="先读取两个文件"),
            StreamEvent(
                tool_calls=[
                    ToolCall("read-a", "read_file", '{"path":"first.txt"}'),
                    ToolCall("read-b", "read_file", '{"path":"second.txt"}'),
                ]
            ),
            StreamEvent(done=True),
            StreamEvent(text="最终总结"),
            StreamEvent(done=True),
        ):
            await fake.events.put(event)
        await wait_for_state(app, SessionState.IDLE)
        output = render(app.transcript)
        first_row = output.index("read_file(")
        second_row = output.index("read_file(", first_row + 1)
        assert output.index("先读取两个文件") < first_row < output.index("已读取 first.txt")
        assert output.index("已读取 first.txt") < second_row
        assert output.index("已读取 second.txt") < output.index("最终总结")
        assert output.count("耗时：") == 1
        assert output.count("最终总结") == 1
        assert [message.role for message in app.conv.messages()] == [
            "user",
            "assistant",
            "tool",
            "assistant",
        ]
        assert [result.tool_call_id for result in app.conv.messages()[2].tool_results] == [
            "read-a",
            "read-b",
        ]
    assert "已读取 first.txt" in render(app.transcript)
    assert "已读取 second.txt" in render(app.transcript)


@pytest.mark.asyncio
async def test_secret_split_across_text_deltas_is_hidden_in_dynamic_and_history(
    tmp_path: Path,
) -> None:
    app, fake = setup_app(tmp_path)
    async with app.run_test() as pilot:
        await app.submit("密钥保护")
        await fake.events.put(StreamEvent(text="回复 unit-test-"))
        await fake.events.put(StreamEvent(text="secret"))
        await pilot.pause()
        assert "unit-test-secret" not in static_text(app, "#streaming")
        assert "[REDACTED]" in static_text(app, "#streaming")
        await fake.events.put(StreamEvent(done=True))
        await wait_for_state(app, SessionState.IDLE)
        assert "unit-test-secret" not in render(app.transcript)
        assert app.conv.messages()[-1].content == "回复 [REDACTED]"


@pytest.mark.asyncio
async def test_tool_errors_and_all_output_are_redacted_with_red_summary(tmp_path: Path) -> None:
    from yincode.tool import Registry

    tool = WaitingTool("读取失败 unit-test-secret other-test-secret", is_error=True)
    tool.release.set()
    registry = Registry()
    registry.register(tool)
    fake = FakeProvider(cfg())
    app = YinCodeApp(
        [cfg(), cfg("备用", "other-test-secret")],
        provider_factory=lambda _: fake,
        cwd=tmp_path,
        registry=registry,
    )
    async with app.run_test() as pilot:
        await pilot.press("enter")
        await app.submit("读取错误")
        for event in (
            StreamEvent(text="前言 unit-test-secret"),
            StreamEvent(tool_calls=[ToolCall("bad", "read_file", '{"path":"other-test-secret"}')]),
            StreamEvent(done=True),
            StreamEvent(text="结语 other-test-secret"),
            StreamEvent(done=True),
        ):
            await fake.events.put(event)
        await wait_for_state(app, SessionState.IDLE)
        output = render(app.transcript)
        assert "unit-test-secret" not in output and "other-test-secret" not in output
        assert "[REDACTED]" in output
        assert any(
            "读取失败" in render([block]) and "red" in str(getattr(block, "style", ""))
            for block in app.transcript
        )
        history = str(app.conv.messages())
        assert "unit-test-secret" not in history and "other-test-secret" not in history


@pytest.mark.asyncio
async def test_quit_cancels_active_tool_and_pairs_the_entire_call_batch(tmp_path: Path) -> None:
    from yincode.tool import Registry

    tool = WaitingTool()
    registry = Registry()
    registry.register(tool)
    fake = FakeProvider(cfg())
    app = YinCodeApp([cfg()], provider_factory=lambda _: fake, cwd=tmp_path, registry=registry)
    async with app.run_test():
        await app.submit("慢工具")
        await fake.events.put(
            StreamEvent(
                tool_calls=[
                    ToolCall("active", "read_file", '{"path":"first.txt"}'),
                    ToolCall("pending", "read_file", '{"path":"second.txt"}'),
                ]
            )
        )
        await fake.events.put(StreamEvent(done=True))
        await asyncio.wait_for(tool.started.wait(), 2)
        task = app._stream_task
        await app.action_quit()
    assert task is not None and task.done() and task.cancelled()
    assert tool.cleaned and fake.stream_closed == 1 and fake.client_closed == 1
    messages = app.conv.messages()
    results = [result for message in messages for result in message.tool_results]
    assert [result.tool_call_id for result in results] == ["active", "pending"]
    assert all(result.is_error for result in results)
    assert messages[-1].role == "assistant"
    assert app._exception is None


@pytest.mark.parametrize(
    "view_kind",
    [
        "tool_name",
        "tool_args",
        "result",
        "running",
        "streaming",
        "markdown",
        "error",
        "user",
        "status",
    ],
)
@pytest.mark.parametrize(
    ("untrusted", "visible"),
    [
        ("before\x1b]52;c;Y2xpcGJvYXJk\x1b\\after", r"\x1b]52;c;Y2xpcGJvYXJk\x1b\after"),
        ("before\x1b[2Jafter", r"\x1b[2J"),
        (
            "before\x00\x07\x08\r\v\f\x1f\x7f\x80\x85\x9b\x9fafter",
            r"\x00\x07\x08\x0d\x0b\x0c\x1f\x7f\x80\x85\x9b\x9f",
        ),
    ],
    ids=["clipboard", "clear-screen", "ascii-and-c1"],
)
def test_console_replay_escapes_untrusted_terminal_controls(
    view_kind: str, untrusted: str, visible: str
) -> None:
    blocks = {
        "tool_name": lambda: view.tool_line(untrusted, "args"),
        "tool_args": lambda: view.tool_line("bash", untrusted),
        "result": lambda: view.tool_result_summary(untrusted, False),
        "running": lambda: view.tool_streaming_block("bash", untrusted, 0.2),
        "streaming": lambda: view.streaming_block(untrusted, 0.2),
        "markdown": lambda: view.render_markdown(untrusted, 0.2),
        "error": lambda: view.error_block(untrusted, 0.2),
        "user": lambda: view.user_block(untrusted),
        "status": lambda: view.status_bar(Mode.DEFAULT, untrusted),
    }
    output = render([blocks[view_kind]()], width=200)
    assert all(
        char in "\n\t" or not (ord(char) < 32 or 127 <= ord(char) <= 159) for char in output
    ), repr(output)
    assert visible in output
    assert "before" in output and "after" in output


def test_terminal_escaping_preserves_unicode_newlines_tabs_and_rich_styles() -> None:
    block = view.user_block("蛇🐍\n\t缩进")
    assert block.plain == "● 蛇🐍\n\t缩进"
    assert "蛇🐍" in render([block]) and "缩进" in render([block])
    error = view.error_block("错误\x1b[2J")
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=True, color_system="standard", no_color=False)
    console.print(error)
    output = buffer.getvalue()
    assert "\x1b[1;31m" in output
    assert r"\x1b[2J" in output and "\x1b[2J" not in output


def test_tool_display_limits_count_escaped_control_bytes() -> None:
    line = view.tool_line("bash", "\x1b" * 500)
    summary = view.tool_result_summary("\x1b" * 4000, False)
    assert "\x1b" not in line.plain + summary.plain
    assert len(line.plain.encode("utf-8")) <= 512
    assert len(summary.plain.encode("utf-8")) <= 4096
    assert len(summary.plain.splitlines()) <= 8
    assert "[truncated]" in line.plain and "[truncated]" in summary.plain


@pytest.mark.asyncio
async def test_ui_escapes_tool_controls_while_followup_receives_original_result(
    tmp_path: Path,
) -> None:
    from yincode.tool import Registry

    unsafe = "原文\x1b]52;c;Y2xpcGJvYXJk\x1b\\\x1b[2J\n\t蛇🐍"
    tool = WaitingTool(unsafe)
    tool.release.set()
    registry = Registry()
    registry.register(tool)
    fake = FakeProvider(cfg())
    app = YinCodeApp([cfg()], provider_factory=lambda _: fake, cwd=tmp_path, registry=registry)
    async with app.run_test():
        await app.submit("读取")
        for event in (
            StreamEvent(tool_calls=[ToolCall("raw", "read_file", '{"path":"sample.txt"}')]),
            StreamEvent(done=True),
            StreamEvent(text="完成"),
            StreamEvent(done=True),
        ):
            await fake.events.put(event)
        await wait_for_state(app, SessionState.IDLE)
        assert fake.requests[1][-1].tool_results[0].content == unsafe
        assert app.conv.messages()[2].tool_results[0].content == unsafe
    output = render(app.transcript)
    assert "\x1b" not in output
    assert r"\x1b]52;c;Y2xpcGJvYXJk" in output
    assert r"\x1b[2J" in output and "蛇🐍" in output
