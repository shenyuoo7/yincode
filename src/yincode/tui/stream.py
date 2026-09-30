"""在界面事件循环中消费流，并在所有结束路径关闭迭代器。"""

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from yincode.agent import Event

from .view import notice_block

if TYPE_CHECKING:
    from .app import YinCodeApp


async def consume_stream(app: "YinCodeApp") -> None:
    iterator: AsyncIterator[Event] | None = None
    error: Exception | None = None
    completed = False
    try:
        assert app.agent is not None
        iterator = app.agent.run(app.conv, app.mode, app.turn_cancel)
        async for event in iterator:
            if event.err is not None:
                error = event.err
                break
            if event.tool is not None:
                app._handle_tool_event(event.tool)
            if event.usage is not None:
                app.usage_in += event.usage.input_tokens
                app.usage_out += event.usage.output_tokens
                app._refresh_status()
            if event.notice:
                app._append_history(notice_block(app._redact(event.notice)))
            if event.iter > 0:
                app.iter = event.iter
                app._refresh_streaming_view()
            if event.done:
                completed = True
                break
            if event.text:
                # 分片可能把密钥拆开；对累积正文再脱敏，保护动态区和完成块。
                app.cur_reply = app._redact(app.cur_reply + event.text)
                app._refresh_streaming_view(follow_output=True)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        error = exc
    finally:
        try:
            # Protocol 允许普通 AsyncIterator；异步生成器则需显式收尾。
            close = getattr(iterator, "aclose", None)
            if close is not None:
                await close()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = error or exc
        finally:
            if app._stream_task is asyncio.current_task():
                app._stream_task = None

    if app._resources_closing:
        return
    if error is not None:
        app._finish_with_error(error)
    elif completed:
        app._finish_with_assistant(app.cur_reply)
    else:
        app._finish_with_error(RuntimeError("响应流提前结束，未收到完成事件，请重试"))
