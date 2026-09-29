"""Rich 渲染块：动态正文保留原文，完成正文使用 Markdown。"""

from rich.console import Group
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text


def user_block(text: str) -> Text:
    return Text("● " + text, style="bold")


def render_markdown(reply: str, elapsed: float) -> Group:
    return Group(Text("●", style="bold"), Markdown(reply), elapsed_block(elapsed))


def elapsed_block(elapsed: float) -> Text:
    return Text(f"耗时：{elapsed:.1f}s", style="dim")


def error_block(message: str, elapsed: float | None = None) -> Text:
    suffix = f"\n耗时：{elapsed:.1f}s" if elapsed is not None else ""
    return Text("● " + message + suffix, style="bold red")


def streaming_block(reply: str, elapsed: float) -> Text:
    text = Text("● " + reply + "\n" if reply else "")
    frame = "|/-\\"[int(elapsed * 10) % 4]
    text.append(f"Imagining… ({int(elapsed)}s) {frame}", style="dim")
    return text


def status_bar(name: str, model: str) -> Table:
    table = Table.grid(expand=True)
    table.add_column(ratio=1, overflow="fold")
    table.add_column(justify="right", overflow="fold")
    table.add_row(Text(name), Text(model))
    return table
