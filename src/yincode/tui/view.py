"""Rich 渲染块：不可信控制字符可见转义，完成正文使用 Markdown。"""

from rich.console import Group
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text


def _visible_text(text: str) -> str:
    """阻止文本控制终端，保留换行、制表符和合法 Unicode。"""
    text = text.replace("\r\n", "\n")
    return "".join(
        f"\\x{ord(char):02x}"
        if char not in "\n\t" and (ord(char) < 32 or 127 <= ord(char) <= 159)
        else char
        for char in text
    )


def user_block(text: str) -> Text:
    return Text("● " + _visible_text(text), style="bold")


def render_markdown(reply: str, elapsed: float | None = None) -> Group:
    reply = _visible_text(reply)
    if elapsed is not None:
        return Group(Text("●", style="bold"), Markdown(reply), elapsed_block(elapsed))
    return Group(Text("●", style="bold"), Markdown(reply))


def elapsed_block(elapsed: float) -> Text:
    return Text(f"耗时：{elapsed:.1f}s", style="dim")


def error_block(message: str, elapsed: float | None = None) -> Text:
    suffix = f"\n耗时：{elapsed:.1f}s" if elapsed is not None else ""
    return Text("● " + _visible_text(message) + suffix, style="bold red")


def streaming_block(reply: str, elapsed: float) -> Text:
    text = Text("● " + _visible_text(reply) + "\n" if reply else "")
    frame = "|/-\\"[int(elapsed * 10) % 4]
    text.append(f"Imagining… ({int(elapsed)}s) {frame}", style="dim")
    return text


def _bounded_text(text: str, *, max_lines: int, max_bytes: int) -> str:
    """预留截断标记的行数和字节，不切开 UTF-8 字符。"""
    lines = text.splitlines()
    if len(lines) <= max_lines and len(text.encode("utf-8")) <= max_bytes:
        return text
    marker = "\n[truncated]" if max_lines > 1 else "[truncated]"
    head = "\n".join(lines[: max_lines - 1 if max_lines > 1 else 1])
    budget = max_bytes - len(marker.encode("utf-8"))
    head = head.encode("utf-8")[:budget].decode("utf-8", errors="ignore").rstrip("\n")
    return head + marker


def tool_line(name: str, args: str) -> Text:
    label = _visible_text(f"{name}({args})").replace("\n", " ")
    label = _bounded_text(label, max_lines=1, max_bytes=512 - len("● ".encode()))
    text = Text("● ", style="bold cyan")
    text.append(label, style="bold")
    return text


def tool_result_summary(result: str, is_error: bool) -> Text:
    # 约束最终显示文本，含缩进与标记，而不只约束结果正文。
    summary = "  ⎿ " + _visible_text(result).replace("\n", "\n    ")
    summary = _bounded_text(summary, max_lines=8, max_bytes=4096)
    return Text(summary, style="red" if is_error else "dim")


def tool_streaming_block(name: str, args: str, elapsed: float) -> Text:
    text = tool_line(name, args)
    frame = "|/-\\"[int(elapsed * 10) % 4]
    text.append(f"\nRunning… ({int(elapsed)}s) {frame}", style="dim")
    return text


def status_bar(name: str, model: str) -> Table:
    table = Table.grid(expand=True)
    table.add_column(ratio=1, overflow="fold")
    table.add_column(justify="right", overflow="fold")
    table.add_row(Text(_visible_text(name)), Text(_visible_text(model)))
    return table
