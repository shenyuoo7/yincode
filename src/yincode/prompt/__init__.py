from rich.cells import cell_len, chop_cells
from rich.text import Text

SYSTEM_PROMPT = (
    "You are yincode, a terminal AI coding assistant. Help the user understand and write code. "
    "Be precise, explain assumptions, and use Markdown when helpful. "
    "You cannot execute commands or access files in this conversation. "
    "Never claim to have performed actions you have not performed."
)

GOLD = "#EDC66F"
BRONZE = "#B8894D"

# 蛇眼用留白表现，交错的两段身体在单色终端中也保留轮廓。
_SNAKE = (
    "    ▄██████▄    ",
    "    ██ ▀████    ",
    " ▄▄▄████████    ",
    "█████▀▀▀▀▀▀▀    ",
    "    ▄▄▄▄▄█████  ",
    "    ████████▀▀  ",
    "    ████▄ ██    ",
    "    ▀██████▀    ",
)
_WORDMARK = (
    "██      ██  ██████████  ██      ██",
    "██      ██      ██      ████    ██",
    "  ██  ██        ██      ████    ██",
    "    ██          ██      ██  ██  ██",
    "    ██          ██      ██    ████",
    "    ██          ██      ██    ████",
    "    ██      ██████████  ██      ██",
    "",
)
YIN_BANNER = "\n".join(
    f"  {snake}    {wordmark}".rstrip() for snake, wordmark in zip(_SNAKE, _WORDMARK, strict=True)
)
COMPACT_BANNER = "YIN  ~<(o)=(o)>~"


def render_banner(version: str, cwd: str, width: int = 80) -> str:
    """保留纯文本接口，供无颜色输出使用。"""
    return render_banner_text(version, cwd, width).plain


def render_banner_text(version: str, cwd: str, width: int = 80) -> Text:
    """生成可直接交给 RichLog 与 Console 的横幅。"""
    width = max(1, width)
    wide = width >= max(cell_len(line) for line in YIN_BANNER.splitlines())
    banner = Text()
    if wide:
        banner.append("\n")
        for index, (snake, wordmark) in enumerate(zip(_SNAKE, _WORDMARK, strict=True)):
            banner.append("  " + snake, GOLD if index < 4 else BRONZE)
            banner.append("    " + wordmark + "\n", GOLD)
        banner.append("\n")
    else:
        banner.append("YIN", "bold " + GOLD)
        banner.append(COMPACT_BANNER[3:] + "\n", BRONZE)
    indent = "  " if wide else ""
    banner.append(f"{indent}yincode v{version}\n", "bold #D8D2C5")
    banner.append(f"{indent}{cwd}\n", "#A39E93")
    banner.append(f"{indent}Ready", GOLD)
    banner.append(" · Enter 发送 · Alt+Enter 换行 · /exit 退出", "#A39E93")

    # 按终端显示单元折行，同时保留颜色和完整的中文路径；不解析路径中的标记。
    wrapped: list[Text] = []
    for line in banner.split("\n", allow_blank=True):
        offset = 0
        for chunk in chop_cells(line.plain, width) or [""]:
            wrapped.append(line[offset : offset + len(chunk)])
            offset += len(chunk)
    return Text("\n").join(wrapped)
