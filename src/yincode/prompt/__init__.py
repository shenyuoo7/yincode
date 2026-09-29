from rich.cells import chop_cells

SYSTEM_PROMPT = (
    "You are yincode, a terminal AI coding assistant. Help the user understand and write code. "
    "Be precise, explain assumptions, and use Markdown when helpful. "
    "You cannot execute commands or access files in this conversation. "
    "Never claim to have performed actions you have not performed."
)

YIN_BANNER = r""" Y   Y  III  N   N       __
  Y Y    I   NN  N    __/o \__
   Y     I   N N N   /  __/~~
   Y    III  N  NN   \_/  snake"""

COMPACT_BANNER = "YIN  ~~~(o)>  snake"


def render_banner(version: str, cwd: str, width: int = 80) -> str:
    width = max(1, width)
    banner = YIN_BANNER if width >= 40 else COMPACT_BANNER
    # 按终端显示单元换行，完整保留中文路径和窄窗口中的关键信息。
    lines = [banner, f"yincode v{version}", cwd, "Ready · Enter 发送 · Alt+Enter 换行 · /exit 退出"]
    return "\n".join(
        "\n".join(chop_cells(line, width) or [""]) for block in lines for line in block.splitlines()
    )
