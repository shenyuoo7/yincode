"""启动终端界面，并在终端恢复后重放本次完成的对话。"""

import argparse
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.text import Text

from yincode import __version__
from yincode.config import ConfigError, load, redact


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="yincode", description="终端 AI 编程助手")
    parser.add_argument("--version", action="version", version=f"yincode {__version__}")
    parser.parse_args(argv)
    stderr = Console(stderr=True)
    cwd = Path.cwd()
    try:
        cfg = load(cwd / ".yincode" / "config.yaml")
    except ConfigError as error:
        stderr.print(Text(str(error), style="red"))
        raise SystemExit(1) from None

    app = None
    exit_code = 0
    try:
        from yincode.tool import new_default_registry
        from yincode.tui import YinCodeApp

        app = YinCodeApp(cfg.providers, cwd=str(cwd), registry=new_default_registry(cwd=cwd))
        app.run()
    except KeyboardInterrupt:
        pass
    except Exception as error:
        message = redact(str(error), (provider.api_key for provider in cfg.providers))
        stderr.print(Text(f"启动失败: {message}", style="red"))
        exit_code = 1
    finally:
        # 此时 Textual 已还原终端，普通输出进入系统 scrollback。
        if app is not None:
            console = Console()
            for block in app.transcript:
                console.print(block)
    if exit_code:
        raise SystemExit(exit_code)
