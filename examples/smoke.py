"""使用本地配置连续对话，观察真实端点的 token 与缓存用量。

运行：uv run python examples/smoke.py "请只回复你好" "请重复上一条回复"
默认消息仅请求纯文本回复；指定消息可验证工作目录内的工具链。
"""

import argparse
import asyncio
import sys
from collections.abc import AsyncGenerator, Sequence
from contextlib import aclosing
from pathlib import Path
from typing import cast

from yincode import __version__
from yincode.agent import Agent, Event, Mode
from yincode.config import Config, ConfigError, load, redact
from yincode.conversation import Conversation
from yincode.llm import new_provider
from yincode.tool import new_default_registry


async def run(config: Config, cwd: Path, prompts: list[str], plan: bool) -> None:
    """共用一个会话，关闭每轮迭代器和提供者，取消向上传播。"""
    secrets = tuple(provider.api_key for provider in config.providers)
    provider = new_provider(config.providers[0])
    try:
        agent = Agent(
            provider,
            new_default_registry(cwd=cwd),
            __version__,
            cwd=cwd,
            redactor=lambda text: redact(text, secrets),
            secrets=secrets,
        )
        conversation = Conversation()
        for prompt in prompts:
            conversation.add_user(prompt)
            response: list[str] = []
            async with aclosing(
                cast(
                    AsyncGenerator[Event, None],
                    agent.run(conversation, Mode.PLAN if plan else Mode.NORMAL, asyncio.Event()),
                )
            ) as events:
                async for event in events:
                    if event.text:
                        response.append(event.text)
                    if event.usage is not None:
                        usage = event.usage
                        print(
                            f"input={usage.input_tokens} output={usage.output_tokens} "
                            f"cache_write={usage.cache_write} cache_read={usage.cache_read}"
                        )
                    if event.err is not None:
                        raise event.err
                    if event.notice:
                        response.append(event.notice)
            # 整轮脱敏可识别跨流分片的密钥，避免直接输出每个分片。
            print(redact("".join(response), secrets))
    finally:
        await provider.aclose()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="纯文本会话与缓存用量冒烟")
    parser.add_argument("--config", type=Path, help="配置路径；默认工作目录下 .yincode/config.yaml")
    parser.add_argument("--cwd", type=Path, default=Path.cwd(), help="会话工作目录")
    parser.add_argument("--plan", action="store_true", help="启用规划提醒")
    parser.add_argument("prompts", nargs="*", help="最多两条消息；第二轮可观察缓存复用")
    args = parser.parse_args(argv)
    if len(args.prompts) > 2:
        parser.error("最多接受两条消息")
    cwd = args.cwd.resolve()
    if not cwd.is_dir():
        parser.error("工作目录不存在或不是目录")
    try:
        config = load(args.config or cwd / ".yincode" / "config.yaml")
    except ConfigError as error:
        print(str(error), file=sys.stderr)
        return 1
    try:
        asyncio.run(run(config, cwd, args.prompts or ["请只回复你好，不调用工具。"], args.plan))
    except Exception as error:
        secrets = tuple(provider.api_key for provider in config.providers)
        print(redact(str(error), secrets), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
