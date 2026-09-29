"""模型选择只在选定后创建客户端。"""

from rich.text import Text
from textual.widgets import OptionList

from yincode.config import ProviderConfig


def provider_options(providers: list[ProviderConfig]) -> OptionList:
    return OptionList(
        *(Text(f"{provider.name} ({provider.model})") for provider in providers),
        id="providers",
    )
