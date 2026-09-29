"""工具参数的严格对象解析，拒绝非有限数字。"""

import json
import math
from typing import Any


def _reject_constant(value: str) -> None:
    raise ValueError("工具参数包含非 JSON 数值")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("工具参数数值超出可表示范围")
    return number


def object_from_json(raw: str) -> dict[str, Any]:
    value = json.loads(raw or "{}", parse_constant=_reject_constant, parse_float=_finite_float)
    if not isinstance(value, dict):
        raise ValueError("工具参数必须是 JSON 对象")
    return value
