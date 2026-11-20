"""严格 JSON 解析与 completion 适配；不修复、不抽取 JSON 子串。"""

import json
import math


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"重复 JSON 键: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"非有限 JSON 数值: {value}")


def _float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("非有限 JSON 数值")
    return parsed


def parse_json_object(raw):
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        value = json.loads(raw, object_pairs_hook=_pairs,
                           parse_constant=_constant, parse_float=_float)
    except (ValueError, TypeError, RecursionError, OverflowError):
        return None
    return value if isinstance(value, dict) else None


def completion_text(completion):
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        if completion.get("role", "assistant") != "assistant":
            return ""
        content = completion.get("content", "")
    elif isinstance(completion, (list, tuple)):
        assistants = [message for message in completion
                      if isinstance(message, dict) and message.get("role") == "assistant"]
        return completion_text(assistants[-1]) if assistants else ""
    else:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list) and all(
        isinstance(part, dict) and part.get("type") in ("text", "output_text")
        and isinstance(part.get("text"), str) for part in content
    ):
        return "".join(part["text"] for part in content)
    return ""
