"""保守规范化：不覆盖原文，不丢失零、不猜歧义日期或金额单位。"""

from datetime import date
from decimal import Decimal, InvalidOperation
import re
import unicodedata

_NUMERIC_FIELDS = frozenset({"quantity", "price", "subtotal", "tax", "total"})
_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)")
_AMOUNT = re.compile(
    r"(?P<prefix>[$€£¥￥]|[A-Z]{3}\s+)?"
    r"(?P<number>[+-]?(?:\d+(?:\.\d*)?|\.\d+|\d{1,3}(?:,\d{3})+\.\d+))"
    r"(?P<suffix>\s*(?:元|[$€£¥￥]|[A-Z]{3}))?"
)


def decimal_value(value):
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        return value if value.is_finite() else None
    if not isinstance(value, (str, int, float)):
        return None
    text = str(value).strip()
    if not _NUMBER.fullmatch(text):
        return None
    try:
        result = Decimal(text)
    except InvalidOperation:
        return None
    return result if result.is_finite() else None


def _decimal_text(value):
    if value == 0:
        return "0"
    return format(value, "f").rstrip("0").rstrip(".") if "." in format(value, "f") else format(value, "f")


def _date(text):
    parts = re.fullmatch(r"(\d{4})[-/.年](\d{1,2})[-/.月](\d{1,2})日?", text)
    if parts:
        year, month, day = map(int, parts.groups())
    else:
        parts = re.fullmatch(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})", text)
        if not parts:
            return text
        first, second, year = map(int, parts.groups())
        if first > 12 and second <= 12:
            day, month = first, second
        elif second > 12 and first <= 12:
            month, day = first, second
        else:
            return text
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return text


def normalize_value(field, value):
    if value is None:
        return None
    if isinstance(value, dict):
        return {key: normalize_value(key, child) for key, child in value.items()}
    if isinstance(value, list):
        return [normalize_value(field, child) for child in value]
    if not isinstance(value, (str, int, float, Decimal)) or isinstance(value, bool):
        return value
    text = " ".join(unicodedata.normalize("NFKC", str(value)).split())
    if not text:
        return None
    if field in _NUMERIC_FIELDS:
        match = _AMOUNT.fullmatch(text)
        if match:
            number = decimal_value(match["number"].replace(",", ""))
            if number is not None:
                prefix = (match["prefix"] or "").strip()
                suffix = (match["suffix"] or "").strip()
                return prefix + _decimal_text(number) + suffix
        return text
    if field == "date":
        return _date(text)
    if field == "time":
        match = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", text)
        if match:
            hour, minute, second = match.groups()
            if int(hour) < 24 and int(minute) < 60 and (second is None or int(second) < 60):
                return f"{int(hour):02d}:{minute}" + (f":{second}" if second else "")
    return text


def normalize_document(data):
    if not isinstance(data, dict):
        raise TypeError("document 必须是 dict")
    return {field: normalize_value(field, value) for field, value in data.items()}
