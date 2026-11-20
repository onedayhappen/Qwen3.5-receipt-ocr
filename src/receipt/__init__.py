"""仅依赖 CPU 的票据 schema、评估和可验证奖励。"""

from .schema import RECEIPT_FIELDS, get_schema, register_schema, validate_output
from .prompts import build_prompt
from .normalization import normalize_document, normalize_value
from .matching import match_items
from .validation import validate_document
from .metrics import aggregate_metrics, evaluate_sample, parse_response

__all__ = [
    "RECEIPT_FIELDS", "get_schema", "register_schema", "validate_output",
    "build_prompt", "normalize_document", "normalize_value", "match_items",
    "validate_document", "aggregate_metrics", "evaluate_sample", "parse_response",
]
