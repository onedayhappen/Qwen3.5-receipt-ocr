"""结构校验与显式标注关系校验；无标注生产请求不猜金额公式。"""

from decimal import Decimal, localcontext

from .io import parse_json_object
from .normalization import decimal_value, normalize_value
from .schema import get_schema, json_value_issues, validate_output


def _number(field, value):
    return decimal_value(normalize_value(field, value))


def _readable(annotation, fields):
    statuses = annotation.get("field_status")
    return not isinstance(statuses, dict) or all(
        statuses.get(field) == "present_readable" for field in fields
    )


def _compare(actual, expected, tolerance, reason=None):
    if actual is None or expected is None or reason:
        return {"passed": False, "reason": reason or "missing_or_ambiguous_operand"}
    return {
        "passed": abs(actual - expected) <= tolerance,
        "actual": str(actual), "expected": str(expected), "tolerance": str(tolerance),
    }


def validate_document(data, annotation=None, schema_version="receipt_v1", fields=None):
    schema = get_schema(schema_version, fields)
    parsed = parse_json_object(data) if isinstance(data, str) else data
    json_valid = isinstance(parsed, dict) and not json_value_issues(parsed)
    issues = validate_output(parsed, schema_version, fields) if json_valid else ["invalid_json_object"]
    schema_valid = json_valid and not issues
    result = {
        "json_valid": bool(json_valid), "schema_valid": bool(schema_valid),
        "issues": issues, "applicable_checks": [], "check_results": {},
    }
    if not isinstance(annotation, dict):
        return result
    declared = set(schema.get("properties", {}))
    if schema_version == "invoice_json_v1" and fields is None:
        declared = set(parsed or {})
    tolerance = decimal_value(annotation.get("amount_tolerance",
                                               annotation.get("currency_min_unit", "0.01")))
    if tolerance is None or tolerance < 0:
        result["issues"].append("invalid_amount_tolerance")
        return result
    doc = parsed if json_valid else {}
    checks = result["check_results"]
    with localcontext() as context:
        # 票据金额通常远小于此精度，避免默认 28 位精度截掉长数值。
        context.prec = 256
        subtotal_rule = annotation.get("subtotal_equation")
        basis = annotation.get("price_basis")
        if {"items", "subtotal"} <= declared and annotation.get("items_complete") is True \
                and _readable(annotation, ("items", "subtotal")):
            unit_rule = subtotal_rule == "sum_quantity_times_unit_price" and basis == "unit"
            line_rule = subtotal_rule in ("sum_line_prices", "sum_line_amounts") \
                and basis in ("line", "line_total", "line_amount")
            if unit_rule or line_rule:
                name = "subtotal_equation"
                expected, reason = Decimal(0), None
                rows = doc.get("items")
                if not isinstance(rows, list) or not rows:
                    reason = "missing_items_for_subtotal"
                else:
                    for row in rows:
                        if not isinstance(row, dict):
                            reason = "invalid_item_operand"
                            break
                        price = _number("price", row.get("price"))
                        quantity = _number("quantity", row.get("quantity")) if unit_rule else Decimal(1)
                        if price is None or quantity is None:
                            reason = "missing_or_ambiguous_item_operand"
                            break
                        expected += price * quantity
                checks[name] = _compare(_number("subtotal", doc.get("subtotal")),
                                        expected, tolerance, reason)
        total_rule = annotation.get("total_equation")
        if total_rule == "subtotal_plus_tax":
            operands = ("subtotal", "tax", "total")
        elif total_rule in ("subtotal_equals_total", "subtotal", "tax_included"):
            operands = ("subtotal", "total")
        else:
            operands = ()
        if operands and set(operands) <= declared and _readable(annotation, operands):
            subtotal = _number("subtotal", doc.get("subtotal"))
            tax = _number("tax", doc.get("tax")) if "tax" in operands else Decimal(0)
            expected = subtotal + tax if subtotal is not None and tax is not None else None
            checks["total_equation"] = _compare(_number("total", doc.get("total")), expected, tolerance)
    result["applicable_checks"] = list(checks)
    result["issues"].extend(f"{name}_failed" for name, check in checks.items() if not check["passed"])
    return result
