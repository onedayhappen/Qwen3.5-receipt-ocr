import itertools
from pathlib import Path
import random
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from receipt.schema import (
    RECEIPT_FIELDS, get_schema, register_invoice_fatura_schema, register_schema, validate_output,
)
from receipt.prompts import build_prompt
from receipt.normalization import decimal_value, normalize_document, normalize_value
from receipt.matching import _maximum_assignment, match_items
from receipt.validation import validate_document


def item(name="牛奶", quantity="2", price="12.50"):
    return {"name": name, "quantity": quantity, "price": price}


def doc():
    result = {field: None for field in RECEIPT_FIELDS}
    result.update(merchant_name="示例商店", date="2026-10-01", items=[item()],
                  subtotal="25.00", tax="0.00", total="25.00")
    return result


def annotation(**changes):
    result = {
        "field_status": {field: "present_readable" if value is not None else "not_printed"
                         for field, value in doc().items()},
        "reviewed": True, "items_complete": True, "price_basis": "unit",
        "subtotal_equation": "sum_quantity_times_unit_price", "total_equation": "subtotal_plus_tax",
    }
    result.update(changes)
    return result


def test_schema_contract_and_copy():
    assert RECEIPT_FIELDS == (
        "merchant_name", "address", "phone", "date", "time", "order_number", "items",
        "subtotal", "tax", "total", "payment_method",
    )
    assert validate_output(doc()) == []
    changed = get_schema()
    changed["properties"]["items"]["items"]["properties"]["price"]["type"] = "number"
    assert get_schema()["properties"]["items"]["items"]["properties"]["price"]["type"] == ["string", "null"]


@pytest.mark.parametrize("fields", [["total"], ["merchant_name", "date"], []])
def test_partial_schema(fields):
    schema = get_schema(fields=fields)
    assert schema["required"] == fields
    assert set(schema["properties"]) == set(fields)
    assert schema["additionalProperties"] is False
    assert validate_output({field: doc()[field] for field in fields}, fields=fields) == []
    assert validate_output({"unexpected": None}, fields=fields)


@pytest.mark.parametrize("fields", ["total", ["unknown"], ["total", "total"], [None]])
def test_reject_invalid_field_selection(fields):
    with pytest.raises(ValueError):
        get_schema(fields=fields)


@pytest.mark.parametrize("bad", [[], "{}", None, 0, {"total": 0}, {"total": False}, {"total": float("nan")}])
def test_schema_rejects_wrong_types(bad):
    assert validate_output(bad, fields=["total"])


@pytest.mark.parametrize("bad_item", [item(price=12), {"name": "牛奶"}, {**item(), "currency": "CNY"}, []])
def test_strict_item_schema(bad_item):
    assert validate_output({"items": [bad_item]}, fields=["items"])


def test_fatura_requires_registration_and_frozen_copy(monkeypatch):
    from receipt import schema as schema_module
    monkeypatch.delitem(schema_module._SCHEMAS, "invoice_fatura_v1", raising=False)
    # 让 monkeypatch 同时撤销本测试的注册，不污染数据代理的真实 schema。
    monkeypatch.setitem(schema_module._SCHEMAS, "invoice_fatura_v1", None)
    monkeypatch.delitem(schema_module._SCHEMAS, "invoice_fatura_v1")
    with pytest.raises(ValueError, match="未注册"):
        get_schema("invoice_fatura_v1")
    schema = {"type": "object", "properties": {
        "SELLER_TEST": {"type": ["string", "null"]},
        "AMOUNT_TEST": {"type": ["number", "null"]},
        "TABLE_TEST": {"type": ["array", "null"], "items": {"type": "object"}},
    }, "required": ["SELLER_TEST", "AMOUNT_TEST", "TABLE_TEST"], "additionalProperties": False}
    register_invoice_fatura_schema(schema)
    schema["properties"]["SELLER_TEST"]["type"] = "number"
    fixed = get_schema("invoice_fatura_v1", ["SELLER_TEST"])
    assert fixed["required"] == ["SELLER_TEST"]
    assert validate_output({"SELLER_TEST": "原文"}, "invoice_fatura_v1", ["SELLER_TEST"]) == []
    with pytest.raises(ValueError, match="不可改写"):
        register_schema("invoice_fatura_v1", schema)


def test_generic_invoice_is_recursive_and_key_limited():
    data = {"TOTAL": 0, "TABLE": [{"name": "原文", "price": 1.5}, None]}
    assert validate_output(data, "invoice_json_v1", ["TOTAL", "TABLE"]) == []
    assert validate_output({**data, "unrequested": None}, "invoice_json_v1", ["TOTAL", "TABLE"])
    assert validate_output({"TOTAL": float("inf")}, "invoice_json_v1", ["TOTAL"])
    assert validate_output({"TOTAL": True}, "invoice_json_v1", ["TOTAL"])


def test_prompt_is_shared_partial_and_no_image_marker():
    prompt = build_prompt(fields=["total"])
    assert "<image>" not in prompt
    assert '"merchant_name"' not in prompt
    assert '"items"' not in prompt
    assert "null" in prompt and "不猜" in prompt
    full = build_prompt()
    assert "无明确上下文" in full and "不默认数量为 1" in full
    assert build_prompt(fields=["total"]) == prompt


@pytest.mark.parametrize("field,value,expected", [
    ("total", None, None), ("total", 0, "0"), ("total", "0.00", "0"),
    ("total", "-0.00", "0"), ("price", "12.500", "12.5"),
    ("quantity", "2.00", "2"), ("name", "  示例　 商品  ", "示例 商品"),
    ("name", "", None), ("total", "$1,234.00", "$1234"),
    ("total", "12.00元", "12元"), ("total", "1,234", "1,234"),
    ("total", "12,50", "12,50"), ("total", "USD 12.00", "USD12"),
    ("date", "2026/2/3", "2026-02-03"), ("date", "03/02/2026", "03/02/2026"),
    ("date", "31/01/2026", "2026-01-31"), ("date", "01/31/2026", "2026-01-31"),
    ("date", "2026-02-30", "2026-02-30"), ("time", "9:05", "09:05"),
    ("time", "25:05", "25:05"),
])
def test_conservative_normalization(field, value, expected):
    assert normalize_value(field, value) == expected


def test_zero_and_nested_normalization():
    normalized = normalize_document({"total": 0, "tax": None, "items": [item(quantity=None, price="0.00")]})
    assert normalized["total"] == "0"
    assert normalized["tax"] is None
    assert normalized["items"][0]["quantity"] is None
    assert normalized["items"][0]["price"] == "0"
    assert decimal_value("0") == 0
    assert decimal_value(None) is None
    assert decimal_value("NaN") is None
    assert decimal_value("1,2") is None


def test_item_permutation_and_duplicate_no_gold_reuse():
    gold = [item(), item("面包", "1", "5")]
    assert match_items(list(reversed(gold)), gold)["f1"] == 1.0
    scored = match_items([item(), item()], [item()])
    assert scored["precision"] == 0.5
    assert scored["recall"] == 1.0
    assert scored["f1"] == pytest.approx(2 / 3)
    assert len({pair["gold_index"] for pair in scored["matched"]}) == 1
    assert scored["extra_count"] == 1 and scored["duplicate_extra"]
    assert scored["field_scores"]["price"]["precision"] == 0.5
    assert match_items([item(), item()], [item(), item()])["f1"] == 1.0


def test_item_attributes_and_unknown_quantity():
    scored = match_items([item(quantity="9")], [item()])
    assert scored["field_scores"]["name"]["f1"] == 1.0
    assert scored["field_scores"]["quantity"]["f1"] == 0.0
    assert scored["field_scores"]["price"]["f1"] == 1.0
    assert scored["f1"] == pytest.approx(2 / 3)
    unknown = match_items([item(quantity="999")], [item(quantity=None)])
    assert unknown["f1"] == 1.0
    assert unknown["field_scores"]["quantity"]["applicable"] is False
    assert unknown["field_scores"]["quantity"]["recall_denominator"] == 0


def test_fabrication_not_matched_by_same_numbers():
    scored = match_items([item("虚构饮料")], [item("面包")])
    assert scored["matched"] == [] and scored["extra"] == [0] and scored["missed"] == [0]
    assert scored["f1"] == 0.0


@pytest.mark.parametrize("pred,gold", [([], []), ([], [item()]), ([item()], [])])
def test_empty_items_do_not_get_perfect_f1(pred, gold):
    assert match_items(pred, gold)["f1"] == 0.0


def test_hungarian_maximum_not_greedy():
    weights = [[8, 7], [7, 0]]
    chosen = _maximum_assignment(weights)
    assert sum(weights[i][j] for i, j in chosen) == 14
    randomizer = random.Random(31)
    for size in range(1, 6):
        for _ in range(5):
            weights = [[randomizer.randint(0, 20) for _ in range(size)] for _ in range(size)]
            chosen = _maximum_assignment(weights)
            maximum = max(sum(weights[i][j] for i, j in enumerate(permutation))
                          for permutation in itertools.permutations(range(size)))
            assert sum(weights[i][j] for i, j in chosen) == maximum


def test_production_no_annotation_does_not_guess_equations():
    candidate = doc()
    candidate["total"] = "999"
    scored = validate_document(candidate)
    assert scored["json_valid"] and scored["schema_valid"]
    assert scored["applicable_checks"] == [] and scored["issues"] == []


def test_annotated_decimal_equations_and_zero_tax():
    scored = validate_document(doc(), annotation())
    assert scored["applicable_checks"] == ["subtotal_equation", "total_equation"]
    assert all(check["passed"] for check in scored["check_results"].values())
    candidate = doc()
    candidate["total"] = "25.02"
    assert not validate_document(candidate, annotation())["check_results"]["total_equation"]["passed"]


def test_line_amount_tax_included_and_negative_refund():
    candidate = doc()
    candidate.update(items=[item(quantity="2", price="25")], tax="3", total="25")
    metadata = annotation(price_basis="line", subtotal_equation="sum_line_prices", total_equation="tax_included")
    assert all(check["passed"] for check in validate_document(candidate, metadata)["check_results"].values())
    candidate.update(items=[item(quantity="2", price="-12.50")], subtotal="-25", tax="0", total="-25")
    assert all(check["passed"] for check in validate_document(candidate, annotation())["check_results"].values())


def test_units_quantity_unknown_and_empty_sum_not_rewarded():
    metadata = annotation(price_basis="unknown")
    assert "subtotal_equation" not in validate_document(doc(), metadata)["applicable_checks"]
    candidate = doc()
    candidate["items"] = [item(quantity=None)]
    assert not validate_document(candidate, annotation())["check_results"]["subtotal_equation"]["passed"]
    candidate.update(items=[], subtotal="0", tax="0", total="0")
    result = validate_document(candidate, annotation())
    assert not result["check_results"]["subtotal_equation"]["passed"]
    assert result["check_results"]["subtotal_equation"]["reason"] == "missing_items_for_subtotal"


def test_partial_fields_disable_unrequested_amount_relations():
    result = validate_document({"total": "25"}, annotation(), fields=["total"])
    assert result["schema_valid"] and result["applicable_checks"] == []


def test_unknown_status_and_currency_text_are_not_silent_arithmetic():
    metadata = annotation()
    metadata["field_status"]["tax"] = "present_unreadable"
    assert "total_equation" not in validate_document(doc(), metadata)["applicable_checks"]
    candidate = doc()
    candidate["total"] = "$25"
    result = validate_document(candidate, annotation())
    assert not result["check_results"]["total_equation"]["passed"]
