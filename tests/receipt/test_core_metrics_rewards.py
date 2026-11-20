from copy import deepcopy
import json
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from receipt.metrics import aggregate_metrics, evaluate_sample, parse_response
from receipt.rewards import receipt_composite_reward, score_completion
from receipt.schema import RECEIPT_FIELDS


def sample(**changes):
    gold = {field: None for field in RECEIPT_FIELDS}
    gold.update(merchant_name="示例商店", date="2026-10-01", subtotal="25.00", tax="0.00", total="25.00",
                items=[{"name": "牛奶", "quantity": "2", "price": "12.50"}])
    result = {
        "sample_id": "test", "schema_version": "receipt_v1", "ground_truth": gold,
        "template_group": "group_a", "language": ["zh"], "quality_tags": ["long"],
        "annotation": {
            "field_status": {field: "present_readable" if value is not None else "not_printed"
                             for field, value in gold.items()},
            "items_complete": True, "reviewed": True, "price_basis": "unit",
            "subtotal_equation": "sum_quantity_times_unit_price", "total_equation": "subtotal_plus_tax",
        },
    }
    result.update(changes)
    return result


def raw(data):
    return json.dumps(data, ensure_ascii=False)


@pytest.mark.parametrize("bad", [
    "", " ", "[]", "null", "0", "true", '"text"',
    '```json\n{"total":"1"}\n```', '答案: {"total":"1"}', '{"total":"1"} 后缀',
    '{"total":NaN}', '{"total":Infinity}', '{"total":-Infinity}', '{"total":1e999}',
    '{"total":"1","total":"2"}', '{"items":[{"name":"a","name":"b"}]}',
    '{"total":"1"', '{"total":"1"}{"total":"2"}',
])
def test_strict_response_parser_rejects_every_non_object_or_repair(bad):
    assert parse_response(bad) is None


def test_parser_accepts_whole_json_whitespace_zero_and_unicode():
    assert parse_response(' \n{"total":0,"merchant_name":"示例"}\t ') == {"total": 0, "merchant_name": "示例"}
    assert parse_response({"total": "1"}) is None


def test_perfect_sample_and_denominators():
    fixture = sample()
    result = evaluate_sample(raw(fixture["ground_truth"]), fixture)
    assert result["json_valid"] and result["schema_valid"]
    assert result["whole_document_raw_em"] is True
    assert result["items"]["f1"] == 1.0
    report = aggregate_metrics([result])
    assert report["sample_count"] == {"value": 1, "denominator": 1}
    for key in ("json_valid_rate", "schema_valid_rate", "items_f1", "whole_document_raw_em"):
        assert report[key]["value"] == 1.0 and report[key]["denominator"] == 1
    assert report["key_fields_macro_raw_em"]["denominator"] == 3
    assert report["key_fields_macro_raw_em"]["sample_field_denominator"] == 3
    assert report["consistency_pass_rate"]["denominator"] == 2
    assert report["groups"]["template_group"]["group_a"]["sample_count"]["value"] == 1


@pytest.mark.parametrize("bad", ["", "[]", "garbage", "{}", '{"total":NaN}'])
def test_failures_stay_in_all_eligible_denominators(bad):
    fixture = sample()
    good = evaluate_sample(raw(fixture["ground_truth"]), fixture)
    failed = evaluate_sample(bad, fixture)
    report = aggregate_metrics([good, failed])
    assert report["json_valid_rate"]["denominator"] == 2
    assert report["schema_valid_rate"]["denominator"] == 2
    assert report["fields"]["total"]["raw_em"]["denominator"] == 2
    assert report["items_f1"]["denominator"] == 2 and report["items_f1"]["value"] == 0.5
    assert report["whole_document_raw_em"]["denominator"] == 2
    assert report["whole_document_raw_em"]["value"] == 0.5


def test_raw_and_normalized_em_separate_and_reordered_items():
    fixture = sample()
    prediction = deepcopy(fixture["ground_truth"])
    prediction["total"] = "25"
    prediction["date"] = "2026/10/01"
    result = evaluate_sample(raw(prediction), fixture)
    assert result["field_metrics"]["total"]["raw_em"] is False
    assert result["field_metrics"]["total"]["normalized_em"] is True
    assert result["whole_document_raw_em"] is False
    assert result["whole_document_normalized_em"] is True
    fixture["ground_truth"]["items"].append({"name": "面包", "quantity": None, "price": "5"})
    reordered = deepcopy(fixture["ground_truth"])
    reordered["items"].reverse()
    assert evaluate_sample(raw(reordered), fixture)["whole_document_raw_em"] is True


def test_zero_not_dropped_from_annotation_or_exact_match():
    fixture = sample()
    result = evaluate_sample(raw(fixture["ground_truth"]), fixture)
    assert result["field_metrics"]["tax"]["applicable"]
    assert result["field_metrics"]["tax"]["raw_em"] is True
    partial = sample(ground_truth={"total": 0}, requested_fields=["total"], schema_version="invoice_json_v1",
                     annotation={"field_status": {"total": "present_readable"}, "reviewed": True})
    result = evaluate_sample('{"total":0}', partial)
    assert result["field_metrics"]["total"]["raw_em"] is True and not result["empty"]
    assert score_completion('{"total":0}', partial)["reward"] == 1.0


def test_not_printed_unknown_and_unreadable_are_distinct():
    fixture = sample()
    fixture["annotation"]["field_status"]["address"] = "unannotated"
    fixture["annotation"]["field_status"]["phone"] = "present_unreadable"
    fixture["ground_truth"]["address"] = None
    prediction = deepcopy(fixture["ground_truth"])
    prediction.update(address="未经审定的值", phone="未知电话", order_number="虚构编号")
    result = evaluate_sample(raw(prediction), fixture)
    assert result["field_metrics"]["address"]["raw_em"] is None
    assert result["field_metrics"]["phone"]["normalized_em"] is None
    assert result["hallucination"]["field_count"] == 1
    assert result["hallucination"]["field_denominator"] == 3
    assert result["whole_document_raw_em"] is None
    assert result["annotation_coverage"]["numerator"] == len(RECEIPT_FIELDS) - 2


def test_partial_annotation_and_partial_request_never_claim_whole_document():
    fixture = sample()
    fixture["annotation"]["field_status"]["address"] = "unannotated"
    result = evaluate_sample(raw(fixture["ground_truth"]), fixture)
    assert result["whole_document_exact"] is None
    partial = sample(requested_fields=["total"])
    result = evaluate_sample('{"total":"25.00"}', partial)
    assert result["schema_valid"]
    assert result["whole_document_exact"] is None and result["items"] is None
    report = aggregate_metrics([result])
    assert report["whole_document_raw_em"]["denominator"] == 0
    assert report["whole_document_raw_em"]["value"] is None


def test_unannotated_items_excluded_and_coverage_reported():
    fixture = sample()
    fixture["annotation"]["items_complete"] = False
    fixture["annotation"]["field_status"]["items"] = "unannotated"
    prediction = deepcopy(fixture["ground_truth"])
    prediction["items"] = []
    result = evaluate_sample(raw(prediction), fixture)
    assert result["items"] is None and result["whole_document_exact"] is None
    report = aggregate_metrics([result])
    assert report["items_f1"]["denominator"] == 0
    assert report["items_coverage"] == {"value": 0.0, "numerator": 0, "denominator": 1}


@pytest.mark.parametrize("finish", ["length", "max_tokens", "max_length", "truncated"])
def test_even_valid_json_truncation_fails_content_and_reward(finish):
    fixture = sample()
    result = evaluate_sample(raw(fixture["ground_truth"]), fixture, finish_reason=finish)
    assert result["truncated"] and result["json_valid"]
    assert result["field_metrics"]["total"]["raw_em"] is False
    assert result["whole_document_exact"] is False and result["items"]["f1"] == 0
    fixture["finish_reason"] = finish
    assert score_completion(raw(fixture["ground_truth"]), fixture)["reward"] == -1


def test_long_receipt_and_truncated_tail():
    fixture = sample()
    fixture["ground_truth"]["items"] = [
        {"name": f"商品{index:03d}", "quantity": "1", "price": "1.00"} for index in range(80)
    ]
    fixture["ground_truth"].update(subtotal="80.00", total="80.00")
    response = raw(fixture["ground_truth"])
    assert len(response) > 4000
    assert evaluate_sample(response, fixture)["items"]["f1"] == 1.0
    assert score_completion(response, fixture)["reward"] == 1.0
    assert score_completion(response[:-10], fixture)["reward"] == -1.0


def test_duplicate_item_metrics_and_reward_penalty():
    fixture = sample()
    prediction = deepcopy(fixture["ground_truth"])
    prediction["items"].append(deepcopy(prediction["items"][0]))
    result = evaluate_sample(raw(prediction), fixture)
    assert result["items"]["extra_count"] == 1
    assert result["hallucination"]["item_count"] == 1
    assert result["whole_document_exact"] is False
    score = score_completion(raw(prediction), fixture)
    assert score["penalties"]["extra_items"] == pytest.approx(0.075)
    assert score["components"]["consistency"] == 0.0
    assert score["reward"] < score_completion(raw(fixture["ground_truth"]), fixture)["reward"]


@pytest.mark.parametrize("prediction_type", ["all_null", "empty_items", "missing_all", "invalid", "zero"])
def test_reward_rejects_reward_hacking(prediction_type):
    fixture = sample()
    prediction = deepcopy(fixture["ground_truth"])
    if prediction_type == "all_null":
        prediction = {field: None for field in RECEIPT_FIELDS}
    elif prediction_type == "empty_items":
        prediction["items"] = []
    elif prediction_type == "missing_all":
        prediction["items"] = [{"name": "虚构饮料", "quantity": "2", "price": "12.50"}]
    elif prediction_type == "invalid":
        prediction = {}
    else:
        prediction.update(subtotal="0", tax="0", total="0", items=[])
    assert score_completion(raw(prediction), fixture)["reward"] == -1.0


def test_self_consistent_wrong_amounts_have_fixed_mask_and_zero_consistency_reward():
    fixture = sample()
    prediction = deepcopy(fixture["ground_truth"])
    prediction["items"][0]["price"] = "50.00"
    prediction.update(subtotal="100.00", total="100.00")
    score = score_completion(raw(prediction), fixture)
    perfect = score_completion(raw(fixture["ground_truth"]), fixture)
    assert score["applicable_mask"] == perfect["applicable_mask"]
    assert score["weight_denominator"] == pytest.approx(1.0)
    assert score["components"]["consistency"] == 0.0
    assert score["consistency_gate_passed"] is False
    assert score["reward"] < 0.7


def test_hallucination_penalty_and_unknown_not_rewarded_as_null():
    fixture = sample()
    prediction = deepcopy(fixture["ground_truth"])
    prediction["address"] = "虚构地址"
    scored = score_completion(raw(prediction), fixture)
    assert scored["penalties"]["hallucination"] > 0
    assert scored["reward"] < 1
    fixture["annotation"]["field_status"]["address"] = "unannotated"
    fixture["ground_truth"]["address"] = None
    with_unknown = score_completion(raw(prediction), fixture)
    prediction["address"] = None
    with_null = score_completion(raw(prediction), fixture)
    assert with_unknown["reward"] == with_null["reward"]
    assert with_unknown["penalties"]["hallucination"] == 0


def test_partial_reward_and_invoice_without_items_reweight_only_audited_parts():
    fixture = sample(requested_fields=["total"])
    result = score_completion('{"total":"25.00"}', fixture)
    assert result["reward"] == 1.0
    assert result["applicable_mask"] == {
        "schema": True, "keyfields": True, "items": False, "fidelity": True, "consistency": False,
    }
    assert result["weight_denominator"] == pytest.approx(0.55)
    invoice = sample(schema_version="invoice_json_v1", requested_fields=["SELLER", "TOTAL"],
                     ground_truth={"SELLER": "示例", "TOTAL": 0},
                     annotation={"field_status": {"SELLER": "present_readable", "TOTAL": "present_readable"},
                                 "reviewed": True})
    assert score_completion(raw(invoice["ground_truth"]), invoice)["reward"] == 1.0


def test_schema_only_or_assistant_only_without_annotation_is_rejected():
    fixture = sample(annotation={})
    assert score_completion(raw(fixture["ground_truth"]), fixture)["reward"] == -1.0
    response = raw(fixture["ground_truth"])
    assert receipt_composite_reward([response], assistant=[response]) == [-1.0]
    fixture = sample(annotation={"reviewed": True, "field_status": {}})
    assert score_completion(response, fixture)["reward"] == -1.0


def test_genuine_no_items_with_real_merchant_not_blanket_failure():
    fixture = sample()
    fixture["ground_truth"]["items"] = []
    fixture["annotation"]["field_status"]["items"] = "not_printed"
    fixture["annotation"].pop("subtotal_equation")
    result = score_completion(raw(fixture["ground_truth"]), fixture)
    assert result["reward"] == 1.0
    assert result["components"]["items"] == 1.0
    assert evaluate_sample(raw(fixture["ground_truth"]), fixture)["items"]["f1"] == 0.0


def test_completions_strings_conversation_and_content_parts():
    fixture = sample()
    response = raw(fixture["ground_truth"])
    completions = [response, [{"role": "assistant", "content": response}],
                   [{"role": "assistant", "content": [{"type": "text", "text": response}]}]]
    rewards = receipt_composite_reward(
        completions, ground_truth=[fixture["ground_truth"]] * 3,
        annotation=[fixture["annotation"]] * 3,
        schema_version=["receipt_v1"] * 3, requested_fields=[None] * 3,
    )
    assert rewards == [1.0, 1.0, 1.0]
    assert all(isinstance(value, float) and math.isfinite(value) for value in rewards)
    assert receipt_composite_reward(
        [{"role": "assistant", "content": response}],
        ground_truth=fixture["ground_truth"], annotation=fixture["annotation"],
    ) == [1.0]


@pytest.mark.parametrize("name,value", [
    ("ground_truth", [{"total": "25"}]), ("assistant", ["{}"]),
    ("annotation", [{}]), ("schema_version", ["receipt_v1"]),
    ("requested_fields", [["total"]]), ("finish_reason", ["stop"]),
])
def test_batch_metadata_length_is_strict(name, value):
    with pytest.raises(ValueError, match="长度"):
        receipt_composite_reward(["{}", "{}"], **{name: value})


def test_shared_requested_field_list_not_misread_as_batch():
    fixture = sample()
    assert receipt_composite_reward(
        ['{"total":"25.00"}', '{"total":"25.00"}'],
        ground_truth=fixture["ground_truth"], annotation=fixture["annotation"], requested_fields=["total"],
    ) == [1.0, 1.0]
    assert receipt_composite_reward([]) == []


def test_aggregate_empty_input_and_json_serializable():
    report = aggregate_metrics([])
    assert report["sample_count"]["value"] == 0
    assert report["json_valid_rate"]["denominator"] == 0
    assert report["json_valid_rate"]["value"] is None
    json.dumps(report, allow_nan=False)
    fixture = sample()
    json.dumps(aggregate_metrics([evaluate_sample(raw(fixture["ground_truth"]), fixture)]), allow_nan=False)


def test_invalid_readable_reference_is_not_silently_reviewed():
    fixture = sample()
    fixture["ground_truth"]["total"] = None
    result = evaluate_sample(raw(fixture["ground_truth"]), fixture)
    assert result["annotation_issues"]
    assert result["whole_document_exact"] is None
    assert score_completion(raw(fixture["ground_truth"]), fixture)["reward"] == -1.0
