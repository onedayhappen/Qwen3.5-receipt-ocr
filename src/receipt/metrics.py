"""审定状态感知的指标；失败保留在分母，raw 与 normalized 分开。"""

from collections import Counter, defaultdict
import json

from .io import parse_json_object
from .matching import match_items
from .normalization import normalize_document, normalize_value
from .schema import KEY_FIELDS, get_schema
from .validation import validate_document

parse_response = parse_json_object
_TRUNCATED = frozenset({"length", "max_tokens", "max_length", "max_completion_length", "truncated"})
_AUDITED = frozenset({"present_readable", "not_printed"})


def has_content(value):
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return any(has_content(child) for child in value.values())
    if isinstance(value, list):
        return any(has_content(child) for child in value)
    return True


def requested_sample_fields(sample):
    version = sample.get("schema_version") or "receipt_v1"
    fields = sample.get("requested_fields")
    schema = get_schema(version, fields)
    if fields is not None:
        return list(schema["required"])
    properties = schema.get("properties")
    return list(properties) if properties is not None else list(sample.get("ground_truth") or {})


def sample_key_fields(sample, requested=None):
    requested = requested_sample_fields(sample) if requested is None else requested
    annotation = sample.get("annotation") or {}
    configured = annotation.get("key_fields")
    if configured is not None:
        return [field for field in configured if field in requested and field != "items"]
    if (sample.get("schema_version") or "receipt_v1") == "receipt_v1":
        return [field for field in KEY_FIELDS if field in requested]
    return [field for field in requested if field != "items"]


def _item_equal(pred, gold, normalized=False):
    if not isinstance(pred, list) or not isinstance(gold, list):
        return pred == gold
    if normalized:
        pred = normalize_value("items", pred)
        gold = normalize_value("items", gold)
    encode = lambda row: json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return Counter(map(encode, pred)) == Counter(map(encode, gold))


def evaluate_sample(raw_response, sample, finish_reason=None):
    if not isinstance(sample, dict):
        raise TypeError("sample 必须是 dict")
    version = sample.get("schema_version") or "receipt_v1"
    fields = sample.get("requested_fields")
    requested = requested_sample_fields(sample)
    annotation = sample.get("annotation") or {}
    if not isinstance(annotation, dict):
        raise TypeError("annotation 必须是 dict")
    gold = sample.get("ground_truth") or {}
    if not isinstance(gold, dict):
        raise TypeError("ground_truth 必须是 dict")
    statuses = annotation.get("field_status") or {}
    prediction = parse_response(raw_response)
    validation = validate_document(prediction, annotation, version, fields)
    truncated = (finish_reason if finish_reason is not None else sample.get("finish_reason")) in _TRUNCATED
    usable = prediction is not None and not truncated
    pred = prediction or {}
    normalized = normalize_document(pred)
    normalized_gold = normalize_document(gold)
    field_metrics, annotation_issues = {}, []
    absent_count, hallucinations = 0, 0
    for field in requested:
        status = statuses.get(field, "unannotated")
        applicable = status in _AUDITED and field in gold
        if status == "present_readable" and not has_content(gold.get(field)):
            annotation_issues.append(f"{field}: readable_without_ground_truth")
            applicable = False
        if status == "not_printed" and has_content(gold.get(field)):
            annotation_issues.append(f"{field}: not_printed_with_ground_truth")
            applicable = False
        if status in _AUDITED and field not in gold:
            annotation_issues.append(f"{field}: missing_ground_truth")
        raw_em = normalized_em = None
        if applicable:
            present = usable and field in pred
            if field == "items":
                raw_em = bool(present and _item_equal(pred[field], gold[field]))
                normalized_em = bool(present and _item_equal(pred[field], gold[field], True))
            else:
                raw_em = bool(present and type(pred[field]) is type(gold[field]) and pred[field] == gold[field])
                normalized_em = bool(present and normalized[field] == normalized_gold[field])
            if status == "not_printed" and field != "items":
                absent_count += 1
                hallucinations += int(prediction is not None and has_content(pred.get(field)))
        field_metrics[field] = {
            "status": status, "applicable": applicable,
            "raw_em": raw_em, "normalized_em": normalized_em,
        }
    items_eligible = "items" in requested and annotation.get("items_complete") is True \
        and field_metrics["items"]["applicable"] and isinstance(gold.get("items"), list)
    items = None
    if items_eligible:
        candidate_items = pred.get("items") if usable else None
        invalid_items = not isinstance(candidate_items, list) or any(
            not isinstance(row, dict) for row in candidate_items
        )
        items = match_items([] if invalid_items else candidate_items, gold["items"])
        items["valid_structure"] = not invalid_items
        if invalid_items:
            items["exact"] = False
    full_schema = get_schema(version)
    full_fields = set(full_schema.get("properties", {}))
    if version == "invoice_json_v1":
        full_fields = set(gold) if annotation.get("document_complete") is True else set()
    full_audit = bool(full_fields) and set(requested) == full_fields \
        and all(field_metrics[field]["applicable"] for field in requested) \
        and annotation.get("reviewed") is not False and not annotation_issues
    if "items" in requested:
        full_audit = full_audit and annotation.get("items_complete") is True
    whole_raw = whole_normalized = None
    if full_audit:
        whole_raw = bool(validation["schema_valid"] and usable
                         and all(metric["raw_em"] for metric in field_metrics.values()))
        whole_normalized = bool(validation["schema_valid"] and usable
                                and all(metric["normalized_em"] for metric in field_metrics.values()))
    expected_content = any(statuses.get(field) == "present_readable" and has_content(gold.get(field))
                           for field in requested)
    blank = not isinstance(raw_response, str) or not raw_response.strip()
    empty = blank or prediction == {} or (expected_content and prediction is not None
                                          and not any(has_content(pred.get(field)) for field in requested))
    groups = {}
    for key in ("schema_version", "template_group", "merchant_group", "language", "quality_tags", "split"):
        value = version if key == "schema_version" else sample.get(key)
        if value is not None:
            groups[key] = value
    return {
        "sample_id": sample.get("sample_id"), "schema_version": version,
        "requested_fields": requested, "key_fields": sample_key_fields(sample, requested),
        "json_valid": prediction is not None, "schema_valid": validation["schema_valid"],
        "empty": bool(empty), "truncated": bool(truncated), "field_metrics": field_metrics,
        "annotation_issues": annotation_issues, "items": items,
        "whole_document_exact": whole_raw, "whole_document_raw_em": whole_raw,
        "whole_document_normalized_em": whole_normalized,
        "hallucination": {
            "field_count": hallucinations, "field_denominator": absent_count,
            "field_rate": hallucinations / absent_count if absent_count else None,
            "item_count": items["extra_count"] if items is not None else None,
            "item_denominator": items["pred_count"] if items is not None else None,
        },
        "validation": validation, "groups": groups,
        "annotation_coverage": {
            "numerator": sum(metric["applicable"] for metric in field_metrics.values()),
            "denominator": len(requested),
        },
    }


def _metric(numerator, denominator):
    return {"value": numerator / denominator if denominator else None,
            "numerator": numerator, "denominator": denominator}


def _aggregate(results):
    count = len(results)
    report = {"sample_count": {"value": count, "denominator": count}}
    for key in ("json_valid", "schema_valid", "empty", "truncated"):
        report[key + "_rate"] = _metric(sum(bool(result[key]) for result in results), count)
    by_field = defaultdict(list)
    key_field_results = defaultdict(list)
    for result in results:
        for field, value in result["field_metrics"].items():
            if value["applicable"] and field != "items":
                by_field[field].append(value)
                if field in result["key_fields"]:
                    key_field_results[field].append(value)
    report["fields"] = {
        field: {name: _metric(sum(metric[name] for metric in values), len(values))
                for name in ("raw_em", "normalized_em")}
        for field, values in sorted(by_field.items())
    }
    for name in ("raw_em", "normalized_em"):
        means = [sum(value[name] for value in values) / len(values)
                 for values in key_field_results.values()]
        macro = _metric(sum(means), len(means))
        macro["sample_field_denominator"] = sum(map(len, key_field_results.values()))
        report["key_fields_macro_" + name] = macro
    audited = sum(result["annotation_coverage"]["numerator"] for result in results)
    requested = sum(result["annotation_coverage"]["denominator"] for result in results)
    report["annotation_coverage"] = _metric(audited, requested)
    eligible_items = [result["items"] for result in results if result["items"] is not None]
    report["items_coverage"] = _metric(len(eligible_items), count)
    for name in ("precision", "recall", "f1"):
        report["items_" + name] = _metric(sum(item[name] for item in eligible_items), len(eligible_items))
    report["item_fields"] = {}
    for field in ("name", "quantity", "price"):
        values = [item["field_scores"][field] for item in eligible_items
                  if item["field_scores"][field]["applicable"]]
        report["item_fields"][field] = {
            name: _metric(sum(value[name] for value in values), len(values))
            for name in ("precision", "recall", "f1")
        }
    for name in ("raw_em", "normalized_em"):
        values = [result["whole_document_" + name] for result in results
                  if result["whole_document_" + name] is not None]
        report["whole_document_" + name] = _metric(sum(values), len(values))
    whole_count = report["whole_document_raw_em"]["denominator"]
    report["whole_document_coverage"] = _metric(whole_count, count)
    report["field_hallucination_rate"] = _metric(
        sum(result["hallucination"]["field_count"] for result in results),
        sum(result["hallucination"]["field_denominator"] for result in results),
    )
    report["item_hallucination_rate"] = _metric(
        sum(item["extra_count"] for item in eligible_items),
        sum(item["pred_count"] for item in eligible_items),
    )
    checks = [check for result in results for check in result["validation"]["check_results"].values()]
    passed = sum(check["passed"] and not result["truncated"] and result["schema_valid"]
                 for result in results for check in result["validation"]["check_results"].values())
    report["consistency_pass_rate"] = _metric(passed, len(checks))
    report["consistency_coverage"] = _metric(
        sum(bool(result["validation"]["applicable_checks"]) for result in results), count)
    return report


def aggregate_metrics(samples_results):
    results = list(samples_results)
    report = _aggregate(results)
    groups = defaultdict(lambda: defaultdict(list))
    for result in results:
        for dimension, values in result.get("groups", {}).items():
            values = values if isinstance(values, (list, tuple)) else [values]
            for value in dict.fromkeys(str(value) for value in values):
                groups[dimension][value].append(result)
    report["groups"] = {
        dimension: {value: _aggregate(members) for value, members in sorted(group.items())}
        for dimension, group in sorted(groups.items())
    }
    return report
