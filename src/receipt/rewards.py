"""固定适用 mask 的票据 composite reward，不导入训练或 GPU 模块。"""

import math

from .io import completion_text
from .metrics import evaluate_sample, has_content, parse_response, requested_sample_fields
from .schema import validate_output
from .validation import validate_document

WEIGHTS = {"schema": 0.05, "keyfields": 0.35, "items": 0.35,
           "fidelity": 0.15, "consistency": 0.10}
FIDELITY_GATE = 0.8


def score_completion(raw_response, sample):
    evaluated = evaluate_sample(raw_response, sample)
    annotation = sample.get("annotation") or {}
    gold = sample.get("ground_truth") or {}
    requested = evaluated["requested_fields"]
    audited = {field: metric for field, metric in evaluated["field_metrics"].items()
               if metric["applicable"]}
    keys = [field for field in evaluated["key_fields"] if field in audited]
    fidelity_fields = [field for field in audited if field != "items"]
    items_applicable = evaluated["items"] is not None
    version = sample.get("schema_version") or "receipt_v1"
    projected_gold = {field: gold[field] for field in requested if field in gold}
    gold_validation = validate_document(projected_gold, annotation, version, sample.get("requested_fields"))
    gold_checks = {name: check for name, check in gold_validation["check_results"].items()
                   if check["passed"]}
    mask = {
        "schema": True, "keyfields": bool(keys), "items": items_applicable,
        "fidelity": bool(fidelity_fields), "consistency": bool(gold_checks),
    }
    components = {name: None for name in WEIGHTS}
    result = {
        "reward": -1.0, "components": components, "applicable_mask": mask,
        "weights": dict(WEIGHTS), "weight_denominator": sum(WEIGHTS[name] for name in WEIGHTS if mask[name]),
        "penalties": {"hallucination": 0.0, "extra_items": 0.0},
        "eligible": False, "reason": None,
        "keyfields_raw_em": None, "keyfields_normalized_em": None,
        "consistency_gate_passed": False,
    }
    if annotation.get("reviewed") is not True:
        result["reason"] = "unreviewed_sample"
        return result
    if evaluated["annotation_issues"] or not audited or validate_output(
        projected_gold, version, sample.get("requested_fields")
    ):
        result["reason"] = "invalid_or_unannotated_reference"
        return result
    if not (mask["keyfields"] or mask["items"] or mask["fidelity"]):
        result["reason"] = "schema_only_sample"
        return result
    result["eligible"] = True
    if not evaluated["json_valid"] or not evaluated["schema_valid"] or evaluated["truncated"]:
        result["reason"] = "invalid_or_truncated_output"
        return result
    prediction = parse_response(raw_response)
    if not any(has_content(prediction.get(field)) for field in requested):
        result["reason"] = "empty_extraction"
        return result
    items = evaluated["items"]
    if items_applicable and gold["items"] and (
        not prediction.get("items") or not items["matched"]
    ):
        result["reason"] = "all_items_missed"
        return result
    components["schema"] = 1.0
    if keys:
        raw_em = sum(audited[field]["raw_em"] for field in keys) / len(keys)
        normalized_em = sum(audited[field]["normalized_em"] for field in keys) / len(keys)
        result["keyfields_raw_em"], result["keyfields_normalized_em"] = raw_em, normalized_em
        components["keyfields"] = normalized_em
    if items_applicable:
        components["items"] = float(items["f1"] if gold["items"] else items["exact"])
    if fidelity_fields:
        components["fidelity"] = sum(audited[field]["raw_em"] for field in fidelity_fields) / len(fidelity_fields)
    if gold_checks:
        # 门控失败仍保留金额项权重，候选不能通过漏字段减少分母。
        gate = bool(keys) and components["keyfields"] >= FIDELITY_GATE \
            and (not items_applicable or components["items"] >= FIDELITY_GATE)
        result["consistency_gate_passed"] = gate
        candidate_checks = evaluated["validation"]["check_results"]
        components["consistency"] = sum(
            bool(gate and candidate_checks.get(name, {}).get("passed")) for name in gold_checks
        ) / len(gold_checks)
    hallucination_rate = evaluated["hallucination"]["field_rate"] or 0.0
    extra_rate = items["extra_rate"] if items_applicable else 0.0
    result["penalties"] = {"hallucination": 0.20 * hallucination_rate,
                           "extra_items": 0.15 * extra_rate}
    weighted = sum(WEIGHTS[name] * components[name] for name in WEIGHTS if mask[name])
    reward = weighted / result["weight_denominator"] - sum(result["penalties"].values())
    result["reward"] = float(max(-1.0, min(1.0, reward)))
    if not math.isfinite(result["reward"]):
        raise ValueError("奖励必须是有限浮点数")
    result["reason"] = "scored"
    return result


def _conversation(value):
    return isinstance(value, (list, tuple)) and bool(value) and all(
        isinstance(message, dict) and "role" in message for message in value
    )


def _batch(value, count, name, field_list=False, conversational=False):
    shared = field_list and isinstance(value, (list, tuple)) and all(isinstance(field, str) for field in value)
    shared = shared or (conversational and _conversation(value))
    if isinstance(value, (list, tuple)) and not shared:
        if len(value) != count:
            raise ValueError(f"{name} 长度 {len(value)} 与 completions 长度 {count} 不一致")
        return list(value)
    return [value] * count


def receipt_composite_reward(completions, assistant=None, ground_truth=None, annotation=None,
                             schema_version=None, requested_fields=None, **kwargs):
    if isinstance(completions, (str, dict)) or _conversation(completions):
        completions = [completions]
    if not isinstance(completions, (list, tuple)):
        raise TypeError("completions 必须是字符串或候选列表")
    count = len(completions)
    references = _batch(ground_truth, count, "ground_truth")
    assistants = _batch(assistant, count, "assistant", conversational=True)
    annotations = _batch(annotation, count, "annotation")
    versions = _batch(schema_version, count, "schema_version")
    fields = _batch(requested_fields, count, "requested_fields", field_list=True)
    reasons = _batch(kwargs.get("finish_reason"), count, "finish_reason")
    rewards = []
    for index, completion in enumerate(completions):
        reference = references[index]
        if reference is None:
            reference = parse_response(completion_text(assistants[index]))
        elif isinstance(reference, str):
            reference = parse_response(reference)
        if reference is not None and not isinstance(reference, dict):
            raise TypeError("ground_truth 必须是 JSON 对象")
        sample = {
            "ground_truth": reference or {}, "annotation": annotations[index] or {},
            "schema_version": versions[index] or "receipt_v1",
            "requested_fields": fields[index], "finish_reason": reasons[index],
        }
        reward = score_completion(completion_text(completion), sample)["reward"]
        if not isinstance(reward, float) or not math.isfinite(reward):
            raise ValueError("奖励必须是有限浮点数")
        rewards.append(reward)
    return rewards
