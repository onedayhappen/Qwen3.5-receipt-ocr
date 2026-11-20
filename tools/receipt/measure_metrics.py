#!/usr/bin/env python3
"""在已构建的 CORD manifest 上实测标注覆盖、奖励判别力与匹配算法差异。

只读 manifest 与训练配置，不加载模型、不依赖 GPU，用于给评估/奖励设计提供可复现的
量化依据。结果写入 reports/metrics_baseline.json。
"""
import argparse
import copy
import json
import statistics
import sys
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from receipt.matching import ITEM_FIELDS, _pair, match_items
from receipt.normalization import decimal_value, normalize_value
from receipt.rewards import score_completion
from receipt.schema import KEY_FIELDS, RECEIPT_FIELDS
from receipt.validation import validate_document

ROOT = Path(__file__).resolve().parents[2]


def load_rows(root):
    path = Path(root) / "data/receipt/manifests/manifest.jsonl"
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def annotation_coverage(by_split):
    coverage = {}
    for split, rows in by_split.items():
        slots = len(rows) * len(RECEIPT_FIELDS)
        key_slots = len(rows) * len(KEY_FIELDS)
        supervised = key_supervised = 0
        per_field = Counter()
        for row in rows:
            statuses = row["annotation"]["field_status"]
            for field in RECEIPT_FIELDS:
                if statuses.get(field) in ("present_readable", "not_printed"):
                    supervised += 1
                    per_field[field] += 1
            for field in KEY_FIELDS:
                if statuses.get(field) in ("present_readable", "not_printed"):
                    key_supervised += 1
        coverage[split] = {
            "samples": len(rows), "field_slots": slots, "supervised_slots": supervised,
            "coverage_pct": round(100 * supervised / slots, 2),
            "key_slots": key_slots, "key_supervised": key_supervised,
            "key_coverage_pct": round(100 * key_supervised / key_slots, 2),
            "per_field": dict(per_field),
        }
    return coverage


def _sample(row):
    return {
        "ground_truth": row["ground_truth"], "annotation": row["annotation"],
        "schema_version": row["schema_version"], "requested_fields": row.get("requested_fields"),
    }


def _negative_payloads(truth):
    if isinstance(truth.get("items"), list) and truth["items"]:
        missing = copy.deepcopy(truth)
        missing["items"] = missing["items"][:-1]
        yield "missing_items", missing
        wrong = copy.deepcopy(truth)
        value = decimal_value(normalize_value("price", wrong["items"][0].get("price")))
        if value is not None:
            wrong["items"][0]["price"] = str(value + Decimal("1"))
            yield "wrong_price", wrong
    if truth.get("total") is not None:
        missing = copy.deepcopy(truth)
        missing.pop("total")
        yield "missing_key_field", missing


def reward_discrimination(rows):
    gold_scores, negatives = [], defaultdict(list)
    margins, mask_hits, eligible = [], Counter(), 0
    for row in rows:
        sample = _sample(row)
        gold = score_completion(json.dumps(row["ground_truth"], ensure_ascii=False), sample)
        gold_scores.append(gold["reward"])
        if gold["eligible"]:
            eligible += 1
            for name, active in gold["applicable_mask"].items():
                if active:
                    mask_hits[name] += 1
        for kind, payload in _negative_payloads(row["ground_truth"]):
            score = score_completion(json.dumps(payload, ensure_ascii=False), sample)
            negatives[kind].append(score["reward"])
            margins.append(gold["reward"] - score["reward"])
    flat = [value for values in negatives.values() for value in values]
    return {
        "n_train": len(rows), "eligible": eligible,
        "eligible_pct": round(100 * eligible / len(rows), 2),
        "mask_component_hits": dict(mask_hits),
        "gold_mean": round(statistics.mean(gold_scores), 4),
        "gold_min": round(min(gold_scores), 4),
        "gold_eq_1_pct": round(100 * sum(1 for x in gold_scores if x == 1.0) / len(gold_scores), 2),
        "negatives": {
            kind: {
                "n": len(values), "mean": round(statistics.mean(values), 4),
                "max": round(max(values), 4),
                "below_gold_pct": round(100 * sum(1 for x in values if x < 1.0) / len(values), 2),
            } for kind, values in negatives.items()
        },
        "neg_mean": round(statistics.mean(flat), 4),
        "margin_mean": round(statistics.mean(margins), 4),
        "margin_median": round(statistics.median(margins), 4),
        "margin_min": round(min(margins), 4),
    }


def positional_f1(pred_items, gold_items):
    """朴素基线：按行序一一配对，不重排、不复用。"""
    preds = [{f: p.get(f) or None for f in ITEM_FIELDS} for p in pred_items]
    golds = [{f: g.get(f) or None for f in ITEM_FIELDS} for g in gold_items]
    correct = sum(_pair(preds[i], golds[i])[1] for i in range(min(len(preds), len(golds))))
    precision = correct / len(preds) if preds else 0.0
    recall = correct / len(golds) if golds else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _bump_price(items):
    out = copy.deepcopy(items)
    value = decimal_value(normalize_value("price", out[0].get("price")))
    if value is not None:
        out[0]["price"] = str(value + Decimal("1"))
    return out


PERTURBATIONS = {
    "oracle": lambda items: copy.deepcopy(items),
    "order_shuffle": lambda items: list(reversed(copy.deepcopy(items))),
    "drop_last_item": lambda items: copy.deepcopy(items)[:-1],
    "duplicate_first": lambda items: [copy.deepcopy(items[0])] + copy.deepcopy(items),
    "price_off_by_1": _bump_price,
}


def matching_gap(rows):
    eligible = [
        row for row in rows
        if row["annotation"].get("items_complete")
        and isinstance(row["ground_truth"].get("items"), list)
        and len(row["ground_truth"]["items"]) >= 2
    ]
    report = {}
    for name, perturb in PERTURBATIONS.items():
        hungarian, positional = [], []
        for row in eligible:
            gold = row["ground_truth"]["items"]
            pred = perturb(gold)
            hungarian.append(match_items(pred, gold)["f1"])
            positional.append(positional_f1(pred, gold))
        report[name] = {
            "n": len(eligible),
            "hungarian_f1": round(statistics.mean(hungarian), 4),
            "positional_f1": round(statistics.mean(positional), 4),
            "delta_pt": round(100 * (statistics.mean(hungarian) - statistics.mean(positional)), 2),
        }
    return report


def consistency_coverage(rows):
    checks, with_check = Counter(), 0
    for row in rows:
        result = validate_document(row["ground_truth"], row["annotation"],
                                   row["schema_version"], row.get("requested_fields"))
        if result["applicable_checks"]:
            with_check += 1
        for name in result["applicable_checks"]:
            checks[name] += 1
        for name, check in result["check_results"].items():
            if not check["passed"]:
                checks["failed:" + name] += 1
    return {"samples_with_check": with_check,
            "pct": round(100 * with_check / len(rows), 2), "checks": dict(checks)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="实测票据评估与奖励的量化基线")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--output", default="reports/metrics_baseline.json")
    args = parser.parse_args(argv)
    rows = load_rows(args.root)
    by_split = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)
    report = {
        "samples": len(rows),
        "split_counts": {split: len(items) for split, items in by_split.items()},
        "annotation_coverage": annotation_coverage(by_split),
        "reward_discrimination": reward_discrimination(by_split["train"]),
        "item_matching": matching_gap(by_split["dev"]),
        "consistency": consistency_coverage(rows),
        "notes": [
            "只读 manifest，不加载模型；无模型精度含义",
            "扰动集为确定性构造，用于检验评估器与奖励的判别力，不是模型输出",
        ],
    }
    path = Path(args.root) / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
