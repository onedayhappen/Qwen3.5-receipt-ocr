"""Full-denominator evaluation joined by sample_id; missing responses count as failures."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from receipt.metrics import aggregate_metrics, evaluate_sample


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate_predictions(manifest, predictions, allow_test=False):
    if not allow_test and any(row.get("split") == "test" for row in manifest):
        raise ValueError("Blind test evaluation requires explicit --allow-test")
    refs = {row["sample_id"]: row for row in manifest}
    if len(refs) != len(manifest):
        raise ValueError("Duplicate sample IDs in manifest")
    candidates = {}
    for row in predictions:
        sample_id = row["sample_id"]
        if sample_id in candidates:
            raise ValueError("Duplicate prediction sample ID")
        if sample_id not in refs:
            raise ValueError("Prediction has unknown sample ID")
        candidates[sample_id] = row
    results = []
    for sample_id, sample in refs.items():
        prediction = candidates.get(sample_id, {"response": "", "finish_reason": "missing"})
        scored = evaluate_sample(prediction.get("response", ""), sample, finish_reason=prediction.get("finish_reason"))
        scored["sample_id"] = sample_id
        scored["latency_ms"] = prediction.get("latency_ms")
        results.append(scored)
    summary = aggregate_metrics(results)
    groups = {}
    for sample_id, sample in refs.items():
        group = str(sample.get("provenance", {}).get("source", "unknown")) + ":" + str(sample.get("schema_version", "unknown"))
        groups.setdefault(group, []).append(next(row for row in results if row["sample_id"] == sample_id))
    return {"summary": summary, "groups": {key: aggregate_metrics(rows) for key, rows in groups.items()},
            "prediction_count": len(predictions), "missing_predictions": len(refs) - len(candidates), "sample_results": results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--data-root", type=Path, help="Root holding training-frozen manifests/*.schema.json")
    args = parser.parse_args(argv)
    manifest = read_jsonl(args.manifest)
    from receipt.data import initialize_source_schemas
    root = args.data_root or args.manifest.resolve().parent.parent
    initialize_source_schemas(manifest, root)
    result = evaluate_predictions(manifest, read_jsonl(args.predictions), args.allow_test)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "sample_results"}, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
