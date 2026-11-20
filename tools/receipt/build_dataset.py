"""Reproducible local build: CORD/FATURA manifests, split audit and train/dev exports."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from receipt.data import read_jsonl, write_json, write_manifests, audit_rows, export_rows, build_preferences
from receipt.data import register_source_schema
from receipt.sources import prepare_cord, prepare_fatura


def build(root, synthetic_preferences=False, include_fatura=True):
    raw = root / "data/raw"
    data = root / "data/receipt"
    cord = prepare_cord(raw / "cord", data)
    rows = list(cord)
    fatura_dir = raw / "fatura"
    if include_fatura:
        if not (fatura_dir / "invoices_dataset_final").exists():
            raise FileNotFoundError("FATURA raw directory missing; download/extract first or explicitly use --cord-only")
        rows.extend(prepare_fatura(fatura_dir, data, split_mode="strict", seed=42))
        register_source_schema(data, "invoice_fatura_v1")
    write_manifests(rows, data, "manifest")
    audit = audit_rows(rows, data)
    if audit["errors"]:
        raise ValueError(f"Combined audit errors: {audit['errors'][:5]}")
    outputs = {}
    for split in ("train", "dev"):
        selected = [row for row in rows if row["split"] == split]
        for kind in ("sft", "grpo"):
            values = export_rows(selected, data, split, kind)
            if kind == "sft":
                # Preserve metadata only for generation evaluation; dataset never sends it to the model.
                for value, sample in zip(values, selected, strict=True):
                    value.update({key: sample[key] for key in ("sample_id", "split", "schema_version", "ground_truth", "annotation", "requested_fields")})
            write_json(data / "exports" / f"{kind}_{split}.json", values)
            outputs[f"{kind}_{split}"] = len(values)
        if synthetic_preferences:
            pairs, report = build_preferences(selected, data, split=split, synthetic_negatives=True, max_pairs=2)
            write_json(data / "exports" / f"dpo_{split}_synthetic.json", pairs)
            write_json(data / "exports" / f"dpo_{split}_synthetic_metadata.json", report)
            outputs[f"dpo_{split}_synthetic"] = len(pairs)
    # Test is retained solely as manifest. Never create a test training export.
    report = {"datasets": ["CORD-v2"] + (["FATURA"] if include_fatura else []), "audit": audit, "exports": outputs,
              "synthetic_preferences": synthetic_preferences, "test_training_exports": False,
              "gpu_training_run": False, "model_generated_preferences": False,
              "limitations": ["CORD has no public merchant/date/address labels", "No new human double-review", "Official CORD splits do not establish merchant OOD", "Synthetic negatives are not model-generated DPO preferences", "Whole items target omitted if any item attribute is unannotated; source GT retained; no token-level masking claimed"]}
    write_json(root / "reports/dataset_build.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--cord-only", action="store_true")
    parser.add_argument("--synthetic-preferences", action="store_true")
    args = parser.parse_args(argv)
    result = build(args.root.resolve(), args.synthetic_preferences, not args.cord_only)
    print(json.dumps({"exports": result["exports"], "samples": result["audit"]["samples"], "split_counts": result["audit"]["split_counts"]},indent=2))

if __name__ == "__main__":
    main()
