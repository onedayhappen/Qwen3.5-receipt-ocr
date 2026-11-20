"""所有渲染图片与源标注都是 synthetic，仅验证 CPU 工程契约。"""
import copy
import csv
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from receipt.data import (audit_main, audit_rows, build_preferences, export_rows, make_fixtures,
                          read_jsonl, safe_path, validate_sample, render_user)
from receipt.sources import cord_target, fatura_target, prepare_cord, prepare_fatura, template_split


def synthetic_png(index):
    image = Image.new("RGB", (160 + index, 100), "white")
    ImageDraw.Draw(image).text((5, 5), f"SYNTHETIC {index} NOT REAL", fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def fixtures(tmp_path):
    return tmp_path, make_fixtures(tmp_path, 6)


def test_cord_unknown_fields_never_become_null_supervision():
    truth, annotation, used = cord_target({"synthetic": True, "gt_parse": {
        "menu": [{"nm": "SYNTHETIC TEA", "price": "10.00"}, {"nm": "SYNTHETIC BREAD", "cnt": "2", "price": "6.00"}],
        "total": {"total_price": "16.00"}, "subtotal": {"subtotal_price": "16.00", "tax_price": "0.00"}}})
    assert set(truth) == {"subtotal", "tax", "total"}
    assert "items" not in truth  # A metadata mask alone does not remove SFT null loss.
    assert annotation["source_item_field_status"][0]["quantity"] == "unannotated"
    assert annotation["suppressed_fields"] == ["items"]
    assert annotation["source_items_complete"] is True
    assert annotation["field_status"]["merchant_name"] == "unannotated"
    assert annotation["items_complete"] is False
    assert annotation["price_basis"] == "line"
    assert "menu.price" in annotation["price_basis_source"]
    assert annotation["total_equation"] is None
    assert annotation["human_reviewed_this_round"] is False
    assert "total.total_price" in used


def test_cord_actual_sub_total_and_nested_menu_projection():
    truth, annotation, used = cord_target({"gt_parse": {
        "menu": [{"nm": "SYNTHETIC PARENT", "cnt": "1", "price": "6,000",
                  "sub": [{"nm": "SYNTHETIC CHILD", "cnt": "1", "price": "0"}]}],
        "sub_total": {"subtotal_price": "6,000", "tax_price": "600"},
        "total": {"total_price": "6,600"}}})
    assert truth["subtotal"] == "6,000" and truth["tax"] == "600"
    assert "sub_total.subtotal_price" in used and "sub_total.tax_price" in used
    assert [item["name"] for item in truth["items"]] == ["SYNTHETIC PARENT", "SYNTHETIC CHILD"]
    assert truth["items"][1]["quantity"] == "1"
    assert annotation["items_complete"] is True


def test_unknown_item_attributes_rejected_in_training_target(fixtures):
    root, rows = fixtures
    broken = copy.deepcopy(rows[0])
    broken["ground_truth"]["items"][0]["quantity"] = None
    broken["annotation"]["item_field_status"][0]["quantity"] = "unannotated"
    assert any("未经审定商品属性" in e for e in validate_sample(broken, root))
    with pytest.raises(ValueError, match="未经审定商品属性"):
        export_rows([broken], root)
    # Explicitly verified absence may be supervised as null.
    broken["annotation"]["item_field_status"][0]["quantity"] = "not_printed"
    assert not validate_sample(broken, root)


def test_cord_only_incomplete_items_retains_raw_but_is_excluded(tmp_path):
    raw = tmp_path / "raw"
    (raw / "train/images").mkdir(parents=True)
    (raw / "train/json").mkdir()
    (raw / "train/images/one.png").write_bytes(synthetic_png(11))
    (raw / "train/json/one.json").write_text(json.dumps({"gt_parse": {
        "menu": [{"nm": "SYNTHETIC UNKNOWN QUANTITY", "price": "5"}]}}), encoding="utf-8")
    output = tmp_path / "receipt"
    assert prepare_cord(raw, output) == []
    report = json.loads((output / "manifests/cord-report.json").read_text(encoding="utf-8"))
    assert report["raw_source_samples"] == 1 and len(report["excluded"]) == 1
    assert report["suppressed_item_documents"] == 1
    assert (output / report["excluded"][0]["source_gt"]).is_file()
    assert (output / report["excluded"][0]["image"]).is_file()


def test_cord_incomplete_menu_not_supervised():
    truth, annotation, _ = cord_target({"gt_parse": {"menu": [{"price": "4.00"}], "total": {"total_price": "4.00"}}})
    assert "items" not in truth
    assert annotation["items_complete"] is False
    with pytest.raises(ValueError):
        cord_target({"total": "4.00"})


def test_cord_parquet_copies_original_media_and_source_gt(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    root, output = tmp_path / "raw", tmp_path / "receipt"
    root.mkdir()
    originals = []
    for index, split in enumerate(("train", "validation", "test")):
        blob = synthetic_png(index)
        originals.append(blob)
        raw = json.dumps({"synthetic": True, "gt_parse": {"menu": [{"nm": f"SYNTHETIC {index}", "price": "1.00"}], "total": {"total_price": "1.00"}}})
        pq.write_table(pa.Table.from_pylist([{"image": {"bytes": blob, "path": None}, "ground_truth": raw}]), root / f"{split}-00000.parquet")
    rows = prepare_cord(root, output)
    assert {r["split"] for r in rows} == {"train", "dev", "test"}
    assert len(list((output / "images").glob("*"))) == 3
    for row in rows:
        assert (output / row["image"]).read_bytes() in originals
        assert (output / row["provenance"]["source_gt"]).is_file()
        assert not Path(row["image"]).is_absolute()
        assert row["annotation"]["field_status"]["date"] == "unannotated"
        assert not validate_sample(row, output)
    assert not audit_rows(rows, output)["errors"]


def test_cord_official_images_json(tmp_path):
    raw = tmp_path / "raw"
    (raw / "train/images").mkdir(parents=True)
    (raw / "train/json").mkdir()
    blob = synthetic_png(3)
    (raw / "train/images/0001.png").write_bytes(blob)
    (raw / "train/json/0001.json").write_text(json.dumps({"synthetic": True, "gt_parse": {"total": {"total_price": "8.00"}}}), encoding="utf-8")
    rows = prepare_cord(raw, tmp_path / "receipt")
    assert len(rows) == 1
    assert (tmp_path / "receipt" / rows[0]["image"]).read_bytes() == blob


def test_cord_direct_images_json_requires_explicit_split(tmp_path):
    raw = tmp_path / "raw"
    (raw / "images").mkdir(parents=True)
    (raw / "json").mkdir()
    (raw / "images/synthetic.png").write_bytes(synthetic_png(7))
    (raw / "json/synthetic.json").write_text(json.dumps({"synthetic": True, "gt_parse": {"total": {"total_price": "7"}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="显式"):
        prepare_cord(raw, tmp_path / "rejected")
    rows = prepare_cord(raw, tmp_path / "receipt", explicit_split="dev")
    assert rows[0]["split"] == "dev"


def test_fatura_placeholder_and_missing_annotations():
    truth, statuses = fatura_target({"NAME": {"text": "SYNTHETIC SELLER", "bbox": [1, 2, 3, 4]},
        "TABLE": [[{}]], "OTHER": ["SYNTHETIC OTHER"], "NULL_UNKNOWN": None,
        "ABSENT": {"text": None, "status": "not_printed"}, "UNREADABLE": {"text": "?", "status": "present_unreadable"}})
    assert truth == {"NAME": "SYNTHETIC SELLER", "ABSENT": None}
    assert statuses["TABLE"] == "unannotated"
    assert statuses["NULL_UNKNOWN"] == "unannotated"
    assert statuses["UNREADABLE"] == "present_unreadable"
    assert "OTHER" not in statuses


def make_synthetic_fatura(root):
    (root / "images").mkdir(parents=True)
    (root / "Annotations/Original_Format").mkdir(parents=True)
    for index, split in enumerate(("train", "dev", "test"), 1):
        name = f"Template{index}_synthetic.png"
        gt_name = f"Template{index}_synthetic.json"
        (root / "images" / name).write_bytes(synthetic_png(index))
        raw = {"NAME": {"text": f"SYNTHETIC {index}", "bbox": [0, 0, 4, 4]}, "TABLE": [[{}]], "OTHER": "SYNTHETIC"}
        if split == "test":
            raw["TEST_ONLY_FIELD"] = "SYNTHETIC DO NOT LEARN SCHEMA"
        (root / "Annotations/Original_Format" / gt_name).write_text(json.dumps(raw), encoding="utf-8")
        with (root / f"strat1_{split}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["img_path", "annot_path"])
            writer.writeheader()
            writer.writerow({"img_path": name, "annot_path": gt_name})


def test_fatura_schema_train_only_and_bbox_provenance(tmp_path):
    raw, output = tmp_path / "raw/invoices_dataset_final", tmp_path / "receipt"
    make_synthetic_fatura(raw)
    rows = prepare_fatura(raw.parent, output, "legacy")
    schema = json.loads((output / "manifests/invoice_fatura_v1.schema.json").read_text(encoding="utf-8"))
    assert schema["fields"] == ["NAME"]
    assert all(r["requested_fields"] == ["NAME"] for r in rows)
    assert all("TABLE" not in r["ground_truth"] and "OTHER" not in r["ground_truth"] for r in rows)
    saved = json.loads((output / rows[0]["provenance"]["source_gt"]).read_text(encoding="utf-8"))
    assert "bbox" in saved["NAME"]
    assert not audit_rows(rows, output)["errors"]


def test_strict_template_group_disjoint(tmp_path):
    root = tmp_path / "raw"
    make_synthetic_fatura(root)
    rows = prepare_fatura(root, tmp_path / "receipt", "strict")
    groups = [{r["template_group"] for r in rows if r["split"] == split} for split in ("train", "dev", "test")]
    assert all(groups)
    assert not groups[0] & groups[1] and not groups[1] & groups[2] and not groups[0] & groups[2]
    with pytest.raises(ValueError):
        template_split(["one", "two"])


def test_sft_grpo_export_mask_is_actual_prompt_and_target(fixtures):
    root, rows = fixtures
    train = [r for r in rows if r["split"] == "train"]
    sft = export_rows(train, root)
    grpo = export_rows(train, root, kind="grpo")
    assert set(sft[0]) == {"image", "conversations"}
    assert [t["from"] for t in sft[0]["conversations"]] == ["human", "gpt"]
    assert sft[0]["conversations"][0]["value"].count("<image>") == 1
    assert set(json.loads(sft[0]["conversations"][1]["value"])) == set(train[0]["requested_fields"])
    prompt = sft[0]["conversations"][0]["value"]
    assert train[0]["sample_id"] not in prompt and "synthetic_generator" not in prompt
    assert "review_source" not in prompt and "items_complete" not in prompt
    assert grpo[0]["annotation"] == train[0]["annotation"]
    assert grpo[0]["split"] == "train" and grpo[0]["requested_fields"] == train[0]["requested_fields"]


def test_train_export_and_preferences_reject_test_or_mixed(fixtures):
    root, rows = fixtures
    test = [r for r in rows if r["split"] == "test"]
    with pytest.raises(ValueError):
        export_rows(rows, root)
    with pytest.raises(ValueError):
        export_rows(test, root, split="test", kind="sft")
    with pytest.raises(ValueError):
        build_preferences(test, root, split="test", synthetic_negatives=True)
    assert export_rows(test, root, split="test", kind="evaluation")


@pytest.mark.parametrize("path", ["../escape.png", "/outside.png", "D:/outside.png", "images\\bad.png", "C:bad.png"])
def test_media_paths_cannot_escape_root(tmp_path, path):
    with pytest.raises(ValueError):
        safe_path(tmp_path, path)


def test_audit_detects_hash_source_parent_and_labels(fixtures):
    root, rows = fixtures
    assert not audit_rows(rows, root)["errors"]
    broken = copy.deepcopy(rows)
    broken[-1]["source_document_id"] = broken[0]["source_document_id"]
    broken[-1]["provenance"]["augmentation_parent_id"] = broken[1]["sample_id"]
    broken[-1]["image"] = broken[0]["image"]
    broken[-1]["source_hash"] = broken[0]["source_hash"]
    broken[0]["annotation"]["field_status"]["total"] = "unannotated"
    errors = audit_rows(broken, root)["errors"]
    assert any("source_document_id" in e for e in errors)
    assert any("source_hash" in e for e in errors)
    assert any("parent" in e for e in errors)
    assert any("未经审定" in e for e in errors)
    broken[0]["conversations"] = [{"from": "system", "value": "x"}]
    assert any("角色" in e for e in audit_rows(broken, root)["errors"])


def test_audit_conflict_cli_nonzero(fixtures):
    root, rows = fixtures
    rows[-1]["source_document_id"] = rows[0]["source_document_id"]
    manifest = root / "conflict.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        audit_main(["--manifest", str(manifest), "--data-root", str(root), "--report", str(root / "audit.json")])
    assert exc.value.code == 1


def test_preferences_synthetic_and_review_gate(fixtures):
    root, rows = fixtures
    train = [r for r in rows if r["split"] == "train"]
    values, metadata = build_preferences(train, root, synthetic_negatives=True, max_pairs=2)
    assert values and len(values) <= 2 * len(train)
    assert set(values[0]) == {"image", "prompt", "chosen", "rejected"}
    assert metadata["synthetic_pairs"] == len(values)
    assert all(m["candidate"]["provenance"] == "synthetic_negatives" for m in metadata["pair_metadata"])
    candidate = {"sample_id": train[0]["sample_id"], "response": "{}", "model_version": "synthetic-model", "prompt_version": "synthetic-prompt", "reviewed": False, "prompt": render_user(train[0])}
    values, metadata = build_preferences(train, root, [candidate])
    assert not values and metadata["filtered"]["unreviewed"] == 1
    candidate["reviewed"] = True
    values, metadata = build_preferences(train, root, [candidate])
    assert len(values) == 1 and metadata["synthetic_pairs"] == 0
    candidate["sample_id"] = rows[-1]["sample_id"]
    with pytest.raises(ValueError):
        build_preferences(train, root, [candidate])


def test_dev_preferences_are_evaluation_and_not_implicit(fixtures):
    root, rows = fixtures
    dev = [r for r in rows if r["split"] == "dev"]
    with pytest.raises(ValueError):
        build_preferences(dev, root, synthetic_negatives=True)
    values, report = build_preferences(dev, root, split="dev", synthetic_negatives=True)
    assert values and report["purpose"] == "evaluation"
    assert all(m["split"] == "dev" for m in report["pair_metadata"])
    assert all(m["synthetic_source"] and m["purpose"] == "cpu_regression" for m in report["pair_metadata"])


def test_preferences_identical_margin_prompt_schema_gates(fixtures):
    root, rows = fixtures
    train = [r for r in rows if r["split"] == "train"]
    candidate = {"sample_id": train[0]["sample_id"], "response": json.dumps(train[0]["ground_truth"]),
                 "model_version": "synthetic-model", "prompt_version": "synthetic-prompt", "reviewed": True, "prompt": render_user(train[0])}
    values, report = build_preferences(train, root, [candidate])
    assert not values and report["filtered"]["identical"] == 1
    candidate["response"] = "{}"
    candidate["schema_version"] = "different"
    with pytest.raises(ValueError, match="schema"):
        build_preferences(train, root, [candidate])
    candidate.pop("schema_version")
    candidate["prompt"] = "SYNTHETIC WRONG PROMPT"
    with pytest.raises(ValueError, match="prompt"):
        build_preferences(train, root, [candidate])
    candidate["prompt"] = render_user(train[0])
    no_prompt = {k: v for k, v in candidate.items() if k != "prompt"}
    with pytest.raises(ValueError, match="prompt"):
        build_preferences(train, root, [no_prompt])
    second = {**candidate, "prompt_version": "synthetic-different"}
    with pytest.raises(ValueError, match="prompt_version"):
        build_preferences(train, root, [candidate, second])
    with pytest.raises(ValueError):
        build_preferences(train, root, [candidate], min_margin=0)


def test_fixture_count_and_manifest(fixtures):
    root, rows = fixtures
    assert len(rows) == 6 and len(list((root / "images").glob("*"))) == 6
    assert all(r["provenance"]["synthetic"] and "synthetic" in r["quality_tags"] for r in rows)
    assert read_jsonl(root / "manifests/manifest.jsonl") == rows
    with pytest.raises(ValueError):
        make_fixtures(root / "too_many", 1000)


@pytest.mark.parametrize("script", ["prepare_cord", "prepare_fatura", "export_training_data", "build_preferences", "audit_splits", "make_fixtures"])
def test_all_cli_help_from_outside_repo(tmp_path, script):
    repo = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, str(repo / "tools/receipt" / f"{script}.py"), "--help"], cwd=tmp_path, capture_output=True)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert b"usage:" in result.stdout


@pytest.mark.parametrize("command", ["prepare_cord", "prepare_fatura", "export_training_data", "build_preferences", "audit_splits", "make_fixtures"])
def test_installed_module_cli_dispatch_help(command, capsys):
    from receipt.data import main
    with pytest.raises(SystemExit) as exc:
        main([command, "--help"])
    assert exc.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_near_duplicate_audit_is_warning_not_auto_deletion(fixtures):
    root, rows = fixtures
    report = audit_rows(rows, root, near_duplicates=True, near_threshold=64)
    assert report["near_duplicates"] and report["warnings"]
    assert not report["errors"]
    assert len(list((root / "images").glob("*"))) == len(rows)


def test_invalid_media_and_status_are_rejected(fixtures):
    root, rows = fixtures
    broken = copy.deepcopy(rows[0])
    broken["annotation"]["field_status"] = "unannotated"
    assert validate_sample(broken, root)
    broken = copy.deepcopy(rows[0])
    (root / broken["image"]).write_bytes(b"SYNTHETIC BROKEN IMAGE")
    assert any("SHA-256" in error for error in validate_sample(broken, root))
