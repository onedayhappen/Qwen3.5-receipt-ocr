"""CORD/FATURA 源标注转换；不从缺失标签推断纸面不存在。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from receipt.data import (SPLITS, audit_rows, json_text, safe_path, store_gt, store_image,
                          write_json, write_manifests)

CORD_MAPPING = {"subtotal": (("sub_total", "subtotal"), "subtotal_price"),
                "tax": (("sub_total", "subtotal"), "tax_price"), "total": (("total",), "total_price")}


def cord_value(group, key):
    if not isinstance(group, dict):
        return None
    value = group.get(key, group.get("menu." + key))
    if isinstance(value, str):
        return value if value.strip() else None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return None


def cord_target(raw):
    from receipt.schema import RECEIPT_FIELDS
    source = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(source, dict) or not isinstance(source.get("gt_parse"), dict):
        raise ValueError("CORD 必须提供含 gt_parse 对象的完整 source ground_truth")
    parsed = source["gt_parse"]
    truth, status, used = {}, {f: "unannotated" for f in RECEIPT_FIELDS}, []
    for field, (groups, key) in CORD_MAPPING.items():
        for group in groups:
            value = cord_value(parsed.get(group), key)
            if value is None:
                value = cord_value(parsed, f"{group}.{key}")
            if value is not None:
                truth[field], status[field] = value, "present_readable"
                used.append(f"{group}.{key}")
                break
    menu = parsed.get("menu")
    if isinstance(menu, dict):
        menu = [menu]
    items, attributes = [], []
    complete = isinstance(menu, list) and bool(menu)
    if complete:
        def flattened(rows):
            for entry in rows:
                yield entry
                children = entry.get("sub") if isinstance(entry, dict) else None
                if isinstance(children, dict):
                    children = [children]
                if isinstance(children, list):
                    yield from flattened(children)
        for row in flattened(menu):
            name = cord_value(row, "nm")
            quantity = cord_value(row, "cnt")
            price = cord_value(row, "price")
            if name is None:
                complete = False
                break
            items.append({"name": name, "quantity": quantity, "price": price})
            attributes.append({"name": "present_readable", "quantity": "present_readable" if quantity is not None else "unannotated",
                               "price": "present_readable" if price is not None else "unannotated"})
        attributes_complete = complete and all(
            all(state == "present_readable" for state in row.values()) for row in attributes
        )
        if attributes_complete:
            truth["items"], status["items"] = items, "present_readable"
            used.append("menu.nm/menu.cnt/menu.price")
    else:
        attributes_complete = False
    # Full-string SFT cannot mask individual JSON attributes. Do not serialize unknowns
    # as null, and do not drop just an incomplete row (which would teach omissions).
    suppressed = ["items"] if complete and not attributes_complete else []
    annotation = {"field_status": status, "items_complete": attributes_complete,
                  "item_field_status": attributes if attributes_complete else [],
                  "source_items_complete": complete,
                  "source_item_field_status": attributes if complete else [],
                  "suppressed_fields": suppressed,
                  "supervision_policy": "omit_whole_items_if_any_attribute_unannotated",
                  "price_basis": "line",
                  "subtotal_equation": None, "total_equation": None, "reviewed": True,
                  "review_source": "cord_public_source_gt", "human_reviewed_this_round": False,
                  "review_limitation": "公开源 GT；本轮未人工复核，更非本轮双审",
                  "price_basis_source": "CORD official category definition: menu.price is total price of menu",
                  "items_projection": "flatten_menu_and_sub_in_source_order; hierarchy retained in source_gt"}
    return truth, annotation, used


def cord_records(input_dir, explicit_split=None):
    root = Path(input_dir)
    parquets = sorted(root.rglob("*.parquet"))
    if parquets:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("parquet 转换需要安装 pyarrow") from exc
        for path in parquets:
            tokens = path.relative_to(root).as_posix().lower()
            source_split = next((s for s in ("validation", "train", "test", "dev")
                                 if re.search(rf"(^|[/_.-]){s}([/_.-]|$)", tokens)), None)
            if source_split is None:
                raise ValueError(f"无法判定 parquet 官方 split: {path.name}")
            split = "dev" if source_split in {"validation", "dev"} else source_split
            offset = 0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
                for row in batch.to_pylist():
                    media = row.get("image")
                    if not isinstance(media, dict):
                        raise ValueError("parquet image 必须是 {bytes,path}")
                    blob = media.get("bytes")
                    if blob is None:
                        if not media.get("path"):
                            raise ValueError("parquet 图片 bytes/path 均缺失")
                        blob = safe_path(root, media["path"]).read_bytes()
                    identity = str(row.get("id") or Path(media.get("path") or f"{path.stem}-{offset}").stem)
                    raw = row.get("ground_truth")
                    yield split, identity, bytes(blob), raw, {
                        "input_format": "cord_v2_parquet", "source_file": path.relative_to(root).as_posix(),
                        "source_row": offset, "official_split": source_split}
                    offset += 1
        return
    count = 0
    split_directories = [(split, root / split) for split in ("train", "validation", "dev", "test")]
    if (root / "json").is_dir():
        if explicit_split not in SPLITS:
            raise ValueError("直接 images/json 输入必须显式 --split，不能猜测原件所属集合")
        split_directories = [(explicit_split, root)]
    for source_split, split_dir in split_directories:
        json_dir = split_dir / "json"
        images_dir = split_dir / "images"
        if not images_dir.is_dir() and (split_dir / "image").is_dir():
            json_dir, images_dir = split_dir / "json", split_dir / "image"
        if not json_dir.is_dir():
            continue
        for path in sorted(json_dir.glob("*.json")):
            images = [image for image in images_dir.glob(path.stem + ".*") if image.suffix.lower() in {".png", ".jpg", ".jpeg"}]
            if len(images) != 1:
                raise ValueError(f"源 GT 必须恰好对应一张原图: {path.name}")
            count += 1
            yield "dev" if source_split in {"validation", "dev"} else source_split, path.stem, images[0].read_bytes(), path.read_text(encoding="utf-8"), {
                "input_format": "cord_official_images_json", "source_file": path.relative_to(root).as_posix(), "official_split": source_split}
    if not count:
        raise ValueError("未找到 CORD train/validation/test parquet 或官方 split/images,json 目录")


def prepare_cord(input_dir, output_dir, explicit_split=None):
    rows, seen, mapping_counts, excluded, suppressed = [], set(), Counter(), [], []
    for split, identity, blob, raw, origin in cord_records(input_dir, explicit_split):
        safe_id = re.sub(r"[^a-zA-Z0-9_.-]", "-", identity)
        sid = f"cord-{split}-{safe_id}"
        if sid in seen:
            raise ValueError(f"重复 CORD ID: {sid}")
        seen.add(sid)
        truth, annotation, used = cord_target(raw)
        image, digest = store_image(blob, output_dir, sid)
        source_gt = store_gt(raw, output_dir, sid)
        if annotation["suppressed_fields"]:
            suppressed.append({"sample_id": sid, "split": split, "fields": annotation["suppressed_fields"],
                               "reason": "unannotated_item_attributes", "source_gt": source_gt})
        if not truth:
            excluded.append({"sample_id": sid, "split": split, "reason": "no_reliable_target_fields",
                             "suppressed_fields": annotation["suppressed_fields"],
                             "image": image, "source_gt": source_gt, "source_hash": digest})
            continue
        mapping_counts.update(used)
        rows.append({"sample_id": sid, "source_document_id": f"cord-original-{digest}",
                     "image": image, "split": split, "task": "receipt_extract", "schema_version": "receipt_v1",
                     "template_group": f"cord-official-{split}", "language": ["id"], "quality_tags": [],
                     "ground_truth": truth, "annotation": annotation, "source_hash": digest,
                     "requested_fields": list(truth), "provenance": {"source": "naver-clova-ix/cord-v2",
                     "source_gt": source_gt, "mapping_version": "cord_gt_parse_v2_safe_item_supervision", "observed_mappings": used,
                     "license": "CC-BY-4.0", "source_url": "https://huggingface.co/datasets/naver-clova-ix/cord-v2",
                     "source_revision": "7f0115a4b758a71d6473b8d085751692da2fef98",
                     "attribution": "Park et al., CORD (2019); NAVER CLOVA",
                     "license_verification": "official dataset README and CORD source README, 2026-11-20", "merchant_group_annotated": False,
                     "grouping": "official_split", "grouping_limitation": "无可靠商户标签，官方切分不是商户隔离；语言为数据集级元信息",
                     **origin}})
    write_manifests(rows, output_dir, "cord")
    report = audit_rows(rows, output_dir)
    report.update({"source": "cord", "raw_source_samples": len(rows) + len(excluded), "excluded": excluded,
                   "supervision_policy": "omit_whole_items_if_any_attribute_unannotated",
                   "suppressed_item_documents": len(suppressed), "suppressed_fields": suppressed,
                   "suppressed_item_split_counts": dict(Counter(s["split"] for s in suppressed)),
                   "supervised_item_documents": sum("items" in r["ground_truth"] for r in rows),
                   "excluded_policy": "original images and source GT retained, but no unsupported training targets invented",
                   "input_formats": sorted({r["provenance"]["input_format"] for r in rows}),
                   "observed_mapping_counts": dict(mapping_counts), "price_basis_verified": True,
                   "subtotal_group_aliases": ["subtotal", "sub_total"],
                   "nested_items": "menu.sub flattened; original hierarchy retained in source_gt",
                   "unverified_mappings": ["merchant_name", "address", "date", "phone", "time", "order_number", "payment_method"],
                   "review_limitation": "没有本轮人工复核，不声称双审；不启用加总方程"})
    write_json(Path(output_dir) / "manifests/cord-report.json", report)
    if report["errors"]:
        raise ValueError(f"CORD 转换后审计失败: {report['errors'][:5]}")
    return rows


def fatura_clean(value):
    if isinstance(value, list):
        return [fatura_clean(v) for v in value if v is not None]
    if isinstance(value, dict):
        content = {key: fatura_clean(v) for key, v in value.items()
                   if key not in {"bbox", "field_status", "status", "comment", "reviewed"}}
        if len(content) == 1:
            return next(iter(content.values()))
        return content
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return value


def fatura_target(raw):
    if not isinstance(raw, dict):
        raise ValueError("FATURA Original_Format 必须是字段对象")
    top_status = raw.get("field_status", {})
    if not isinstance(top_status, dict):
        raise ValueError("源 field_status 必须是对象")
    truth, statuses = {}, {}
    for key, value in raw.items():
        if key in {"OTHER", "field_status", "annotation", "metadata"}:
            continue
        declared = top_status.get(key)
        if isinstance(value, dict):
            declared = value.get("field_status", value.get("status", declared))
        cleaned = fatura_clean(value)
        if declared == "not_printed":
            statuses[key], truth[key] = "not_printed", None
        elif key == "TABLE" and cleaned == [[{}]]:
            statuses[key] = "unannotated"
        elif declared in {"unannotated", "present_unreadable"}:
            statuses[key] = declared
        elif cleaned is None or cleaned == "" or cleaned == {} or cleaned == []:
            statuses[key] = "unannotated"
        else:
            truth[key], statuses[key] = cleaned, "present_readable"
    # 只有源注释明确 not_printed 才监督 null，缺少键默认未标注。
    for key, status in top_status.items():
        if key not in statuses and key not in {"OTHER", "TABLE"} and status == "not_printed":
            statuses[key], truth[key] = status, None
    return truth, statuses


def template_split(groups, seed=42):
    ordered = sorted(set(groups))
    if len(ordered) < 3:
        raise ValueError("strict 模板组切分至少需要三个独立模板，不能伪装 OOD")
    random.Random(seed).shuffle(ordered)
    held_out = max(1, round(len(ordered) * 0.15))
    held_out = min(held_out, (len(ordered)-1)//2)
    mapping = {g: "train" for g in ordered}
    mapping.update({g: "dev" for g in ordered[-2*held_out:-held_out]})
    mapping.update({g: "test" for g in ordered[-held_out:]})
    return mapping


def prepare_fatura(input_dir, output_dir, split_mode="strict", seed=42):
    root = Path(input_dir)
    if (root / "invoices_dataset_final").is_dir():
        root = root / "invoices_dataset_final"
    records = []
    for split in SPLITS:
        path = root / f"strat1_{split}.csv"
        if not path.is_file():
            raise ValueError(f"缺少 FATURA split CSV: {path.name}")
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"img_path", "annot_path"} <= set(reader.fieldnames or []):
                raise ValueError("FATURA CSV 需要 img_path/annot_path 列")
            for row in reader:
                image_path = safe_path(root / "images", row["img_path"])
                gt_path = safe_path(root / "Annotations/Original_Format", row["annot_path"])
                template = re.search(r"Template(\d+)", image_path.name, flags=re.I)
                if not template:
                    raise ValueError(f"原图文件名缺少明确模板组: {image_path.name}")
                records.append({"legacy_split": split, "split": split, "image_path": image_path,
                                "gt_path": gt_path, "group": f"fatura-template-{int(template.group(1))}"})
    if not records:
        raise ValueError("FATURA 输入无记录")
    if split_mode == "strict":
        assigned = template_split([r["group"] for r in records], seed)
        for record in records:
            record["split"] = assigned[record["group"]]
    elif split_mode != "legacy":
        raise ValueError("未知 FATURA split_mode")
    # 字段集合和类型只读取最终 train 标签，冻结后才接触 dev/test GT。
    schema_fields, observed_types = set(), {}
    for record in records:
        if record["split"] == "train":
            truth, statuses = fatura_target(json.loads(record["gt_path"].read_text(encoding="utf-8")))
            schema_fields.update(truth)
            for field, value in truth.items():
                kind = "null" if value is None else ("object" if isinstance(value, dict) else "array" if isinstance(value, list) else "string")
                observed_types.setdefault(field, set()).add(kind)
    frozen = sorted(schema_fields)
    if not frozen:
        raise ValueError("训练集没有可审定 FATURA 字段")
    schema = {"schema_version": "invoice_fatura_v1", "source_split": "train", "fields": frozen,
              "observed_types": {f: sorted(types) for f, types in observed_types.items()},
              "type": "object", "additionalProperties": False,
              "properties": {f: {"type": sorted(types | {"null"})} for f, types in observed_types.items()}}
    write_json(Path(output_dir) / "manifests/invoice_fatura_v1.schema.json", schema)
    rows, unseen, seen = [], Counter(), set()
    for record in records:
        raw_text = record["gt_path"].read_text(encoding="utf-8")
        truth, statuses = fatura_target(json.loads(raw_text))
        unseen.update(set(truth) - schema_fields)
        truth = {key: value for key, value in truth.items() if key in schema_fields}
        if not truth:
            raise ValueError(f"{record['image_path'].name}: 训练冻结规范中没有可监督字段")
        for field in frozen:
            statuses.setdefault(field, "unannotated")
        identity = record["image_path"].relative_to(root / "images").as_posix()
        sid = "fatura-" + re.sub(r"[^a-zA-Z0-9_.-]", "-", Path(identity).stem)
        if sid in seen:
            raise ValueError(f"FATURA CSV 重复原件: {sid}")
        seen.add(sid)
        image, digest = store_image(record["image_path"].read_bytes(), output_dir, sid)
        source_gt = store_gt(raw_text, output_dir, sid)
        rows.append({"sample_id": sid, "source_document_id": sid, "image": image, "split": record["split"],
                     "task": "invoice_extract", "schema_version": "invoice_fatura_v1", "template_group": record["group"],
                     "language": ["en"], "quality_tags": ["source_synthetic_invoice"], "ground_truth": truth,
                     "requested_fields": list(truth), "source_hash": digest,
                     "annotation": {"field_status": statuses, "items_complete": False, "price_basis": "unknown",
                                    "subtotal_equation": None, "total_equation": None, "reviewed": True,
                                    "review_source": "fatura_public_source_gt", "human_reviewed_this_round": False,
                                    "review_limitation": "源公开标注；本轮未人工复核，bbox 保存在 source_gt"},
                     "provenance": {"source": "FATURA", "source_gt": source_gt, "source_file": identity,
                                    "mapping_version": "fatura_original_format_v1", "bbox_preserved": True,
                                    "schema_fields": frozen, "schema_source_split": "train", "legacy_split": record["legacy_split"],
                                    "split_mode": split_mode, "seed": seed, "grouping": "template" if split_mode == "strict" else "legacy",
                                    "grouping_limitation": "legacy 随机切分不保证模板隔离" if split_mode == "legacy" else "按明确 Template 文件名组隔离；不是商户隔离",
                                    "license_verification": "需按下载时上游许可证记录验证"}})
    write_manifests(rows, output_dir, f"fatura-{split_mode}")
    report = audit_rows(rows, output_dir)
    report.update({"source": "FATURA", "split_mode": split_mode, "schema_source_split": "train",
                   "frozen_schema_fields": frozen, "unseen_fields_excluded": dict(unseen),
                   "placeholder_table_policy": "[[{}]] 保留源 GT，但不请求、不监督",
                   "human_reviewed_this_round": False})
    write_json(Path(output_dir) / f"manifests/fatura-{split_mode}-report.json", report)
    if report["errors"]:
        raise ValueError(f"FATURA 转换后审计失败: {report['errors'][:5]}")
    return rows


def cord_main(argv=None):
    parser = argparse.ArgumentParser(description="转换 CORD parquet 或官方 images/json；复制真实原图与原始 GT")
    parser.add_argument("--input-dir", default="data/raw/cord")
    parser.add_argument("--output-dir", default="data/receipt")
    parser.add_argument("--split", choices=SPLITS, help="直接 images/json 输入的显式集合；不覆盖官方 parquet split")
    args = parser.parse_args(argv)
    prepare_cord(args.input_dir, args.output_dir, args.split)


def fatura_main(argv=None):
    parser = argparse.ArgumentParser(description="独立转换 FATURA，训练集冻结 schema；legacy/strict 模板切分")
    parser.add_argument("--input-dir", default="data/raw/fatura")
    parser.add_argument("--output-dir", default="data/receipt")
    parser.add_argument("--split-mode", choices=("legacy", "strict"), default="strict")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    prepare_fatura(args.input_dir, args.output_dir, args.split_mode, args.seed)
