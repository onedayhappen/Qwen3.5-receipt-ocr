"""票据事实清单、受限监督导出与独立 CPU 数据工具。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import re
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path, PurePosixPath, PureWindowsPath

FIELD_STATUSES = {"present_readable", "not_printed", "present_unreadable", "unannotated"}
REQUIRED_KEYS = {"sample_id", "source_document_id", "image", "split", "task", "schema_version",
                 "template_group", "language", "quality_tags", "ground_truth", "annotation",
                 "provenance", "source_hash", "requested_fields"}
SPLITS = ("train", "dev", "test")


def json_text(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"JSONL 第 {number} 行不是对象")
                rows.append(row)
    return rows


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def write_manifests(rows, root, name="manifest"):
    root = Path(root)
    directory = root / "manifests"
    directory.mkdir(parents=True, exist_ok=True)
    for suffix, selected in [("", rows)] + [(f"-{split}", [r for r in rows if r["split"] == split]) for split in SPLITS]:
        (directory / f"{name}{suffix}.jsonl").write_text(
            "".join(json_text(row) + "\n" for row in selected), encoding="utf-8")


def safe_path(root, relative):
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("媒体路径必须为非空 POSIX 相对路径")
    if PureWindowsPath(relative).drive or PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts:
        raise ValueError(f"路径越出数据根目录: {relative}")
    root = Path(root).resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"路径越出数据根目录: {relative}")
    return target


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def store_image(blob, root, sample_id):
    from PIL import Image
    with Image.open(io.BytesIO(blob)) as image:
        image.verify()
        extension = {"PNG": "png", "JPEG": "jpeg"}.get(image.format)
    if extension is None:
        raise ValueError("只接收 PNG/JPEG 原图，不静默重新编码")
    relative = f"images/{sample_id}.{extension}"
    path = safe_path(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != blob:
        raise ValueError(f"拒绝覆盖同名不同原图: {relative}")
    path.write_bytes(blob)
    return relative, sha256_bytes(blob)


def store_gt(raw, root, sample_id):
    relative = f"source_gt/{sample_id}.json"
    path = safe_path(root, relative)
    payload = (raw if isinstance(raw, str) else json_text(raw)).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != payload:
        raise ValueError(f"拒绝覆盖同名不同源标注: {relative}")
    path.write_bytes(payload)
    return relative


def register_source_schema(root, schema_version):
    if schema_version == "invoice_fatura_v1":
        from receipt.schema import register_invoice_fatura_schema
        path = safe_path(root, "manifests/invoice_fatura_v1.schema.json")
        frozen = json.loads(path.read_text(encoding="utf-8"))
        if frozen.get("source_split") != "train":
            raise ValueError("FATURA schema 必须来自训练集")
        # 审计元信息不注册到业务 schema，因而不会渲染到 prompt。
        business_schema = {key: frozen[key] for key in ("type", "properties", "additionalProperties")}
        register_invoice_fatura_schema(business_schema)


def initialize_source_schemas(samples, root):
    """Load training-frozen contracts once per process, never infer from eval/test GT."""
    for version in sorted({row.get("schema_version", "receipt_v1") for row in samples}):
        register_source_schema(root, version)


def validate_sample(sample, root, check_schema=True):
    from PIL import Image
    errors = []
    missing = REQUIRED_KEYS - sample.keys()
    if missing:
        return [f"清单缺少键: {sorted(missing)}"]
    if sample["split"] not in SPLITS:
        errors.append("未知 split")
    for key in ("sample_id", "source_document_id", "template_group"):
        if not isinstance(sample[key], str) or not sample[key]:
            errors.append(f"{key} 必须是非空字符串")
    fields = sample["requested_fields"]
    truth = sample["ground_truth"]
    annotation = sample["annotation"]
    if not isinstance(annotation, dict) or not isinstance(sample["provenance"], dict):
        return errors + ["annotation/provenance 必须为对象"]
    if not isinstance(sample["language"], list) or not isinstance(sample["quality_tags"], list):
        errors.append("language/quality_tags 必须为列表")
    if annotation.get("price_basis") not in {"unit", "line", "unknown"}:
        errors.append("price_basis 必须明确为 unit/line/unknown")
    if not isinstance(annotation.get("items_complete"), bool):
        errors.append("items_complete 必须为显式布尔值")
    if annotation.get("price_basis") == "unknown" and annotation.get("subtotal_equation"):
        errors.append("未知价格口径不得启用商品加总方程")
    if not isinstance(sample["source_hash"], str) or not re.fullmatch(r"[0-9a-f]{64}", sample["source_hash"]):
        errors.append("source_hash 必须是 SHA-256")
    if not isinstance(fields, list) or not fields or any(not isinstance(f, str) for f in fields):
        return errors + ["requested_fields 必须是非空字符串列表"]
    if len(set(fields)) != len(fields) or not isinstance(truth, dict) or set(fields) != set(truth):
        return errors + ["target 必须恰好包含 requested_fields，禁止仅以 metadata mask"]
    statuses = annotation.get("field_status", {})
    if not isinstance(statuses, dict):
        return errors + ["field_status 必须是对象"]
    if any(not isinstance(s, str) or s not in FIELD_STATUSES for s in statuses.values()):
        errors.append("未知 field_status")
    for field in fields:
        status = statuses.get(field)
        if status not in {"present_readable", "not_printed"}:
            errors.append(f"请求了未经审定字段: {field}")
        if status == "not_printed" and truth[field] is not None and not (field == "items" and truth[field] == []):
            errors.append(f"not_printed 标签和值冲突: {field}")
        if status == "present_readable" and (truth[field] is None or truth[field] == "" or truth[field] == []):
            errors.append(f"可读标签不应为空: {field}")
    if sample["schema_version"] == "receipt_v1":
        from receipt.schema import RECEIPT_FIELDS
        if set(statuses) != set(RECEIPT_FIELDS):
            errors.append("receipt field_status 必须覆盖完整规范")
        if not set(fields) <= set(RECEIPT_FIELDS):
            errors.append("未知 receipt 字段")
    if sample["schema_version"] == "invoice_fatura_v1":
        frozen = sample["provenance"].get("schema_fields", [])
        if not frozen or not set(fields) <= set(frozen):
            errors.append("invoice 字段未包含在训练冻结规范中")
        if any(f == "OTHER" or (f == "TABLE" and truth[f] == [[{}]]) for f in fields):
            errors.append("未知 TABLE/OTHER 不可监督")
    if "items" in truth and isinstance(truth["items"], list):
        item_status = annotation.get("item_field_status")
        if item_status is None:
            errors.append("可监督商品必须显式提供item_field_status，不能推断未知属性缺失")
        else:
            if not isinstance(item_status, list) or len(item_status) != len(truth["items"]):
                errors.append("商品属性标注数量不匹配")
            else:
                for item, status_map in zip(truth["items"], item_status):
                    if not isinstance(item, dict) or not isinstance(status_map, dict):
                        errors.append("商品和属性状态必须是对象")
                        continue
                    for key in ("name", "quantity", "price"):
                        if status_map.get(key) not in FIELD_STATUSES:
                            errors.append(f"未知商品属性状态: {key}")
                        state = status_map.get(key)
                        if state not in {"present_readable", "not_printed"}:
                            errors.append(f"请求了未经审定商品属性: {key}; 不能仅用metadata屏蔽整串SFT")
                        elif state == "not_printed" and item.get(key) is not None:
                            errors.append(f"not_printed 商品属性和值冲突: {key}")
                        elif state == "present_readable" and item.get(key) in (None, ""):
                            errors.append(f"可读商品属性不应为空: {key}")
    try:
        image_path = safe_path(root, sample["image"])
        blob = image_path.read_bytes()
        if sha256_bytes(blob) != sample["source_hash"]:
            errors.append("原图 SHA-256 不匹配")
        with Image.open(io.BytesIO(blob)) as image:
            image.verify()
        source_gt = sample["provenance"].get("source_gt")
        if source_gt:
            safe_path(root, source_gt).read_bytes()
    except (ValueError, OSError) as exc:
        errors.append(str(exc))
    if annotation.get("reviewed") is not True:
        errors.append("参考标签未经审定")
    if not annotation.get("review_source"):
        errors.append("必须明示 review_source，不能暗示本轮人工双审")
    if check_schema:
        from receipt.schema import validate_output
        try:
            register_source_schema(root, sample["schema_version"])
            errors.extend(validate_output(truth, sample["schema_version"], fields))
        except (ValueError, OSError) as exc:
            errors.append(str(exc))
    return errors


def select_split(rows, split, evaluation=False):
    if split == "test" and not evaluation:
        raise ValueError("test 只能显式导出 evaluation，禁止用于训练/偏好")
    if any(row.get("split") != split for row in rows):
        raise ValueError("输入必须是单一指定 split 清单，拒绝混合集或隐式过滤 test")
    if not rows:
        raise ValueError("清单为空")
    return rows


def render_user(sample):
    from receipt.prompts import build_prompt
    instruction = build_prompt(schema_version=sample["schema_version"], fields=sample["requested_fields"])
    if "<image>" in instruction:
        raise ValueError("共享 build_prompt 必须返回不含 image 占位符的指令")
    return "<image>\n" + instruction


def export_rows(rows, root, split="train", kind="sft"):
    select_split(rows, split, evaluation=kind == "evaluation")
    output = []
    for sample in rows:
        errors = validate_sample(sample, root)
        if errors:
            raise ValueError(f"{sample['sample_id']}: {errors}")
        prompt = render_user(sample)
        row = {"image": sample["image"], "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": json_text(sample["ground_truth"])}]}
        if prompt.count("<image>") != 1:
            raise ValueError("单图必须恰好有一个 image 占位符")
        if kind in {"grpo", "evaluation"}:
            row.update({key: copy.deepcopy(sample[key]) for key in
                        ("sample_id", "split", "schema_version", "ground_truth", "annotation", "requested_fields")})
        output.append(row)
    return output


def audit_rows(rows, root, near_duplicates=False, near_threshold=4):
    errors, warnings = [], []
    buckets = {key: defaultdict(list) for key in ("sample_id", "source_document_id", "source_hash", "parent")}
    coverage, groups, split_counts = Counter(), defaultdict(set), Counter()
    for sample in rows:
        sid = sample.get("sample_id", "unknown")
        errors.extend(f"{sid}: {error}" for error in validate_sample(sample, root))
        split = sample.get("split")
        split_counts[split] += 1
        for key in ("sample_id", "source_document_id", "source_hash"):
            if sample.get(key):
                buckets[key][sample[key]].append((sid, split))
        parent = sample.get("augmentation_parent_id") or sample.get("provenance", {}).get("augmentation_parent_id")
        for identity in {sid, sample.get("source_document_id"), parent} - {None}:
            buckets["parent"][identity].append((sid, split))
        for field, status in sample.get("annotation", {}).get("field_status", {}).items():
            coverage[(sample.get("schema_version"), split, field, status)] += 1
        groups[(sample.get("schema_version"), sample.get("template_group"))].add(split)
        if sample.get("conversations"):
            turns = sample["conversations"]
            if [turn.get("from") for turn in turns] != ["human", "gpt"]:
                errors.append(f"{sid}: 不合法训练角色")
            if len(turns) == 2 and turns[1].get("value") != json_text(sample.get("ground_truth")):
                errors.append(f"{sid}: assistant label 与 GT 不一致")
        if sample.get("messages"):
            errors.append(f"{sid}: 清单不能包含 SWIFT/messages 角色格式")
    for key, mapping in buckets.items():
        for value, members in mapping.items():
            if len({split for _, split in members}) > 1:
                errors.append(f"跨集泄漏 {key}={value}: {members}")
            elif key == "sample_id" and len(members) > 1:
                errors.append(f"重复 sample_id: {value}")
    overlaps = [{"schema_version": schema, "group": group, "splits": sorted(splits)}
                for (schema, group), splits in groups.items() if len(splits) > 1]
    if overlaps:
        warnings.append("存在共享模板组；不能声称三集合完全商户/模板隔离")
    limitations = sorted({s.get("provenance", {}).get("grouping_limitation", "") for s in rows} - {""})
    near = []
    if near_duplicates:
        from PIL import Image
        hashes = []
        for sample in rows:
            try:
                with Image.open(safe_path(root, sample["image"])) as image:
                    values = list(image.convert("L").resize((8, 8)).tobytes())
                    mean = sum(values) / len(values)
                    fingerprint = sum((1 << index) for index, value in enumerate(values) if value >= mean)
                    hashes.append((sample, fingerprint))
            except (ValueError, OSError):
                continue
        for index, (a, ah) in enumerate(hashes):
            for b, bh in hashes[index + 1:]:
                distance = (ah ^ bh).bit_count()
                if a["split"] != b["split"] and distance <= near_threshold:
                    near.append({"a": a["sample_id"], "b": b["sample_id"], "distance": distance})
        if near:
            warnings.append("发现 average-hash 跨集近重复候选，仅警告、不自动删图")
    return {"samples": len(rows), "split_counts": dict(split_counts), "errors": errors,
            "synthetic_samples": sum(s.get("provenance", {}).get("synthetic", False) for s in rows),
            "warnings": warnings, "group_overlaps": overlaps, "grouping_limitations": limitations,
            "near_duplicates": near, "gt_coverage": [dict(schema_version=a, split=b, field=c, status=d, count=n)
                                                      for (a, b, c, d), n in sorted(coverage.items())]}


def synthetic_responses(sample):
    truth = sample["ground_truth"]
    if isinstance(truth.get("items"), list) and truth["items"]:
        missing = copy.deepcopy(truth)
        missing["items"] = missing["items"][:-1]
        yield "missing_items", json_text(missing)
        for index, item in enumerate(truth["items"]):
            if item.get("price") is not None:
                wrong = copy.deepcopy(truth)
                try:
                    wrong["items"][index]["price"] = str(Decimal(item["price"]) + Decimal("1.00"))
                except Exception:
                    wrong["items"][index]["price"] = "SYNTHETIC_WRONG_PRICE"
                yield "wrong_price", json_text(wrong)
                break
    if truth:
        missing = copy.deepcopy(truth)
        missing.pop(next(iter(missing)))
        yield "missing_field", json_text(missing)


def build_preferences(rows, root, candidates=(), split="train", synthetic_negatives=False,
                      max_pairs=2, min_margin=0.05):
    from receipt.rewards import score_completion
    select_split(rows, split)
    if max_pairs < 1 or not math.isfinite(min_margin) or min_margin <= 0:
        raise ValueError("max_pairs 必须为正，min_margin 必须为有限正数")
    initialize_source_schemas(rows, root)
    by_id = {sample["sample_id"]: sample for sample in rows}
    if len(by_id) != len(rows):
        raise ValueError("重复 sample_id")
    pools = defaultdict(list)
    prompt_versions = {}
    rejected_counts = Counter()
    for candidate in candidates:
        required = {"sample_id", "response", "model_version", "prompt_version", "reviewed", "prompt"}
        if not required <= candidate.keys():
            raise ValueError("候选缺少 sample_id/response/model_version/prompt_version/reviewed/prompt")
        sid = candidate["sample_id"]
        if sid not in by_id or candidate.get("split", split) != split:
            raise ValueError(f"候选不属于指定 {split} 集: {sid}")
        if not isinstance(candidate["response"], str):
            raise ValueError("候选 response 必须为字符串")
        if candidate["reviewed"] is not True:
            rejected_counts["unreviewed"] += 1
            continue
        if not candidate["model_version"] or not candidate["prompt_version"]:
            raise ValueError("候选版本信息不得为空")
        actual_prompt = render_user(by_id[sid])
        if candidate["prompt"] != actual_prompt:
            raise ValueError("候选实际 prompt 与当前业务合同不一致")
        if candidate["prompt_version"].startswith("sha256:") and candidate["prompt_version"] != "sha256:" + sha256_bytes(actual_prompt.encode("utf-8")):
            raise ValueError("候选 prompt_version hash 不匹配")
        if candidate.get("schema_version", by_id[sid]["schema_version"]) != by_id[sid]["schema_version"]:
            raise ValueError("候选 schema 与参考不一致")
        if candidate.get("requested_fields", by_id[sid]["requested_fields"]) != by_id[sid]["requested_fields"]:
            raise ValueError("候选 requested_fields 与参考不一致")
        if sid in prompt_versions and prompt_versions[sid] != candidate["prompt_version"]:
            raise ValueError("同图候选不能混合不同 prompt_version")
        prompt_versions[sid] = candidate["prompt_version"]
        pools[sid].append(("model_candidate", candidate["response"], candidate))
    result, metadata = [], []
    for sample in rows:
        errors = validate_sample(sample, root)
        if errors:
            raise ValueError(f"{sample['sample_id']}: {errors}")
        chosen = json_text(sample["ground_truth"])
        chosen_result = score_completion(chosen, sample)
        chosen_reward = float(chosen_result["reward"])
        if not math.isfinite(chosen_reward) or chosen_result.get("eligible") is False:
            raise ValueError(f"参考标签不适用于偏好评分: {sample['sample_id']}")
        options = list(pools[sample["sample_id"]])
        if synthetic_negatives:
            options += [(kind, raw, {"provenance": "synthetic_negatives", "synthetic": True})
                        for kind, raw in synthetic_responses(sample)]
        seen = {chosen}
        accepted = []
        for kind, raw, origin in options:
            if "prompt" in origin and origin["prompt"] != render_user(sample):
                raise ValueError("候选实际 prompt 与导出 prompt 不一致")
            try:
                canonical = json_text(json.loads(raw))
            except (ValueError, TypeError):
                canonical = raw
            if canonical in seen:
                rejected_counts["identical"] += 1
                continue
            seen.add(canonical)
            reward = float(score_completion(raw, sample)["reward"])
            margin = chosen_reward - reward
            if not math.isfinite(margin) or margin < min_margin:
                rejected_counts["small_margin"] += 1
                continue
            accepted.append((margin, kind, raw, origin, reward))
        accepted.sort(key=lambda value: value[0], reverse=True)
        for margin, kind, raw, origin, reward in accepted[:max_pairs]:
            result.append({"image": sample["image"], "prompt": render_user(sample),
                           "chosen": chosen, "rejected": raw})
            metadata.append({"sample_id": sample["sample_id"], "split": split,
                             "synthetic_source": sample["provenance"].get("synthetic", False),
                             "purpose": "cpu_regression" if sample["provenance"].get("synthetic") else ("evaluation" if split == "dev" else "training"),
                             "error_type": kind, "chosen_reward": chosen_reward,
                             "rejected_reward": reward, "margin": margin,
                             "chosen_review_source": sample["annotation"]["review_source"],
                             "candidate": origin})
    return result, {"pairs": len(result), "split": split, "purpose": "evaluation" if split == "dev" else "training",
                    "synthetic_pairs": sum(m["candidate"].get("synthetic", False) for m in metadata),
                    "synthetic_source_pairs": sum(m["synthetic_source"] for m in metadata),
                    "limitation": "synthetic_negatives 是工程错误注入，不等于真实模型偏好；source-reviewed 不等于本轮人工双审",
                    "filtered": dict(rejected_counts), "pair_metadata": metadata}


def make_fixtures(root, count=6):
    from PIL import Image, ImageDraw
    from receipt.schema import RECEIPT_FIELDS
    if not 3 <= count <= 12:
        raise ValueError("CPU 合成夹具只允许 3 至 12 张，不是业务训练集")
    rows = []
    for index in range(count):
        sid = f"synthetic-receipt-{index:02d}"
        split = "train" if index < count - 2 else ("dev" if index == count - 2 else "test")
        price = str(Decimal("2.50") + index)
        truth = {"merchant_name": f"SYNTHETIC STORE {index}", "items": [
            {"name": f"SYNTHETIC ITEM {index}", "quantity": "1", "price": price}], "total": price}
        image = Image.new("RGB", (360, 220 + index * 7), "white")
        draw = ImageDraw.Draw(image)
        draw.text((16, 16), f"SYNTHETIC CPU FIXTURE - NOT REAL\nSTORE {index}\nITEM {index}  x1  {price}\nTOTAL {price}", fill="black", spacing=16)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        relative, digest = store_image(buffer.getvalue(), root, sid)
        gt = store_gt({"synthetic": True, "ground_truth": truth}, root, sid)
        sample = {"sample_id": sid, "source_document_id": sid, "image": relative, "split": split,
                  "task": "receipt_extract", "schema_version": "receipt_v1", "template_group": f"synthetic-group-{index}",
                  "language": ["en"], "quality_tags": ["synthetic", "cpu_fixture"], "ground_truth": truth,
                  "annotation": {"field_status": {f: "present_readable" if f in truth else "unannotated" for f in RECEIPT_FIELDS},
                                 "item_field_status": [{"name": "present_readable", "quantity": "present_readable", "price": "present_readable"}],
                                 "items_complete": True, "price_basis": "line", "subtotal_equation": None,
                                 "total_equation": None, "reviewed": True, "review_source": "synthetic_generator",
                                 "human_reviewed_this_round": False},
                  "provenance": {"source": "synthetic_cpu_fixture", "synthetic": True, "source_gt": gt},
                  "source_hash": digest, "requested_fields": list(truth)}
        rows.append(sample)
    write_manifests(rows, root)
    pairs, report = build_preferences([r for r in rows if r["split"] == "train"], root, synthetic_negatives=True)
    report.update({"synthetic": True, "purpose": "cpu_regression", "not_business_training_data": True})
    write_json(Path(root) / "exports/dpo_train_synthetic.json", pairs)
    write_json(Path(root) / "exports/dpo_train_synthetic_metadata.json", report)
    return rows


def export_main(argv=None):
    parser = argparse.ArgumentParser(description="导出受限监督 SFT/GRPO 或封存评估数据")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", default="data/receipt")
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--format", choices=("sft", "grpo", "evaluation"), default="sft")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    output = Path(args.output_dir) if args.output_dir else Path(args.data_root) / "exports"
    samples = read_jsonl(args.manifest)
    values = export_rows(samples, args.data_root, args.split, args.format)
    write_json(output / f"{args.format}_{args.split}.json", values)
    lengths = sorted(len(row["conversations"][1]["value"]) for row in values)
    write_json(output / f"{args.format}_{args.split}_report.json", {
        "samples": len(values), "synthetic": all(s["provenance"].get("synthetic", False) for s in samples),
        "target_characters_max": max(lengths),
        "target_characters_p95": lengths[min(len(lengths)-1, int(len(lengths)*0.95))],
        "token_budget_verified": False, "limitation": "字符统计不等于 tokenizer/视觉 token 预算；训练前仍需处理器预检"})


def audit_main(argv=None):
    parser = argparse.ArgumentParser(description="检查原件、SHA、增强父件与字段监督跨集泄漏")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", default="data/receipt")
    parser.add_argument("--near-duplicates", action="store_true")
    parser.add_argument("--near-threshold", type=int, default=4)
    parser.add_argument("--report", default="reports/data_pipeline_audit.json")
    args = parser.parse_args(argv)
    report = audit_rows(read_jsonl(args.manifest), args.data_root, args.near_duplicates, args.near_threshold)
    write_json(args.report, report)
    if report["errors"]:
        raise SystemExit(1)


def preferences_main(argv=None):
    parser = argparse.ArgumentParser(description="用审定候选构造 train 偏好，dev 仅用于显式评估")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", default="data/receipt")
    parser.add_argument("--candidates")
    parser.add_argument("--synthetic-negatives", action="store_true")
    parser.add_argument("--split", choices=("train", "dev"), default="train")
    parser.add_argument("--max-pairs-per-image", type=int, default=2)
    parser.add_argument("--min-margin", type=float, default=0.05)
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    if not args.candidates and not args.synthetic_negatives:
        parser.error("必须提供审定候选或显式 --synthetic-negatives")
    values, report = build_preferences(read_jsonl(args.manifest), args.data_root,
                                      read_jsonl(args.candidates) if args.candidates else [], args.split,
                                      args.synthetic_negatives, args.max_pairs_per_image, args.min_margin)
    output = Path(args.output_dir) if args.output_dir else Path(args.data_root) / "exports"
    name = f"dpo_{args.split}" + ("_synthetic" if args.synthetic_negatives else "")
    write_json(output / f"{name}.json", values)
    write_json(output / f"{name}_metadata.json", report)


def fixtures_main(argv=None):
    parser = argparse.ArgumentParser(description="生成最多 12 张显式 synthetic CPU 票据夹具")
    parser.add_argument("--output-dir", default="data/fixtures")
    parser.add_argument("--count", type=int, default=6)
    args = parser.parse_args(argv)
    make_fixtures(args.output_dir, args.count)


def main(argv=None):
    import sys
    from receipt.sources import cord_main, fatura_main
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {"prepare_cord": cord_main, "prepare_fatura": fatura_main,
                "export_training_data": export_main, "build_preferences": preferences_main,
                "audit_splits": audit_main, "make_fixtures": fixtures_main}
    if argv and argv[0] in commands:
        commands[argv[0]](argv[1:])
        return
    parser = argparse.ArgumentParser(description="独立安装包票据数据管线 CLI")
    parser.add_argument("command", choices=tuple(commands))
    parser.parse_args(argv)


if __name__ == "__main__":
    main()
