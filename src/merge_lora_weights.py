import argparse
import json
from pathlib import Path

from utils import (
    base_checkpoint_shards, base_only_tensor_keys, checkpoint_model_key, get_model_name_from_path,
    is_lora_model, load_pretrained_model,
)


def _base_checkpoint_shards(base_path):
    return base_checkpoint_shards(base_path)


def _to_model_key(key, conversion_mapping):
    return checkpoint_model_key(key, conversion_mapping)


def collect_base_only_tensors(model_base, model):
    from safetensors import safe_open
    shards = base_checkpoint_shards(model_base)
    model_keys = set(model.state_dict())
    mapping = getattr(type(model), "_checkpoint_conversion_mapping", None) or {}
    shard_keys = {}
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            shard_keys[shard] = list(handle.keys())
    # 先检查键布局，再分配base-only张量，避免错误布局触发巨大额外内存分配。
    extra_keys = set(base_only_tensor_keys(
        (key for keys in shard_keys.values() for key in keys), model_keys, mapping,
    ))
    extra = {}
    for shard, keys in shard_keys.items():
        selected = [key for key in keys if key in extra_keys]
        if selected:
            with safe_open(shard, framework="pt", device="cpu") as handle:
                extra.update({key: handle.get_tensor(key) for key in selected})
    return extra


def merge_lora(args):
    base_checkpoint_shards(args.model_base)
    adapter = Path(args.model_path)
    output = Path(args.save_model_path)
    if not is_lora_model(adapter):
        raise ValueError("model-path必须是完整本地safetensors adapter。")
    if output.resolve() in (adapter.resolve(), Path(args.model_base).resolve()):
        raise ValueError("不能覆盖adapter或base目录。")
    if output.exists() and any(output.iterdir()):
        raise ValueError("合并输出目录必须为空，禁止覆盖已有模型。")
    stage_path = adapter / "run_stage_config.json"
    if not stage_path.is_file():
        stage_path = adapter.parent / "run_stage_config.json"
    stage_config = json.loads(stage_path.read_text(encoding="utf-8")) if stage_path.is_file() else None
    if stage_config and Path(stage_config["model_id"]).is_dir() and Path(stage_config["model_id"]).resolve() != Path(args.model_base).resolve():
        raise ValueError("adapter记录的父模型与model-base不一致。")
    processor, model = load_pretrained_model(
        model_path=args.model_path, model_base=args.model_base,
        model_name=get_model_name_from_path(args.model_path), device_map="cpu",
        allow_missing_non_lora=getattr(args, "allow_missing_non_lora", False), preserve_base_fp32=True,
    )
    state_dict = model.state_dict()
    base_only = collect_base_only_tensors(args.model_base, model)
    if set(state_dict) & set(base_only):
        raise ValueError("base-only张量键与已合并模型冲突。")
    # MTP等base-only张量保持原始dtype，不统一向bf16/fp16降精度。
    state_dict.update(base_only)
    model.save_pretrained(output, state_dict=state_dict, safe_serialization=True)
    processor.save_pretrained(output)
    manifest = {
        "model_base": str(Path(args.model_base).resolve()),
        "adapter": str(adapter.resolve()),
        "preserved_base_only_keys": sorted(base_only),
        **getattr(model, "_receipt_merge_metadata", {}),
    }
    (output / "merge_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if stage_config:
        stage_config["merged"] = True
        stage_config["merge_manifest"] = "merge_manifest.json"
        (output / "run_stage_config.json").write_text(json.dumps(stage_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"合并完成，保留{len(base_only)}个base-only张量；fp32保留情况已记录。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, required=True)
    parser.add_argument("--save-model-path", type=str, required=True)
    parser.add_argument("--safe-serialization", action="store_true", default=True)
    parser.add_argument("--allow-missing-non-lora", action="store_true")
    merge_lora(parser.parse_args())
