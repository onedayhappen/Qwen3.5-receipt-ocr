from pathlib import Path
import json
import re
import importlib
import inspect
from typing import Callable, List


def disable_torch_init():
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def base_checkpoint_shards(model_base):
    base = Path(model_base)
    if not base.is_dir():
        raise ValueError("合并要求本地完整base safetensors目录，不能使用远程model_id。")
    index = base / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("base safetensors索引缺少有效weight_map。")
        shards = sorted({base / name for name in weight_map.values()})
    else:
        shards = sorted(base.glob("model*.safetensors"))
    if not shards or any(not shard.is_file() or shard.suffix != ".safetensors" or base.resolve() not in shard.resolve().parents for shard in shards):
        raise ValueError("base safetensors缺失、索引不完整或包含越界路径。")
    return shards


def checkpoint_model_key(key, conversion_mapping):
    for pattern, replacement in conversion_mapping.items():
        renamed = re.sub(pattern, replacement, key)
        if renamed != key:
            return renamed
    return key


def base_only_tensor_keys(base_keys, model_keys, conversion_mapping):
    base_keys = list(base_keys)
    if len(base_keys) != len(set(base_keys)):
        raise ValueError("base分片包含重复张量键。")
    extra = [key for key in base_keys if checkpoint_model_key(key, conversion_mapping) not in model_keys]
    if extra and len(extra) > len(base_keys) // 4:
        raise ValueError(f"{len(extra)}/{len(base_keys)} base张量未匹配，可能存在模型键布局错误；禁止静默丢弃。")
    return extra


def normalize_non_lora_keys(weights):
    normalized = {(key[11:] if key.startswith("base_model.") else key): value for key, value in weights.items()}
    if any(key.startswith("model.model.") for key in normalized):
        normalized = {(key[6:] if key.startswith("model.") else key): value for key, value in normalized.items()}
    if len(normalized) != len(weights):
        raise ValueError("non_lora权重键重命名发生冲突。")
    return normalized


def pure_llm_adapter(adapter_config, stage_config=None):
    targets = adapter_config.get("target_modules")
    if not isinstance(targets, list) or not targets:
        return False
    if adapter_config.get("bias", "none") != "none" or adapter_config.get("modules_to_save"):
        return False
    if any(not isinstance(name, str) or "language_model." not in name or "visual" in name or "merger" in name for name in targets):
        return False
    if stage_config is not None:
        training = stage_config.get("training", {})
        if not training.get("lora_enable") or not training.get("freeze_llm") or training.get("vision_lora") or not training.get("freeze_vision_tower") or not training.get("freeze_merger") or training.get("unfreeze_topk_llm", 0) or training.get("unfreeze_topk_vision", 0):
            return False
    return True


def restore_base_fp32_tensors(model_base, model):
    import torch
    from safetensors import safe_open
    tensors = dict(model.named_parameters())
    tensors.update(dict(model.named_buffers()))
    mapping = getattr(type(model), "_checkpoint_conversion_mapping", None) or {}
    restored = []
    for shard in base_checkpoint_shards(model_base):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                model_key = checkpoint_model_key(key, mapping)
                if model_key not in tensors or handle.get_slice(key).get_dtype() != "F32":
                    continue
                value = handle.get_tensor(key)
                target = tensors[model_key]
                if target.shape != value.shape:
                    raise ValueError(f"fp32 base张量形状不一致: {key}。")
                target.data = value.to(device=target.device, dtype=torch.float32)
                restored.append(model_key)
    return restored


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False,
                          device_map="auto", device="cuda", use_flash_attn=False,
                          allow_missing_non_lora=False, preserve_base_fp32=False, **kwargs):
    import torch
    from peft import PeftModel
    from transformers import BitsAndBytesConfig, AutoProcessor, AutoConfig
    from model.load_model import load_qwen_vl_generation_model
    kwargs = dict(kwargs)
    kwargs.setdefault("device_map", device_map if device == "cuda" else {"": device})
    if "dtype" not in kwargs and "torch_dtype" not in kwargs:
        kwargs["dtype"] = torch.bfloat16
    kwargs.setdefault("attn_implementation", "flash_attention_2" if use_flash_attn else "sdpa")
    kwargs.setdefault("low_cpu_mem_usage", True)
    if load_8bit and load_4bit:
        raise ValueError("不能同时启用4bit和8bit。")
    if load_8bit or load_4bit:
        kwargs.setdefault("quantization_config", BitsAndBytesConfig(
            load_in_8bit=load_8bit, load_in_4bit=load_4bit,
            bnb_4bit_compute_dtype=kwargs.get("dtype", kwargs.get("torch_dtype")),
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        ))
    adapter_dir = Path(model_path)
    if (adapter_dir / "adapter_config.json").exists() and not is_lora_model(adapter_dir):
        raise ValueError("adapter目录不完整，缺少adapter_model.safetensors。")
    if is_lora_model(adapter_dir):
        if model_base is None:
            raise ValueError("加载adapter必须明确提供合并父model_base。")
        adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
        # config采用完整父模型，避免adapter配置覆盖父模型加载契约。
        kwargs.setdefault("config", AutoConfig.from_pretrained(model_base))
        model = load_qwen_vl_generation_model(model_base, **kwargs)
        processor_source = adapter_dir if (adapter_dir / "preprocessor_config.json").is_file() else model_base
        processor = AutoProcessor.from_pretrained(processor_source)
        restored = restore_base_fp32_tensors(model_base, model) if preserve_base_fp32 else []
        non_lora_path = adapter_dir / "non_lora_state_dict.bin"
        stage_path = adapter_dir / "run_stage_config.json"
        if not stage_path.is_file():
            stage_path = adapter_dir.parent / "run_stage_config.json"
        stage_config = json.loads(stage_path.read_text(encoding="utf-8")) if stage_path.is_file() else None
        missing_allowed = False
        if non_lora_path.is_file():
            weights = normalize_non_lora_keys(torch.load(non_lora_path, map_location="cpu", weights_only=True))
            result = model.load_state_dict(weights, strict=False)
            if result.unexpected_keys:
                raise RuntimeError(f"non_lora包含未匹配权重: {result.unexpected_keys}；禁止丢弃训练参数。")
        elif allow_missing_non_lora and pure_llm_adapter(adapter_config, stage_config):
            missing_allowed = True
            print("已显式允许纯LLM adapter缺少non_lora；视觉/merger沿用父模型，并记录合并清单。")
        else:
            raise FileNotFoundError("缺少non_lora_state_dict.bin；只有显式允许且证明是纯LLM adapter才能跳过。")
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload(safe_merge=True)
        model._receipt_merge_metadata = {"missing_non_lora_allowed": missing_allowed, "restored_fp32_keys": restored}
    else:
        kwargs.setdefault("config", AutoConfig.from_pretrained(model_path))
        processor = AutoProcessor.from_pretrained(model_path)
        model = load_qwen_vl_generation_model(model_path, **kwargs)
    return processor, model


def is_lora_model(model_path: str | Path) -> bool:
    model_dir = Path(model_path)
    return (model_dir / "adapter_config.json").is_file() and (model_dir / "adapter_model.safetensors").is_file()


def get_model_name_from_path(model_path):
    model_paths = model_path.strip("/").split("/")
    if model_paths[-1].startswith("checkpoint-"):
        return model_paths[-2] + "_" + model_paths[-1]
    return model_paths[-1]


def load_reward_funcs(module_path: str = "train.reward_funcs", *, name_pred=lambda n: n.endswith("_reward"),
                      obj_pred=lambda o: callable(o), keep_order=True) -> List[Callable]:
    # 保留原通用接口，但票据GRPO入口仅使用显式reward_entrypoint。
    mod = importlib.import_module(module_path)
    funcs = [(name, obj) for name, obj in inspect.getmembers(mod, predicate=obj_pred) if name_pred(name)]
    if keep_order:
        funcs.sort(key=lambda pair: inspect.getsourcelines(pair[1])[1])
    return [obj for _, obj in funcs]
