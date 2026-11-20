import ast
import importlib
import json
from pathlib import Path


def lora_exclusions(value, *, vision_lora=False):
    names = ast.literal_eval(value) if isinstance(value, str) else list(value or [])
    if not isinstance(names, list) or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("lora_namespan_exclude 必须是非空字符串列表。")
    names += ["in_proj_qkv"]
    if not vision_lora:
        names += ["visual"]
    return list(dict.fromkeys(names))


def load_reward_entrypoint(entrypoint):
    module, separator, name = entrypoint.rpartition(".")
    if not separator:
        raise ValueError("reward_entrypoint 必须是 module.function。")
    reward = getattr(importlib.import_module(module), name)
    if not callable(reward):
        raise TypeError(f"{entrypoint} 不是可调用奖励函数。")
    return reward


def validate_grpo_batch(training_args):
    group = training_args.num_generations
    world = training_args.world_size
    batch = training_args.per_device_train_batch_size
    steps = training_args.steps_per_generation or training_args.gradient_accumulation_steps
    generation_batch = getattr(training_args, "generation_batch_size", None) or world * batch * steps
    if group < 2 or generation_batch % group:
        raise ValueError(f"GRPO generation_batch_size={generation_batch} 必须被 num_generations={group} 整除且G>=2。")
    if generation_batch % (world * batch):
        raise ValueError("GRPO生成批次必须被进程数×每进程batch整除。")
    if str(training_args.eval_strategy) not in ("no", "IntervalStrategy.NO"):
        eval_batch = world * training_args.per_device_eval_batch_size
        eval_group = getattr(training_args, "num_generations_eval", None) or group
        if eval_batch % eval_group:
            raise ValueError(f"GRPO eval总batch={eval_batch} 必须被G={eval_group}整除；单卡可设eval_batch_size=4。")
    if not training_args.freeze_vision_tower or not training_args.freeze_merger or training_args.vision_lora or training_args.unfreeze_topk_vision:
        raise ValueError("GRPO主线必须冻结视觉塔/merger，禁止视觉LoRA或视觉解冻。")
    if training_args.use_vllm or getattr(training_args, "rollout_func", None):
        raise ValueError("当前GRPO视觉缓存主线仅支持原生generate，不支持外部rollout/vLLM。")
    return generation_batch


def resolve_stage_contract(model_args, data_args, training_args, stage):
    model_path = Path(model_args.model_id)
    if (model_path / "adapter_config.json").exists():
        raise ValueError("model_id 必须指向完整模型/已合并父checkpoint，不能指向adapter。")
    if training_args.lora_weight_path:
        raise ValueError("每个阶段应在合并父模型上创建新adapter；续训请显式设置 resume_checkpoint。")
    if stage in ("dpo", "grpo"):
        allowed_parents = ("sft",) if stage == "dpo" else ("sft", "dpo")
        if model_args.parent_stage not in allowed_parents:
            raise ValueError(f"{stage}必须显式设置有效parent_stage: {allowed_parents}；默认配置从已合并SFT分叉。")
        if not model_path.is_dir() or not list(model_path.glob("model*.safetensors")):
            raise ValueError("DPO/GRPO model_id 必须是本地已合并父模型，请先训练并合并。")
        parent_config = model_path / "run_stage_config.json"
        if not parent_config.is_file():
            raise ValueError("父模型缺少有效SFT阶段配置，不能确认已合并父checkpoint来源。")
        parent = json.loads(parent_config.read_text(encoding="utf-8"))
        if parent.get("stage") != model_args.parent_stage or parent.get("merged") is not True or not (model_path / "merge_manifest.json").is_file():
            raise ValueError("父模型阶段不匹配，或不是带合并清单的merged产物。")
    output = Path(training_args.output_dir)
    resume = model_args.resume_checkpoint
    hf_resume = getattr(training_args, "resume_from_checkpoint", None)
    if resume and hf_resume and str(resume) != str(hf_resume):
        raise ValueError("resume_checkpoint 与 resume_from_checkpoint 不能指定不同路径。")
    resume = resume or hf_resume
    if resume:
        if str(resume).lower() in ("true", "false"):
            raise ValueError("resume_checkpoint 必须是明确checkpoint路径，不允许自动搜索。")
        checkpoint = Path(resume)
        if not checkpoint.is_dir() or not (checkpoint / "trainer_state.json").is_file():
            raise ValueError("resume_checkpoint 缺少 trainer_state.json。")
        metadata_path = checkpoint / "run_stage_config.json"
        if not metadata_path.is_file():
            metadata_path = checkpoint.parent / "run_stage_config.json"
        if not metadata_path.is_file():
            raise ValueError("续训checkpoint缺少阶段配置。")
        saved = json.loads(metadata_path.read_text(encoding="utf-8"))
        current_model = str(model_path.resolve()) if model_path.is_dir() else model_args.model_id
        for key, expected in (("stage", stage), ("model_id", current_model), ("parent_stage", model_args.parent_stage)):
            if saved.get(key) != expected:
                raise ValueError(f"续训阶段契约不匹配: {key}。")
        saved_training = saved.get("training", {})
        for key in ("lora_enable", "vision_lora", "use_dora", "lora_rank", "lora_alpha", "freeze_vision_tower", "freeze_merger"):
            if saved_training.get(key) != getattr(training_args, key):
                raise ValueError(f"续训训练配置不匹配: {key}。")
        return str(checkpoint)
    if output.exists() and (list(output.glob("checkpoint-*")) or (output / "adapter_config.json").exists() or (output / "run_stage_config.json").exists()):
        raise ValueError("输出目录已包含checkpoint或阶段产物；请显式续训或使用新目录。")
    return None


def save_stage_config(model_args, data_args, training_args, stage):
    if not training_args.should_save:
        return
    output = Path(training_args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_path = Path(model_args.model_id)
    config = {
        "stage": stage,
        "parent_stage": model_args.parent_stage,
        "model_id": str(model_path.resolve()) if model_path.is_dir() else model_args.model_id,
        "resume_checkpoint": model_args.resume_checkpoint or getattr(training_args, "resume_from_checkpoint", None),
        "data": vars(data_args),
        "training": training_args.to_dict(),
    }
    path = output / "run_stage_config.json"
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def generation_metrics(prediction, evaluate_sample, aggregate_metrics=None):
    results = [evaluate_sample(raw, sample, finish_reason=finish) for raw, sample, finish in zip(
        prediction.predictions, prediction.samples, prediction.finish_reasons, strict=True
    )]
    if aggregate_metrics is None:
        numeric_keys = {key for result in results for key, value in result.items() if isinstance(value, (int, float))}
        metrics = {}
        for key in numeric_keys:
            values = [result[key] for result in results if isinstance(result.get(key), (int, float))]
            metrics[key] = sum(values) / len(values)
            metrics[key + "_denominator"] = len(values)
        return metrics
    report = aggregate_metrics(results)
    metrics = {}

    def flatten(prefix, value):
        if isinstance(value, dict) and "value" in value:
            if isinstance(value["value"], (int, float)):
                metrics[prefix] = value["value"]
            if "denominator" in value:
                metrics[prefix + "_denominator"] = value["denominator"]
        elif isinstance(value, dict):
            for key, child in value.items():
                if key != "groups":
                    flatten(prefix + "_" + key if prefix else key, child)

    flatten("", report)
    return metrics


def log_trainable_parameters(model, printer=print):
    trainable = [(name, getattr(param, "ds_numel", param.numel())) for name, param in model.named_parameters() if param.requires_grad]
    if not trainable:
        raise ValueError("没有可训练参数。")
    total = sum(getattr(param, "ds_numel", param.numel()) for param in model.parameters())
    count = sum(size for _, size in trainable)
    printer(f"实际训练参数: {count}/{total} ({100 * count / total:.4f}%)")
    printer("实际训练模块: " + ", ".join(name for name, _ in trainable))
    return count
