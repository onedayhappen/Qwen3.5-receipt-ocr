"""Repo-local, configuration-driven training launcher. No torch import for --dry-run."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

PATH_KEYS = {"data_path", "eval_path", "image_folder", "eval_image_folder", "output_dir", "deepspeed", "resume_checkpoint", "resume_from_checkpoint"}


def build_command(config: dict, root: Path, stage: str, python=None, world_size=1) -> tuple[list[str], dict]:
    if stage not in {"sft", "dpo", "grpo", "cls"}:
        raise ValueError("Unsupported training stage")
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    options = dict(config)
    for key in PATH_KEYS:
        if key in options and options[key] is not None and not Path(str(options[key])).is_absolute():
            options[key] = str((root / str(options[key])).resolve())
    model = options.get("model_id", "")
    if model and not model.startswith("Qwen/") and not Path(model).is_absolute():
        options["model_id"] = str((root / model).resolve())
    if stage in {"dpo", "grpo"}:
        parent = Path(options["model_id"])
        if parent.exists() and (parent / "adapter_config.json").exists():
            raise ValueError("Post-training model_id must be the merged parent, not an adapter directory")
    if options.get("disable_flash_attn2") is not True:
        raise ValueError("Qwen3.5 receipt configs must explicitly use SDPA")
    for data_key in ("data_path", "eval_path"):
        path = options.get(data_key)
        if path and Path(path).exists():
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(data, list) or not data:
                raise ValueError(f"{data_key} must contain a non-empty JSON array")
            if any(row.get("split") == "test" for row in data):
                raise ValueError("Test samples are forbidden in training/dev data")
    if stage == "grpo":
        generations = int(options.get("num_generations", 4))
        batch = int(options.get("per_device_train_batch_size", 1))
        accumulation = int(options.get("gradient_accumulation_steps", 1))
        steps = int(options.get("steps_per_generation") or accumulation)
        rollout = int(options.get("generation_batch_size") or world_size * batch * steps)
        if generations < 2 or rollout % generations or rollout % (world_size * batch):
            raise ValueError("GRPO effective rollout batch must be divisible by num_generations >= 2 and world_size*batch")
        if options.get("eval_strategy", "no") != "no":
            eval_group = int(options.get("num_generations_eval") or generations)
            if world_size * int(options.get("per_device_eval_batch_size", 1)) % eval_group:
                raise ValueError("GRPO eval batch must be divisible by num_generations_eval")
    executable = python or sys.executable
    module = f"train.train_{stage}"
    command = [executable, "-m", module] if world_size == 1 else [executable, "-m", "torch.distributed.run", "--nproc_per_node", str(world_size), "--module", module]
    for key, value in options.items():
        if value is None:
            continue
        if isinstance(value, bool):
            value = "True" if value else "False"
        elif isinstance(value, (dict, list)):
            value = json.dumps(value)
        command += ["--" + key, str(value)]
    return command, options


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["sft", "dpo", "grpo", "cls"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--model", help="Override model_id with a fixed base or merged parent")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.model:
        config["model_id"] = args.model
    command, effective = build_command(config, root, args.stage, world_size=args.world_size)
    run = {"stage": args.stage, "command": command, "config": effective, "world_size": args.world_size,
           "global_batch_size": args.world_size * int(effective.get("per_device_train_batch_size", 1)) * int(effective.get("gradient_accumulation_steps", 1))}
    print(json.dumps(run, indent=2, ensure_ascii=False))
    if args.dry_run:
        return
    for key in ("data_path", "eval_path", "image_folder"):
        if effective.get(key) and not Path(effective[key]).exists():
            raise FileNotFoundError(f"Missing {key}: {effective[key]}")
    if args.stage in ("dpo", "grpo") and not Path(effective["model_id"]).is_dir():
        raise FileNotFoundError("First train and merge the SFT parent, then pass --model models/sft-merged")
    output = Path(effective["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "launcher_config.json").write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    subprocess.run(command, cwd=root, env=environment, check=True)

if __name__ == "__main__":
    main()
