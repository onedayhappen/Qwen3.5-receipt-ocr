import ast
import sys
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def load_pure(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


contracts = load_pure("src/dataset/contracts.py", "receipt_dataset_contracts")
stages = load_pure("src/train/training_contracts.py", "receipt_training_contracts")
utils = load_pure("src/utils.py", "receipt_model_helpers")


def image_sample():
    return {
        "image": "images/a.jpg", "sample_id": "a", "source_id": "source-a",
        "schema_version": "receipt_v1", "annotation": {"reviewed": True},
        "ground_truth": {"total": "10.00"}, "requested_fields": ["total"],
        "conversations": [{"from": "human", "value": "<image>\n提取票据"},
                          {"from": "gpt", "value": '{"total":"10.00"}'}],
    }


def grpo_args(world=1, **overrides):
    values = dict(num_generations=4, world_size=world, per_device_train_batch_size=1,
        per_device_eval_batch_size=4, gradient_accumulation_steps=16, steps_per_generation=4,
        generation_batch_size=None, eval_strategy="epoch", freeze_vision_tower=True,
        freeze_merger=True, vision_lora=False, unfreeze_topk_vision=0, use_vllm=False)
    values.update(overrides)
    return SimpleNamespace(**values)


class DatasetContractsTests(unittest.TestCase):
    def test_strict_roles_and_pairs(self):
        contracts.validate_media_sample(image_sample())
        for turns in ([], [{"from": "human", "value": "x"}],
                      [{"from": "user", "value": "<image>"}, {"from": "gpt", "value": "x"}],
                      [{"from": "gpt", "value": "<image>"}, {"from": "human", "value": "x"}]):
            sample = image_sample()
            sample["conversations"] = turns
            with self.subTest(turns=turns), self.assertRaises(ValueError):
                contracts.validate_media_sample(sample)

    def test_media_count_and_ambiguity(self):
        for mutation in ({"image": ["a.jpg", "b.jpg"]}, {"image": []}, {"video": "a.mp4"}):
            sample = image_sample()
            sample.update(mutation)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                contracts.validate_media_sample(sample)
        sample = image_sample()
        sample["conversations"][1]["value"] = "<image>回答"
        with self.assertRaises(ValueError):
            contracts.validate_media_sample(sample)

    def test_grpo_single_round_images_only(self):
        contracts.validate_media_sample(image_sample(), single_turn_image=True)
        sample = image_sample()
        sample["conversations"] += [{"from": "human", "value": "继续"}, {"from": "gpt", "value": "回答"}]
        with self.assertRaises(ValueError):
            contracts.validate_media_sample(sample, single_turn_image=True)
        text = {"conversations": [{"from": "human", "value": "提取"}, {"from": "gpt", "value": "回答"}]}
        with self.assertRaises(ValueError):
            contracts.validate_media_sample(text, single_turn_image=True)

    def test_dpo_reasoning_is_bool_not_text(self):
        self.assertIs(contracts.has_reasoning("推理A"), True)
        self.assertIs(contracts.has_reasoning("推理B"), True)
        self.assertEqual(contracts.has_reasoning("推理A"), contracts.has_reasoning("推理B"))
        for value in (None, "", "  ", False, 1):
            self.assertIs(contracts.has_reasoning(value), False)
        preference = dict(image="a.jpg", prompt="<image>提取", chosen="好", rejected="差")
        contracts.validate_media_sample(preference, preference=True)
        preference["prompt"] = "无图片占位符"
        with self.assertRaises(ValueError):
            contracts.validate_media_sample(preference, preference=True)

    def test_token_budget_fail_fast(self):
        self.assertEqual(contracts.require_token_budget(4096, 4096), 4096)
        with self.assertRaisesRegex(ValueError, "禁止静默截断"):
            contracts.require_token_budget(4097, 4096, sample_id="a")
        with self.assertRaises(ValueError):
            contracts.require_token_budget(1, 0)

    def test_reward_metadata_is_separate(self):
        sample = image_sample()
        metadata = contracts.reward_metadata(sample)
        self.assertEqual(set(metadata), set(contracts.REWARD_METADATA_FIELDS))
        self.assertNotIn("conversations", metadata)
        self.assertEqual(metadata["source_id"], "source-a")
        tree = ast.parse((ROOT / "src/dataset/grpo_dataset.py").read_text(encoding="utf-8"))
        prompt_assign = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                             and any(isinstance(target, ast.Name) and target.id == "prompt" for target in node.targets))
        prompt_code = ast.unparse(prompt_assign)
        for key in contracts.REWARD_METADATA_FIELDS:
            self.assertNotIn(key, prompt_code)
        self.assertNotIn("turns[1]", prompt_code)


class StageContractsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "tests/receipt")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.model = SimpleNamespace(model_id="Qwen/Qwen3.5-4B", parent_stage=None, resume_checkpoint=None)
        self.data = SimpleNamespace(data_path="train.json")
        self.args = SimpleNamespace(output_dir=str(self.path / "output"), lora_weight_path="",
            lora_enable=True, vision_lora=False, use_dora=False, lora_rank=16, lora_alpha=32,
            freeze_vision_tower=True, freeze_merger=True, should_save=True)
        self.args.to_dict = lambda: {key: value for key, value in vars(self.args).items() if key != "to_dict"}

    def test_fresh_stage_and_explicit_resume(self):
        self.assertIsNone(stages.resolve_stage_contract(self.model, self.data, self.args, "sft"))
        stages.save_stage_config(self.model, self.data, self.args, "sft")
        checkpoint = Path(self.args.output_dir) / "checkpoint-1"
        checkpoint.mkdir()
        (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            stages.resolve_stage_contract(self.model, self.data, self.args, "sft")
        self.model.resume_checkpoint = str(checkpoint)
        self.assertEqual(stages.resolve_stage_contract(self.model, self.data, self.args, "sft"), str(checkpoint))
        self.args.use_dora = True
        with self.assertRaisesRegex(ValueError, "use_dora"):
            stages.resolve_stage_contract(self.model, self.data, self.args, "sft")

    def test_hf_explicit_resume_alias(self):
        stages.save_stage_config(self.model, self.data, self.args, "sft")
        checkpoint = Path(self.args.output_dir) / "checkpoint-1"
        checkpoint.mkdir()
        (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")
        self.args.resume_from_checkpoint = str(checkpoint)
        self.assertEqual(stages.resolve_stage_contract(self.model, self.data, self.args, "sft"), str(checkpoint))
        self.model.resume_checkpoint = "different-checkpoint"
        with self.assertRaises(ValueError):
            stages.resolve_stage_contract(self.model, self.data, self.args, "sft")

    def test_business_metrics_and_reward_cpu_integration(self):
        from receipt.metrics import evaluate_sample, aggregate_metrics
        sample = image_sample()
        sample["ground_truth"] = {"total": "10.00"}
        sample["annotation"] = {"reviewed": True, "field_status": {"total": "present_readable"}}
        prediction = SimpleNamespace(predictions=['{"total":"10.00"}', 'bad'], samples=[sample, sample],
                                     finish_reasons=["stop", "length"])
        metrics = stages.generation_metrics(prediction, evaluate_sample, aggregate_metrics)
        self.assertEqual(metrics["json_valid_rate"], 0.5)
        self.assertEqual(metrics["json_valid_rate_denominator"], 2)
        self.assertEqual(metrics["truncated_rate"], 0.5)
        self.assertEqual(metrics["fields_total_raw_em"], 0.5)
        reward = stages.load_reward_entrypoint("receipt.rewards.receipt_composite_reward")
        rewards = reward(completions=prediction.predictions, assistant=[sample["conversations"][1]["value"]] * 2,
                         **{key: [value, value] for key, value in contracts.reward_metadata(sample).items()})
        self.assertEqual(len(rewards), 2)
        self.assertTrue(all(isinstance(value, float) for value in rewards))
        self.assertGreater(rewards[0], rewards[1])

    def test_boolean_resume_and_adapter_model_id_rejected(self):
        self.model.resume_checkpoint = "True"
        with self.assertRaises(ValueError):
            stages.resolve_stage_contract(self.model, self.data, self.args, "sft")
        self.model.resume_checkpoint = None
        (self.path / "adapter_config.json").write_text("{}", encoding="utf-8")
        self.model.model_id = str(self.path)
        with self.assertRaises(ValueError):
            stages.resolve_stage_contract(self.model, self.data, self.args, "sft")

    def test_preference_requires_merged_sft_parent(self):
        with self.assertRaises(ValueError):
            stages.resolve_stage_contract(self.model, self.data, self.args, "dpo")
        base = self.path / "sft-merged"
        base.mkdir()
        (base / "model.safetensors").touch()
        self.model.model_id, self.model.parent_stage = str(base), "sft"
        with self.assertRaises(ValueError):
            stages.resolve_stage_contract(self.model, self.data, self.args, "dpo")
        (base / "run_stage_config.json").write_text(json.dumps({"stage": "sft", "merged": True}), encoding="utf-8")
        (base / "merge_manifest.json").write_text("{}", encoding="utf-8")
        self.assertIsNone(stages.resolve_stage_contract(self.model, self.data, self.args, "dpo"))
        self.assertIsNone(stages.resolve_stage_contract(self.model, self.data, self.args, "grpo"))
        self.model.parent_stage = "dpo"
        (base / "run_stage_config.json").write_text(json.dumps({"stage": "dpo", "merged": True}), encoding="utf-8")
        self.assertIsNone(stages.resolve_stage_contract(self.model, self.data, self.args, "grpo"))
        with self.assertRaises(ValueError):
            stages.resolve_stage_contract(self.model, self.data, self.args, "dpo")

    def test_grpo_group_divisibility(self):
        for world in (1, 2, 3, 4, 8):
            self.assertEqual(stages.validate_grpo_batch(grpo_args(world)), world * 4)
        for overrides in ({"steps_per_generation": 3}, {"per_device_eval_batch_size": 1},
                          {"freeze_merger": False}, {"vision_lora": True}, {"use_vllm": True}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                stages.validate_grpo_batch(grpo_args(**overrides))

    def test_lora_risk_module_exclusions(self):
        self.assertEqual(stages.lora_exclusions("['lm_head']"), ["lm_head", "in_proj_qkv", "visual"])
        self.assertNotIn("visual", stages.lora_exclusions([], vision_lora=True))
        with self.assertRaises(ValueError):
            stages.lora_exclusions("'visual'")
        with self.assertRaises(ValueError):
            stages.load_reward_entrypoint("bad_entrypoint")
        self.assertIs(stages.load_reward_entrypoint("json.loads"), json.loads)

    def test_generation_metrics_none_denominator_and_finish(self):
        prediction = SimpleNamespace(predictions=["ok", "bad"], samples=[{"score": True}, {"score": None}],
                                     finish_reasons=["stop", "length"])
        seen = []
        def evaluate(raw, sample, finish_reason=None):
            seen.append((raw, finish_reason))
            return {"score": sample["score"], "truncated": finish_reason == "length"}
        report = stages.generation_metrics(prediction, evaluate)
        self.assertEqual(report["score"], 1.0)
        self.assertEqual(report["score_denominator"], 1)
        self.assertEqual(report["truncated"], 0.5)
        self.assertEqual(seen, [("ok", "stop"), ("bad", "length")])


class MergeHelpersTests(unittest.TestCase):
    def test_mtp_keys_retained_after_checkpoint_conversion(self):
        base_keys = ["visual.weight", "model.weight", "lm_head.weight", "mtp.weight"]
        mapping = {r"^visual\.": "model.visual."}
        model_keys = {"model.visual.weight", "model.weight", "lm_head.weight"}
        self.assertEqual(utils.base_only_tensor_keys(base_keys, model_keys, mapping), ["mtp.weight"])
        with self.assertRaisesRegex(ValueError, "重复"):
            utils.base_only_tensor_keys(base_keys + ["mtp.weight"], model_keys, mapping)
        with self.assertRaisesRegex(ValueError, "键布局"):
            utils.base_only_tensor_keys(base_keys, {"unrelated"}, mapping)

    def test_non_lora_key_normalization(self):
        self.assertEqual(utils.normalize_non_lora_keys({"base_model.model.model.visual.weight": 1}), {"model.visual.weight": 1})
        with self.assertRaises(ValueError):
            utils.normalize_non_lora_keys({"base_model.x": 1, "x": 2})
        self.assertEqual(utils.checkpoint_model_key("visual.weight", {r"^visual\.": "model.visual."}), "model.visual.weight")

    def test_missing_non_lora_only_provably_pure_llm(self):
        config = {"target_modules": ["model.language_model.layers.0.q_proj"], "bias": "none"}
        self.assertTrue(utils.pure_llm_adapter(config))
        for mutation in ({"target_modules": ["q_proj"]}, {"target_modules": ["model.visual.q_proj"]},
                         {"bias": "all"}, {"modules_to_save": ["lm_head"]}):
            self.assertFalse(utils.pure_llm_adapter(dict(config, **mutation)))
        self.assertFalse(utils.pure_llm_adapter(config, {"training": {"freeze_vision_tower": False}}))
        frozen = {"lora_enable": True, "freeze_llm": True, "freeze_vision_tower": True, "freeze_merger": True}
        self.assertTrue(utils.pure_llm_adapter(config, {"training": frozen}))
        self.assertFalse(utils.pure_llm_adapter(config, {"training": dict(frozen, freeze_llm=False)}))

    def test_base_shards_fail_fast_and_no_path_escape(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests/receipt") as temp:
            base = Path(temp)
            with self.assertRaises(ValueError):
                utils.base_checkpoint_shards(base)
            (base / "model-00001.safetensors").touch()
            self.assertEqual(len(utils.base_checkpoint_shards(base)), 1)
            index = base / "model.safetensors.index.json"
            index.write_text(json.dumps({"weight_map": {"a": "missing.safetensors"}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                utils.base_checkpoint_shards(base)
            index.write_text(json.dumps({"weight_map": {"a": "../outside.safetensors"}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                utils.base_checkpoint_shards(base)
        with self.assertRaises(ValueError):
            utils.base_checkpoint_shards("Qwen/Qwen3.5-4B")


class ASTWiringTests(unittest.TestCase):
    def source(self, relative):
        return (ROOT / relative).read_text(encoding="utf-8")

    def test_sdpa_defaults_and_liger_disabled(self):
        tree = ast.parse(self.source("src/params.py"))
        for cls in (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name.endswith("Arguments")):
            for node in cls.body:
                if isinstance(node, ast.AnnAssign) and node.target.id == "disable_flash_attn2":
                    self.assertIs(ast.literal_eval(node.value.keywords[0].value), True)
                if isinstance(node, ast.AnnAssign) and node.target.id in ("use_liger_loss", "use_liger_kernel"):
                    self.assertIs(ast.literal_eval(node.value), False)
        self.assertNotIn("dpo_loss:", self.source("src/params.py"))

    def test_entrypoints_dora_resume_and_reward(self):
        for stage in ("sft", "dpo", "grpo"):
            source = self.source(f"src/train/train_{stage}.py")
            tree = ast.parse(source)
            config = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                          and isinstance(node.func, ast.Name) and node.func.id == "LoraConfig")
            self.assertEqual(ast.unparse(next(key.value for key in config.keywords if key.arg == "use_dora")), "training_args.use_dora")
            self.assertIn("resume_from_checkpoint=resume_checkpoint", source)
            self.assertNotIn("resume_from_checkpoint=True", source)
            self.assertIn("log_trainable_parameters", source)
            self.assertIn("load_resume_non_lora", source)
            self.assertIn("StageConfigCallback", source)
        self.assertIn("load_reward_entrypoint(training_args.reward_entrypoint)", self.source("src/train/train_grpo.py"))
        self.assertNotIn("load_reward_funcs", self.source("src/train/train_grpo.py"))

    def test_sft_preflight_eval_folders_and_left_padding(self):
        dataset = self.source("src/dataset/sft_dataset.py")
        self.assertIn('self.lengths = [len(self[index]["input_ids"])', dataset)
        self.assertIn("require_token_budget(len(input_ids)", dataset)
        for name in ("sft", "dpo", "grpo"):
            self.assertIn("eval_args.image_folder = data_args.eval_image_folder or data_args.image_folder", self.source(f"src/dataset/{name}_dataset.py"))
        trainer = self.source("src/trainer/sft_trainer.py")
        self.assertIn("padded_prompts[i, -prompt_len:] = prompt", trainer)
        self.assertIn("padded_mm_token_type_ids[i, -prompt_len:]", trainer)
        self.assertIn('prompt_width = gen_inputs["input_ids"].shape[1]', trainer)
        self.assertIn("generated_ids[i][prompt_width:]", trainer)

    def test_grpo_no_visual_token_truncation_or_reprocessing(self):
        source = self.source("src/trainer/grpo_trainer.py")
        self.assertNotIn('"truncation": True', source)
        self.assertIn('"truncation": False', source)
        self.assertIn("require_token_budget(length, self.max_prompt_length", source)
        self.assertIn("forward_kwargs = self._rollout_forward_kwargs", source)
        self.assertIn("*REWARD_METADATA_FIELDS", source)
        self.assertIn('extra_fields = {"finish_reason":', source)
        tree = ast.parse(source)
        score = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_generate_and_score_completions")
        self.assertNotIn("self.processing_class(**processor_kwargs)", ast.unparse(score))

    def test_loader_preserves_kwargs_fp32_mtp(self):
        source = self.source("src/utils.py")
        self.assertIn('kwargs.setdefault("attn_implementation", "flash_attention_2" if use_flash_attn else "sdpa")', source)
        self.assertIn('kwargs["dtype"] = torch.bfloat16', source)
        self.assertIn('"dtype" not in kwargs and "torch_dtype" not in kwargs', source)
        self.assertIn("safe_merge=True", source)
        merge = self.source("src/merge_lora_weights.py")
        self.assertIn("state_dict.update(base_only)", merge)
        self.assertNotIn("v.to(target_dtype)", merge)
        self.assertIn("preserve_base_fp32=True", merge)

    def test_json_configs_and_script_defaults(self):
        for stage in ("sft", "dpo", "grpo", "vision_lora"):
            config = json.loads(self.source(f"configs/receipt/{stage}.json"))
            self.assertEqual(config["lora_rank"], 16)
            self.assertEqual(config["lora_alpha"], 32)
            self.assertEqual(config["per_device_train_batch_size"], 1)
            self.assertEqual(config["gradient_accumulation_steps"], 16)
            self.assertTrue(config["disable_flash_attn2"])
            self.assertTrue(config["freeze_vision_tower"])
            self.assertTrue(config["freeze_merger"])
            self.assertTrue(config["bf16"])
            for key in ("data_path", "eval_path", "image_folder", "output_dir"):
                self.assertFalse(Path(config[key]).is_absolute())
            if stage in ("dpo", "grpo"):
                self.assertEqual(config["model_id"], "models/sft-merged")
                self.assertEqual(config["parent_stage"], "sft")
        for file in (ROOT / "scripts").glob("finetune*.sh"):
            source = file.read_text(encoding="utf-8")
            self.assertIn("NUM_DEVICES", source)
            self.assertNotIn("NUM_DEVICES=8", source)
            self.assertNotIn("/path/to/", source)


if __name__ == "__main__":
    unittest.main()
