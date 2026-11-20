# Third-Party Notices

This project is an independent derivative work. Portions of the source code are
derived from the upstream projects listed below. Their copyright notices and
license terms are reproduced here as required by their licenses.

---

## Qwen-VL-Series-Finetune

- **Project:** Qwen-VL-Series-Finetune
- **Copyright:** Copyright (c) 2024 Yuxuan Zhang, Xingyu Liu
- **Source:** https://github.com/2U1/Qwen-VL-Series-Finetune
- **License:** Apache License, Version 2.0 (full text in [`LICENSE`](LICENSE))
- **Upstream snapshot:** `fff1851`

The following files contain code derived from this project and remain under the
Apache License, Version 2.0:

```
src/model/load_model.py
src/model/modeling_cls.py
src/model/__init__.py
src/train/monkey_patch_forward.py
src/dataset/data_utils.py
src/dataset/cls_dataset.py
src/dataset/dpo_dataset.py
src/dataset/grpo_dataset.py
src/dataset/sft_dataset.py
src/trainer/grpo_trainer.py
```

### Modifications in this project

1. Removed support for the `qwen2_vl`, `qwen2_5_vl`, `qwen3_vl` and
   `qwen3_vl_moe` model families. Only the natively multimodal Qwen3.5 family
   remains: `model_type` `qwen3_5` (dense) and `qwen3_5_moe` (MoE).
2. Removed the Qwen2.x default-system-message branch and the Qwen2.x
   `second_per_grid_ts` video path from the dataset layer. Qwen3.5 uses
   `video_metadata` for 3D RoPE instead.
3. Changed default image pixel budgets from 28-aligned to 32-aligned values
   (one Qwen3.5 vision token covers 32x32 pixels), in `src/params.py` and
   `scripts/*.sh`.
4. Changed `scripts/*.sh` to pass `--disable_flash_attn2 True`, which Qwen3.5
   requires.
5. Changed the default `model_id` from `Qwen/Qwen2-VL-7B-Instruct` to
   `Qwen/Qwen3.5-4B`.
6. Deleted `src/train/monkey_patch_vision.py`, which only patched Qwen2.5-VL.

---

## Scope of this project's MIT license

Everything **not** listed under a third-party notice above — including the
receipt business layer in `src/receipt/`, `src/serve/`, `configs/`, `scripts/`,
`tests/` and `tools/` — is original work released under the MIT license; see
[`LICENSE`](LICENSE).

The MIT license of this repository does **not** relicense the
Qwen-VL-Series-Finetune derived files listed above. Those files remain governed
by the Apache License, Version 2.0, and this notice must be retained in
redistributions and derivative works.

---

## Data licenses are separate

Code licensing does not cover data. Dataset licenses, model licenses and any
private receipt photographs are recorded separately in
[`configs/data_sources.json`](configs/data_sources.json) and in the project data
card. Original receipt images are not distributed in this repository.

---

## Historical source snapshot

An earlier internal snapshot of this project's business design was adapted from
`qwen-receipt-ocr` (MIT, Copyright (c) 2026 Dr.AlexLiu, snapshot `cbcabfe`).
That work is by the same copyright holder as this repository, so it is covered
by the root [`LICENSE`](LICENSE) and is not restated here.
