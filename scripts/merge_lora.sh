#!/bin/bash
set -euo pipefail
# BASE_MODEL必须指向已下载的本地完整safetensors父模型，不能用远程model_id。
export PYTHONPATH="src:${PYTHONPATH:-}"
python src/merge_lora_weights.py \
    --model-path "${ADAPTER_PATH:-models/sft-adapter}" \
    --model-base "${BASE_MODEL:-models/Qwen3.5-4B}" \
    --save-model-path "${MERGED_MODEL:-models/sft-merged}" \
    --safe-serialization "$@"
