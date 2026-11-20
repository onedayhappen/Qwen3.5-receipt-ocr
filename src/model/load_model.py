from __future__ import annotations

from typing import Any

from transformers import AutoConfig, AutoModelForImageTextToText, PretrainedConfig

from model.modeling_cls import (
    Qwen3_5ForSequenceClassification,
    Qwen3_5MoeForSequenceClassification,
)
from train.monkey_patch_forward import (
    replace_qwen3_5_moe_with_mixed_modality_forward,
    replace_qwen3_5_with_mixed_modality_forward,
)

# This fork only supports the natively-multimodal Qwen3.5 family:
#   qwen3_5      -> dense  (e.g. Qwen/Qwen3.5-4B)
#   qwen3_5_moe  -> MoE    (e.g. Qwen/Qwen3.5-*-A*-Instruct)
_GENERATION_MODEL_TYPES = {
    "qwen3_5",
    "qwen3_5_moe",
}

_PATCHERS = {
    "qwen3_5": (replace_qwen3_5_with_mixed_modality_forward,),
    "qwen3_5_moe": (replace_qwen3_5_moe_with_mixed_modality_forward,),
}

_SEQUENCE_CLASSIFICATION_MODEL_CLS = {
    "qwen3_5": Qwen3_5ForSequenceClassification,
    "qwen3_5_moe": Qwen3_5MoeForSequenceClassification,
}


def get_qwen_vl_generation_backbone(model):
    if not hasattr(model, "model"):
        raise TypeError(f"Unsupported generation model wrapper: {type(model)!r}")
    return model.model


def apply_qwen_vl_monkey_patches(model_type: str) -> str:
    try:
        patchers = _PATCHERS[model_type]
    except KeyError as exc:
        supported = ", ".join(sorted(_PATCHERS))
        raise ValueError(f"Unsupported Qwen3.5 model_type: {model_type}. Supported: {supported}") from exc

    for patcher in patchers:
        patcher()

    return model_type


def load_qwen_vl_generation_model(
    model_name_or_path: str, *, config: PretrainedConfig | None = None, **kwargs: Any
):
    if config is None:
        config = AutoConfig.from_pretrained(model_name_or_path)
    if config.model_type not in _GENERATION_MODEL_TYPES:
        supported = ", ".join(sorted(_GENERATION_MODEL_TYPES))
        raise ValueError(
            f"Unsupported Qwen3.5 generation model_type: {config.model_type}. Supported: {supported}"
        )

    apply_qwen_vl_monkey_patches(config.model_type)
    return AutoModelForImageTextToText.from_pretrained(
        model_name_or_path,
        config=config,
        **kwargs,
    )


def get_qwen_vl_sequence_classification_model_cls(model_type: str):
    try:
        return _SEQUENCE_CLASSIFICATION_MODEL_CLS[model_type]
    except KeyError as exc:
        supported = ", ".join(sorted(_SEQUENCE_CLASSIFICATION_MODEL_CLS))
        raise ValueError(
            f"Unsupported Qwen3.5 sequence classification model_type: {model_type}. Supported: {supported}"
        ) from exc


def load_qwen_vl_sequence_classification_model(
    model_name_or_path: str, *, config: PretrainedConfig | None = None, **kwargs: Any
):
    if config is None:
        config = AutoConfig.from_pretrained(model_name_or_path)
    apply_qwen_vl_monkey_patches(config.model_type)
    model_cls = get_qwen_vl_sequence_classification_model_cls(config.model_type)
    return model_cls.from_pretrained(
        model_name_or_path,
        config=config,
        **kwargs,
    )
