"""Load only the selected training dataset, not classification or optional branches."""
from importlib import import_module

_MODULES = {
    "make_dpo_data_module": ".dpo_dataset",
    "make_supervised_data_module": ".sft_dataset",
    "make_grpo_data_module": ".grpo_dataset",
    "make_classification_data_module": ".cls_dataset",
}
__all__ = list(_MODULES)


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(name)
    return getattr(import_module(_MODULES[name], __name__), name)