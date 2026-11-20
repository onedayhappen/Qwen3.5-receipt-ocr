import copy
import os
import json
from torch.utils.data import Dataset

from constants import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from .contracts import reward_metadata, validate_media_sample
from .data_utils import get_image_info, get_qwen_multimodal_settings, llava_to_openai


class GRPODataset(Dataset):
    def __init__(self, data_path, processor, data_args, model_id, padding=True):
        super().__init__()
        if isinstance(data_path, str):
            with open(data_path, encoding="utf-8") as handle:
                samples = json.load(handle)
        else:
            samples = data_path
        self.list_data_dict = [validate_media_sample(sample, single_turn_image=True) for sample in samples]
        from receipt.data import initialize_source_schemas
        initialize_source_schemas(self.list_data_dict, data_args.image_folder or ".")
        self.processor = processor
        self.data_args = data_args
        self.model_type, self.image_patch_size, _ = get_qwen_multimodal_settings(model_id)
        if data_args.enable_reasoning:
            raise ValueError("GRPO票据主线只支持非思考模式，防止参考回答改变生成prompt。")
        self.processor.image_processor.do_resize = False

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, index):
        sample = self.list_data_dict[index]
        image_files = sample["image"]
        if isinstance(image_files, str):
            image_files = [image_files]
        images = []
        for path in image_files:
            if not os.path.exists(path) and not path.startswith(("http://", "https://")):
                path = os.path.join(self.data_args.image_folder or "", path)
            images.append(get_image_info(
                path, self.data_args.image_min_pixels, self.data_args.image_max_pixels,
                self.data_args.image_resized_width, self.data_args.image_resized_height,
                self.image_patch_size,
            ))
        turns = llava_to_openai(sample["conversations"], preserve_whitespace=self.data_args.preserve_media_token_whitespace)
        # 奖励元数据与参考回答绝不参与prompt构造。
        prompt = (
            f"{DEFAULT_IM_START_TOKEN}user\n{turns[0]['content']}{DEFAULT_IM_END_TOKEN}\n"
            f"{DEFAULT_IM_START_TOKEN}assistant\n<think>\n\n</think>\n\n"
        )
        return dict(prompt=prompt, assistant=turns[1]["content"], images=images, **reward_metadata(sample))


def make_grpo_data_module(model_id, processor, data_args):
    train_dataset = GRPODataset(data_args.data_path, processor, data_args, model_id)
    eval_dataset = None
    if data_args.eval_path:
        eval_args = copy.copy(data_args)
        eval_args.image_folder = data_args.eval_image_folder or data_args.image_folder
        eval_dataset = GRPODataset(data_args.eval_path, processor, eval_args, model_id)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)
