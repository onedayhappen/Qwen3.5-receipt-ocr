import copy
import os
from typing import Dict
import torch
import transformers
import ujson as json
from torch.utils.data import Dataset

from params import DataArguments
from constants import (
    IGNORE_INDEX,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
)

from .contracts import has_reasoning, require_token_budget, validate_media_sample

from .data_utils import (
    chat_template_uses_reasoning_prefill,
    format_assistant_response,
    get_image_info,
    get_mm_token_type_ids,
    get_qwen_multimodal_settings,
    get_video_info,
    llava_to_openai,
    model_supports_optional_reasoning,
    pad_sequence,
)

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id,
        padding=True,
        max_length=32768,
        generation_eval=False,
    ):
        super(SupervisedDataset, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r", encoding="utf-8"))
        else:
            list_data_dict = data_path

        self.model_id = model_id
        self.processor = processor
        self.list_data_dict = [validate_media_sample(sample) for sample in list_data_dict]
        from receipt.data import initialize_source_schemas
        initialize_source_schemas(self.list_data_dict, data_args.image_folder or ".")
        self.max_length = max_length
        self.generation_eval = generation_eval
        if generation_eval and any(len(sample["conversations"]) != 2 for sample in self.list_data_dict):
            raise ValueError("生成评估只支持单轮业务输入，多轮请使用外部批评估。")
        self.data_args = data_args
        self.padding = padding
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.video_min_pixel = data_args.video_min_pixels
        self.video_max_pixel = data_args.video_max_pixels
        self.image_resized_w = data_args.image_resized_width
        self.image_resized_h = data_args.image_resized_height
        self.video_resized_w = data_args.video_resized_width
        self.video_resized_h = data_args.video_resized_height
        self.fps = data_args.fps
        self.nframes = data_args.nframes

        self.model_type, self.image_patch_size, self.return_video_metadata = get_qwen_multimodal_settings(
            self.model_id
        )
        self.reasoning_supported = chat_template_uses_reasoning_prefill(self.processor, self.model_type)
        self.use_reasoning_prefill = self.data_args.enable_reasoning and self.reasoning_supported
        self.optional_reasoning_supported = model_supports_optional_reasoning(self.model_type)
        if self.data_args.enable_reasoning and not self.reasoning_supported:
            raise ValueError(
                f"`enable_reasoning` is only supported for Qwen3-VL Thinking or Qwen3.5 models with official reasoning chat templates. "
                f"Current model_type={self.model_type!r} does not qualify."
            )

        # 完整调用processor预检真实视觉token展开长度，不使用字符数估算或静默裁剪。
        self.lengths = [len(self[index]["input_ids"]) for index in range(len(self.list_data_dict))]

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]

        is_video = False

        processor = self.processor
        if "image" in sources:
            videos = None
            grid_key = "image_grid_thw"
            pixel_key = "pixel_values"

            image_files = sources["image"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]

            images = []

            for image_file in image_files:
                if not os.path.exists(image_file):
                    if not image_file.startswith("http"):
                        image_file = os.path.join(image_folder or "", image_file)
                image_input = get_image_info(
                        image_file, 
                        self.image_min_pixel, 
                        self.image_max_pixel, 
                        self.image_resized_w, 
                        self.image_resized_h, 
                        self.image_patch_size
                    )
                images.append(image_input)

        elif "video" in sources:
            is_video = True
            images=None
            grid_key = "video_grid_thw"
            pixel_key = "pixel_values_videos"

            video_files = sources["video"]
            video_folder = self.data_args.image_folder

            if isinstance(video_files, str):
                video_files = [video_files]

            videos = []
            for video_file in video_files:
                if not os.path.exists(video_file):
                    if not video_file.startswith("http"):
                        video_file = os.path.join(video_folder, video_file)
                video_input, video_kwargs = get_video_info(
                    video_file, 
                    self.video_min_pixel, 
                    self.video_max_pixel, 
                    self.video_resized_w, 
                    self.video_resized_h, 
                    self.data_args.fps,
                    self.image_patch_size,
                    return_video_metadata=self.return_video_metadata
                )
                videos.append(video_input)
        else:
            grid_key = None
            pixel_key = None
            images=None
            videos=None

        sources = copy.deepcopy(llava_to_openai(
            sources['conversations'], is_video=is_video,
            preserve_whitespace=getattr(self.data_args, 'preserve_media_token_whitespace', False),
        ))

        all_input_ids = []
        all_labels = []
        all_mm_token_type_ids = []
        all_pixel_values = []
        all_image_grid_thw = []

        image_curr_count = 0
        video_curr_count = 0
        
        # Qwen3.5 does not use a default system message.
        for _, j in enumerate(range(0, len(sources), 2)):
            user_input = sources[j]
            gpt_response = sources[j + 1]
            reasoning_present = has_reasoning(gpt_response.get("reasoning"))
            if self.data_args.enable_reasoning and self.reasoning_supported and not self.optional_reasoning_supported and not reasoning_present:
                raise ValueError(
                    "When `--enable_reasoning True` is used with Qwen3-VL Thinking, every assistant turn must include a non-empty `reasoning` field."
                )
            use_reasoning_prefill = self.use_reasoning_prefill and reasoning_present
            use_closed_think_prefill = self.optional_reasoning_supported and not use_reasoning_prefill
            assistant_prefill, assistant_content = format_assistant_response(
                gpt_response["content"],
                gpt_response.get("reasoning"),
                enable_reasoning=self.data_args.enable_reasoning,
                use_reasoning_prefill=use_reasoning_prefill,
                use_closed_think_prefill=use_closed_think_prefill,
            )

            user_input = (
                f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}{DEFAULT_IM_END_TOKEN}\n"
                f"{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n{assistant_prefill}"
            )
            gpt_response = f"{assistant_content}{DEFAULT_IM_END_TOKEN}\n"

            if DEFAULT_IMAGE_TOKEN in user_input:
                num_images = user_input.count(DEFAULT_IMAGE_TOKEN)
                # Slice the images list to get the images for the current turn.
                images_for_this_turn = images[image_curr_count : image_curr_count + num_images]
                inputs = processor(text=[user_input], images=images_for_this_turn, videos=videos, padding=False, do_resize=False, return_tensors='pt')
                prompt_input_ids = inputs['input_ids']
                prompt_mm_token_type_ids = get_mm_token_type_ids(inputs, prompt_input_ids)
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])
                image_curr_count += num_images

            elif DEFAULT_VIDEO_TOKEN in user_input:
                num_videos = user_input.count(DEFAULT_VIDEO_TOKEN)
                # Slice the videos list to get the videos for the current turn.
                videos_for_this_turn = videos[video_curr_count : video_curr_count + num_videos]
                video_datas_for_turn, video_metadatas_for_turn = zip(*videos_for_this_turn)
                video_datas_for_turn = list(video_datas_for_turn)
                video_metadatas_for_turn = list(video_metadatas_for_turn)

                inputs = processor(
                    text=[user_input],
                    images=images,
                    videos=video_datas_for_turn,
                    padding=False,
                    do_resize=False,
                    return_tensors='pt',
                    **video_kwargs,
                    video_metadata=video_metadatas_for_turn,
                )
                prompt_mm_token_type_ids = get_mm_token_type_ids(inputs, inputs["input_ids"])
                prompt_input_ids = inputs['input_ids']
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])
                video_curr_count += num_videos

            else:
                prompt_input_ids = processor.tokenizer(user_input, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']
                prompt_mm_token_type_ids = torch.zeros_like(prompt_input_ids, dtype=torch.long)

            response_input_ids = processor.tokenizer(gpt_response, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']
            response_mm_token_type_ids = torch.zeros_like(response_input_ids, dtype=torch.long)

            input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
            mm_token_type_ids = torch.cat([prompt_mm_token_type_ids, response_mm_token_type_ids], dim=1).squeeze(0)
            labels = torch.cat(
                [
                    torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),
                    response_input_ids.squeeze(0),
                ],
                dim=0,
            )

            all_input_ids.append(input_ids)
            all_labels.append(labels)
            all_mm_token_type_ids.append(mm_token_type_ids)

        # There is no need for eos or bos tokens in the input_ids
        # Qwen3.5 does not use them
        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)
        mm_token_type_ids = torch.cat(all_mm_token_type_ids, dim=0).to(torch.long)

        require_token_budget(len(input_ids), self.max_length, sample_id=self.list_data_dict[i].get("sample_id", i))

        attention_mask = (input_ids > -1000000).to(torch.long)

        data_dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            mm_token_type_ids=mm_token_type_ids,
        )

        if pixel_key and grid_key:
            pixel_values = torch.cat(all_pixel_values, dim=0)
            image_thw = torch.cat(all_image_grid_thw, dim=0)
            data_dict[pixel_key] = pixel_values
            data_dict[grid_key] = image_thw

        if self.generation_eval:
            data_dict["receipt_sample"] = self.list_data_dict[i]
        return data_dict

class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_pixel_video_values = []
        batch_video_thw = []
        batch_image_thw = []
        batch_second_per_grid_ts = []
        batch_mm_token_type_ids = []

        for example in examples:
            keys = example.keys()
            if "pixel_values_videos" in keys:
                batch_pixel_video_values.append(example["pixel_values_videos"])
                batch_video_thw.append(example["video_grid_thw"])
            elif "pixel_values" in keys:
                batch_pixel_values.append(example["pixel_values"])
                batch_image_thw.append(example["image_grid_thw"])

            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])
            batch_mm_token_type_ids.append(example["mm_token_type_ids"])

            if "second_per_grid_ts" in keys:
                batch_second_per_grid_ts.extend(example["second_per_grid_ts"])

        input_ids = pad_sequence(
            batch_input_ids, padding_side='right', padding_value=self.pad_token_id
        )

        attention_mask = input_ids != self.pad_token_id
        labels = pad_sequence(batch_label_ids, padding_side='right', padding_value=IGNORE_INDEX)
        mm_token_type_ids = pad_sequence(batch_mm_token_type_ids, padding_side='right', padding_value=0)

        data_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'mm_token_type_ids': mm_token_type_ids,
        }

        if len(batch_pixel_values) > 0:
            pixel_values = torch.cat(batch_pixel_values, dim=0)
            image_thw = torch.cat(batch_image_thw, dim=0)
            data_dict["pixel_values"] = pixel_values
            data_dict["image_grid_thw"] = image_thw

        if len(batch_pixel_video_values) > 0:
            pixel_video_values = torch.cat(batch_pixel_video_values, dim=0)
            video_thw = torch.cat(batch_video_thw, dim=0)
            data_dict["pixel_values_videos"] = pixel_video_values
            data_dict["video_grid_thw"] = video_thw

        if len(batch_second_per_grid_ts) > 0:
            data_dict["second_per_grid_ts"] = batch_second_per_grid_ts

        if "receipt_sample" in examples[0]:
            data_dict["receipt_samples"] = [example["receipt_sample"] for example in examples]
        return data_dict

def make_supervised_data_module(model_id, processor, data_args, max_length=32768, generation_eval=False):
    """Make dataset and collator for supervised fine-tuning."""
    sft_dataset = SupervisedDataset(
        data_path=data_args.data_path, processor=processor, data_args=data_args, model_id=model_id, max_length=max_length
    )
    eval_dataset = None
    if data_args.eval_path is not None:
        eval_args = copy.copy(data_args)
        eval_args.image_folder = data_args.eval_image_folder or data_args.image_folder
        eval_dataset = SupervisedDataset(
              data_path=data_args.eval_path,
              processor=processor,
              data_args=eval_args,
              model_id=model_id,
              max_length=max_length,
              generation_eval=generation_eval
          )
        
    data_collator = DataCollatorForSupervisedDataset(pad_token_id=processor.tokenizer.pad_token_id)

    return dict(train_dataset=sft_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)
