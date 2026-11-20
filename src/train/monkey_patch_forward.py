from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ModelOutputWithPast
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeModelOutputWithPast
import torch
from typing import Optional, Union
import transformers.models.qwen3_5.modeling_qwen3_5
import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe
from transformers.utils import TransformersKwargs
from transformers.processing_utils import Unpack
from transformers.cache_utils import Cache


def _flatten_vision_features(vision_outputs):
    pooled = getattr(vision_outputs, "pooler_output", vision_outputs)
    if isinstance(pooled, torch.Tensor):
        return pooled
    if isinstance(pooled, (tuple, list)):
        return torch.cat(list(pooled), dim=0)
    raise TypeError(f"Unsupported vision output type: {type(vision_outputs)!r}")


def _make_dummy_qwen3_visual_inputs(visual):
    dummy_grid = torch.tensor([[1, 32, 32]], device=visual.device)
    patch_embed = visual.patch_embed
    patch_dim = (
        patch_embed.in_channels
        * patch_embed.temporal_patch_size
        * patch_embed.patch_size
        * patch_embed.patch_size
    )
    num_patches = int(dummy_grid.prod().item())
    dummy_pixel = torch.zeros((num_patches, patch_dim), device=visual.device, dtype=visual.dtype)
    return dummy_pixel, dummy_grid


def _expand_video_grid_to_frames(video_grid_thw):
    if video_grid_thw is None:
        return None

    frame_grids = []
    for grid in video_grid_thw:
        num_frames = int(grid[0].item())
        per_frame_grid = grid.unsqueeze(0).expand(num_frames, -1).clone()
        per_frame_grid[:, 0] = 1
        frame_grids.append(per_frame_grid)

    if not frame_grids:
        return video_grid_thw

    return torch.cat(frame_grids, dim=0)


def replace_qwen3_5_with_mixed_modality_forward():
    transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5Model.forward = qwen3_5_mixed_modality_forward

def replace_qwen3_5_moe_with_mixed_modality_forward():
    transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.Qwen3_5MoeModel.forward = qwen3_5_moe_mixed_modality_forward


def _qwen3_5_mixed_modality_forward_impl(
    self,
    *,
    output_cls,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    mm_token_type_ids: Optional[torch.IntTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[TransformersKwargs],
):
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    if pixel_values is None and pixel_values_videos is None:
        dummy_pixel, dummy_grid = _make_dummy_qwen3_visual_inputs(self.visual)
        image_outputs = self.get_image_features(dummy_pixel, dummy_grid, return_dict=True)
        image_embeds = _flatten_vision_features(image_outputs).to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds += image_embeds.mean() * 0

    if pixel_values is not None:
        image_outputs = self.get_image_features(pixel_values, image_grid_thw, return_dict=True)
        image_embeds = _flatten_vision_features(image_outputs).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        video_outputs = self.get_video_features(pixel_values_videos, video_grid_thw, return_dict=True)
        video_embeds = _flatten_vision_features(video_outputs).to(inputs_embeds.device, inputs_embeds.dtype)
        _, video_mask = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    if position_ids is None:
        rope_video_grid_thw = _expand_video_grid_to_frames(video_grid_thw)
        position_ids = self.compute_3d_position_ids(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=rope_video_grid_thw,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            mm_token_type_ids=mm_token_type_ids,
        )

    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        **kwargs,
    )

    return output_cls(
        **outputs,
        rope_deltas=self.rope_deltas,
    )


def qwen3_5_mixed_modality_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    mm_token_type_ids: Optional[torch.IntTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> Union[tuple, Qwen3_5ModelOutputWithPast]:
    return _qwen3_5_mixed_modality_forward_impl(
        self,
        output_cls=Qwen3_5ModelOutputWithPast,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        mm_token_type_ids=mm_token_type_ids,
        cache_position=cache_position,
        **kwargs,
    )


def qwen3_5_moe_mixed_modality_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    mm_token_type_ids: Optional[torch.IntTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> Union[tuple, Qwen3_5MoeModelOutputWithPast]:
    return _qwen3_5_mixed_modality_forward_impl(
        self,
        output_cls=Qwen3_5MoeModelOutputWithPast,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        mm_token_type_ids=mm_token_type_ids,
        cache_position=cache_position,
        **kwargs,
    )
