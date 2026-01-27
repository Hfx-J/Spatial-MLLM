from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast

from src.qwenvl.model.connector import get_connector
from src.qwenvl.model.spatial_encoder import VGGTSpatialEncoderConfig, VGGTSpatialEncoderPreTrainedModel


class SpatialMLLMConfig(Qwen2_5_VLConfig):
    model_type = "spatial-mllm"

    def __init__(self, spatial_config=None, connector_config=None, **kwargs):
        super().__init__(**kwargs)
        self.sub_configs["spatial_config"] = VGGTSpatialEncoderConfig
        if isinstance(spatial_config, dict):
            self.spatial_config = self.sub_configs["spatial_config"](**spatial_config)
        elif spatial_config is None:
            self.spatial_config = self.sub_configs["spatial_config"]()

        self.connector_config = connector_config if connector_config is not None else {}


class SpatialMLLMForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.spatial_encoder = VGGTSpatialEncoderPreTrainedModel(config.spatial_config)
        self.connector = get_connector(config)

        # Initialize weights and apply final processing
        self.post_init()

    # def forward(
    #     self,
    #     input_ids: Optional[torch.LongTensor] = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     past_key_values: Optional[List[torch.FloatTensor]] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     labels: Optional[torch.LongTensor] = None,
    #     use_cache: Optional[bool] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     return_dict: Optional[bool] = None,
    #     pixel_values: Optional[torch.Tensor] = None,
    #     pixel_values_videos: Optional[torch.FloatTensor] = None,
    #     image_grid_thw: Optional[torch.LongTensor] = None,
    #     video_grid_thw: Optional[torch.LongTensor] = None,
    #     image_tchw: Optional[List[torch.FloatTensor]] = None,
    #     video_tchw: Optional[List[torch.FloatTensor]] = None,
    #     rope_deltas: Optional[torch.LongTensor] = None,
    #     cache_position: Optional[torch.LongTensor] = None,
    #     second_per_grid_ts: Optional[torch.Tensor] = None,
    # ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    #     r"""
    #         labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
    #             Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
    #             config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
    #             (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    #     Returns:

    #     Example:

    #     ```python
    #     >>> from PIL import Image
    #     >>> import requests
    #     >>> from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    #     >>> model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    #     >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    #     >>> messages = [
    #         {
    #             "role": "user",
    #             "content": [
    #                 {"type": "image"},
    #                 {"type": "text", "text": "What is shown in this image?"},
    #             ],
    #         },
    #     ]
    #     >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    #     >>> image = Image.open(requests.get(url, stream=True).raw)

    #     >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    #     >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

    #     >>> # Generate
    #     >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    #     >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    #     "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
    #     ```"""

    #     output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    #     output_hidden_states = (
    #         output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    #     )
    #     return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    #     if inputs_embeds is None:
    #         inputs_embeds = self.model.embed_tokens(input_ids)
    #         if pixel_values is not None:
    #             assert image_tchw is not None, "`image_tchw` must be provided when `pixel_values` is not None."
    #             pixel_values = pixel_values.type(self.visual.dtype)
    #             image_tchw = [image_tchw_i.type(self.visual.dtype) for image_tchw_i in image_tchw]

    #             # get image embeddings
    #             image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
    #             n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
    #             n_image_features = image_embeds.shape[0]
    #             if n_image_tokens != n_image_features:
    #                 raise ValueError(
    #                     f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
    #                 )

    #             # get spatial embeddings
    #             spatial_embeds_list, patch_start_idx = self.spatial_encoder(image_tchw)

    #             print(f"image_embeds shape: {image_embeds.shape} spatial_embeds_list lengths: {[se.shape for se in spatial_embeds_list]}")

    #             # fuse video and spatial embeddings
    #             fused_embeds = self.connector(
    #                 image_embeds=image_embeds,
    #                 spatial_embeds_list=spatial_embeds_list,
    #                 patch_start_idx=patch_start_idx,
    #                 grid_thw=image_grid_thw,
    #             )

    #             mask = input_ids == self.config.image_token_id
    #             mask_unsqueezed = mask.unsqueeze(-1)
    #             mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
    #             image_mask = mask_expanded.to(inputs_embeds.device)

    #             fused_embeds = fused_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
    #             inputs_embeds = inputs_embeds.masked_scatter(image_mask, fused_embeds)

    #         if pixel_values_videos is not None:
    #             assert video_tchw is not None, "`video_tchw` must be provided when `pixel_values_videos` is not None."
    #             pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
    #             video_tchw = [video_tchw_i.type(self.visual.dtype) for video_tchw_i in video_tchw]

    #             # get video embeddings
    #             video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
    #             n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
    #             n_video_features = video_embeds.shape[0]
    #             if n_video_tokens != n_video_features:
    #                 raise ValueError(
    #                     f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
    #                 )

    #             # get spatial embeddings
    #             spatial_embeds_list, patch_start_idx = self.spatial_encoder(video_tchw, grid_thw=video_grid_thw)

    #             print(f"video_tchw lengths: {len(video_tchw)},{video_tchw[0].shape}")
    #             print(f"pixel_values_videos: {pixel_values_videos.shape}")
    #             print(f"video_embeds shape: {video_embeds.shape}")
    #             print(f"spatial_embeds_list lengths: {len(spatial_embeds_list)},{len(spatial_embeds_list[0])},{spatial_embeds_list[0][0].shape}")


    #             # fuse video and spatial embeddings
    #             fused_embeds = self.connector(
    #                 video_embeds=video_embeds,
    #                 spatial_embeds_list=spatial_embeds_list,
    #                 patch_start_idx=patch_start_idx,
    #                 grid_thw=video_grid_thw,
    #             )

    #             mask = input_ids == self.config.video_token_id
    #             mask_unsqueezed = mask.unsqueeze(-1)
    #             mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
    #             video_mask = mask_expanded.to(inputs_embeds.device)

    #             fused_embeds = fused_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
    #             inputs_embeds = inputs_embeds.masked_scatter(video_mask, fused_embeds)

    #         if attention_mask is not None:
    #             attention_mask = attention_mask.to(inputs_embeds.device)

    #     # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
    #     if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
    #         # calculate RoPE index once per generation in the pre-fill stage only
    #         # 计算旋转位置编码的增量只在生成的预填充阶段进行一次
    #         if (
    #             (cache_position is not None and cache_position[0] == 0)
    #             or self.rope_deltas is None
    #             or (past_key_values is None or past_key_values.get_seq_length() == 0)
    #         ):
    #             position_ids, rope_deltas = self.get_rope_index(
    #                 input_ids,
    #                 image_grid_thw,
    #                 video_grid_thw,
    #                 second_per_grid_ts,
    #                 attention_mask,
    #             )
    #             self.rope_deltas = rope_deltas
    #         # then use the prev pre-calculated rope-deltas to get the correct position ids
    #         else:
    #             batch_size, seq_length, _ = inputs_embeds.shape
    #             delta = (
    #                 (cache_position[0] + self.rope_deltas).to(inputs_embeds.device) if cache_position is not None else 0
    #             )
    #             position_ids = torch.arange(seq_length, device=inputs_embeds.device)
    #             position_ids = position_ids.view(1, -1).expand(batch_size, -1)
    #             if cache_position is not None:  # otherwise `deltas` is an int `0`
    #                 delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
    #             position_ids = position_ids.add(delta)
    #             position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    #     outputs = self.model(
    #         input_ids=None,
    #         position_ids=position_ids,
    #         attention_mask=attention_mask,
    #         past_key_values=past_key_values,
    #         inputs_embeds=inputs_embeds,
    #         use_cache=use_cache,
    #         output_attentions=output_attentions,
    #         output_hidden_states=output_hidden_states,
    #         return_dict=return_dict,
    #         cache_position=cache_position,
    #     )

    #     hidden_states = outputs[0]
    #     logits = self.lm_head(hidden_states)

    #     loss = None
    #     if labels is not None:
    #         # Upcast to float if we need to compute the loss to avoid potential precision issues
    #         logits = logits.float()
    #         # Shift so that tokens < n predict n
    #         shift_logits = logits[..., :-1, :].contiguous()
    #         shift_labels = labels[..., 1:].contiguous()
    #         # Flatten the tokens
    #         loss_fct = CrossEntropyLoss()
    #         shift_logits = shift_logits.view(-1, self.config.vocab_size)
    #         shift_labels = shift_labels.view(-1)
    #         # Enable model parallelism
    #         shift_labels = shift_labels.to(shift_logits.device)
    #         loss = loss_fct(shift_logits, shift_labels)

    #     if not return_dict:
    #         output = (logits,) + outputs[1:]
    #         return (loss,) + output if loss is not None else output

    #     return Qwen2_5_VLCausalLMOutputWithPast(
    #         loss=loss,
    #         logits=logits,
    #         past_key_values=outputs.past_key_values,
    #         hidden_states=outputs.hidden_states,
    #         attentions=outputs.attentions,
    #         rope_deltas=self.rope_deltas,
    #     )

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            pixel_values: Optional[torch.Tensor] = None,
            pixel_values_videos: Optional[torch.FloatTensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            image_tchw: Optional[List[torch.FloatTensor]] = None,
            video_tchw: Optional[List[torch.FloatTensor]] = None,
            rope_deltas: Optional[torch.LongTensor] = None,
            cache_position: Optional[torch.LongTensor] = None,
            second_per_grid_ts: Optional[torch.Tensor] = None,
        ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
            
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            if inputs_embeds is None:
                inputs_embeds = self.model.embed_tokens(input_ids)
                
                # --- Image Processing Block (Same logic can be applied here if needed, keeping original for safety) ---
                if pixel_values is not None:
                    assert image_tchw is not None, "`image_tchw` must be provided when `pixel_values` is not None."
                    pixel_values = pixel_values.type(self.visual.dtype)
                    image_tchw = [image_tchw_i.type(self.visual.dtype) for image_tchw_i in image_tchw]

                    image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                    n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                    n_image_features = image_embeds.shape[0]
                    if n_image_tokens != n_image_features:
                        raise ValueError(
                            f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                        )

                    spatial_embeds_list, patch_start_idx = self.spatial_encoder(image_tchw)

                    # Original Image Connector (Use same logic as below if you want consistency, otherwise keep connector)
                    fused_embeds = self.connector(
                        image_embeds=image_embeds,
                        spatial_embeds_list=spatial_embeds_list,
                        patch_start_idx=patch_start_idx,
                        grid_thw=image_grid_thw,
                    )

                    mask = input_ids == self.config.image_token_id
                    mask_unsqueezed = mask.unsqueeze(-1)
                    mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                    image_mask = mask_expanded.to(inputs_embeds.device)
                    fused_embeds = fused_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds.masked_scatter(image_mask, fused_embeds)

                # --- Video Processing Block (Modified) ---
                if pixel_values_videos is not None:
                    assert video_tchw is not None, "`video_tchw` must be provided when `pixel_values_videos` is not None."
                    pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                    video_tchw = [video_tchw_i.type(self.visual.dtype) for video_tchw_i in video_tchw]

                    # 1. Get compressed video embeddings (Shape: [Total_Compressed_Tokens, 2048])
                    video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                    
                    n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                    n_video_features = video_embeds.shape[0]
                    if n_video_tokens != n_video_features:
                        raise ValueError(
                            f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                        )

                    # 2. Get high-res spatial embeddings (List of per-video features)
                    # Structure: List[Batch_i] -> List[Layers] -> Tensor[T, Tokens, Dim]
                    # We typically take the last layer: spatial_embeds_list[i][-1]
                    spatial_embeds_list, patch_start_idx = self.spatial_encoder(video_tchw, grid_thw=video_grid_thw)

                    fused_video_embeds_list = []
                    video_embeds_offset = 0

                    for i, spatial_layers in enumerate(spatial_embeds_list):
                        spatial_feat = spatial_layers[-1] 
                        
                        # === FIX START: 获取当前索引 ===
                        if isinstance(patch_start_idx, list):
                            current_start_idx = patch_start_idx[i]
                        else:
                            current_start_idx = patch_start_idx
                        
                        if isinstance(current_start_idx, torch.Tensor):
                            current_start_idx = current_start_idx.item()
                        # === FIX END ===

                        T, _, H, W = video_tchw[i].shape
                        H_grid, W_grid = H // 14, W // 14
                        num_patches = H_grid * W_grid
                        
                        # 使用 current_start_idx 进行切片
                        patches_raw = spatial_feat[:, current_start_idx : current_start_idx + num_patches, :]
                        
                        token_indices = torch.arange(spatial_feat.shape[1], device=spatial_feat.device)
                        # 使用 current_start_idx 进行掩码计算
                        special_mask = (token_indices < current_start_idx) | (token_indices >= current_start_idx + num_patches)
                        special_raw = spatial_feat[:, special_mask, :]
                        # -- Step B: Align Patches (Downsample to match video_embeds) --
                        # 1. Restore Grid: [T, H_grid, W_grid, D]
                        patches_grid = patches_raw.view(T, H_grid, W_grid, -1)
                        
                        # 2. Time Pooling (T -> T/2): Matches video_embeds stride=2
                        # [T/2, 2, H, W, D] -> mean -> [T/2, H, W, D]
                        patches_pooled_time = patches_grid.view(T // 2, 2, H_grid, W_grid, -1).mean(dim=1)
                        
                        # 3. Spatial Pooling (2x2): Matches video_embeds spatial pooling
                        # Permute to [N, D, H, W] for pool2d
                        patches_permuted = patches_pooled_time.permute(0, 3, 1, 2) 
                        patches_downsampled = torch.nn.functional.avg_pool2d(patches_permuted, kernel_size=2, stride=2)
                        
                        # 4. Flatten back: [T/2, D, H/2, W/2] -> [Flattened_Tokens, D]
                        patches_aligned = patches_downsampled.permute(0, 2, 3, 1).reshape(-1, patches_raw.shape[-1])
                        
                        # -- Step C: Incorporate Special Tokens (Context Injection) --
                        if special_raw.shape[1] > 0:
                            # 1. Time Pooling (Align with T/2)
                            special_pooled_time = special_raw.view(T // 2, 2, -1, special_raw.shape[-1]).mean(dim=1)
                            
                            # 2. Global Aggregation (Get one context vector per time-step)
                            # Shape: [T/2, 1, D]
                            context_per_frame = special_pooled_time.mean(dim=1, keepdim=True)
                            
                            # 3. Broadcast to all spatial tokens in that frame
                            # Determine how many spatial tokens exist per frame after downsampling
                            tokens_per_frame = patches_aligned.shape[0] // (T // 2)
                            
                            # Expand: [T/2, Tokens_Per_Frame, D]
                            context_expanded = context_per_frame.expand(-1, tokens_per_frame, -1)
                            context_aligned = context_expanded.reshape(-1, special_raw.shape[-1])
                            
                            # Add Context to Patches (Residual connection style)
                            patches_aligned = patches_aligned + context_aligned

                        # -- Step D: Final Fusion with Video Embeds --
                        # Slice the corresponding part from the flat video_embeds
                        n_current_tokens = patches_aligned.shape[0]
                        curr_video_embeds = video_embeds[video_embeds_offset : video_embeds_offset + n_current_tokens]
                        
                        # Fusion: Element-wise addition 
                        # (Maintains dim=2048, acts as refining the compressed features with high-res + context info)
                        fused_chunk = curr_video_embeds + patches_aligned
                        
                        fused_video_embeds_list.append(fused_chunk)
                        video_embeds_offset += n_current_tokens

                    # Concatenate all processed videos back into one batch tensor
                    fused_embeds = torch.cat(fused_video_embeds_list, dim=0)
                    # === Custom Fusion Logic End ===

                    mask = input_ids == self.config.video_token_id
                    mask_unsqueezed = mask.unsqueeze(-1)
                    mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                    video_mask = mask_expanded.to(inputs_embeds.device)

                    fused_embeds = fused_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds.masked_scatter(video_mask, fused_embeds)

                if attention_mask is not None:
                    attention_mask = attention_mask.to(inputs_embeds.device)

            # ... (Rest of the function remains unchanged: RoPE calculation, Output generation, etc.)
            
            # if we get 4D attention mask we cannot calculate rope deltas anymore.
            if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
                if (
                    (cache_position is not None and cache_position[0] == 0)
                    or self.rope_deltas is None
                    or (past_key_values is None or past_key_values.get_seq_length() == 0)
                ):
                    position_ids, rope_deltas = self.get_rope_index(
                        input_ids,
                        image_grid_thw,
                        video_grid_thw,
                        second_per_grid_ts,
                        attention_mask,
                    )
                    self.rope_deltas = rope_deltas
                else:
                    batch_size, seq_length, _ = inputs_embeds.shape
                    delta = (
                        (cache_position[0] + self.rope_deltas).to(inputs_embeds.device) if cache_position is not None else 0
                    )
                    position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                    position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                    if cache_position is not None:
                        delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                    position_ids = position_ids.add(delta)
                    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

            outputs = self.model(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)

            loss = None
            if labels is not None:
                logits = logits.float()
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

            if not return_dict:
                output = (logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output

            return Qwen2_5_VLCausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                rope_deltas=self.rope_deltas,
            )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            use_cache=use_cache,
            **kwargs,
        )

        # Qwen2-5-VL position_ids are prepareed with rope_deltas in forward
        model_inputs["position_ids"] = None

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
            model_inputs["image_tchw"] = None
            model_inputs["video_tchw"] = None

        return model_inputs