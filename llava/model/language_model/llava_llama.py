# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# This file is modified from https://github.com/haotian-liu/LLaVA/


import os
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torchvision.transforms as T
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from llava.model.loss import soft_cross_entropy, proxy_loss
from llava.model.utils.packing import set_seqlens_in_batch
from llava.train.sequence_parallel.globals import get_pg_manager
from llava.utils.logging import logger

from ...train.utils import calculate_loss_weight
from ..configuration_llava import LlavaConfig
from ..llava_arch import LlavaMetaForCausalLM, LlavaMetaModel
import einops

from llava.utils import distributed as dist


# Short labels for the L4P modalities — used in the per-step loss log line.
_TASK_SHORT = {"depth": "D", "flow_2d_backward": "F", "dyn_mask": "M", "camray": "C"}


def _fmt_loss_breakdown(step: int, sft: float, ld, ed: dict, ed_trained=None) -> str:
    """Format the per-step loss decomposition for logging.

    `ld` may be a float (current single-bucket L_LD per paper §3.2) or a
    dict[task -> float] if per-modality LD is added later (option C).
    Tasks in `ed` but not in `ed_trained` are monitored-only and shown in parens.
    """
    short = lambda t: _TASK_SHORT.get(t, t)
    parts = [f"SFT={sft:.4f}"]
    if isinstance(ld, dict):
        parts.append("LD " + " ".join(f"{short(k)}={v:.2f}" for k, v in ld.items()))
    else:
        parts.append(f"LD={ld:.2f}")
    if ed:
        trained = set(ed.keys()) if ed_trained is None else set(ed_trained)
        items = [
            (f"{short(k)}={v:.2f}" if k in trained else f"({short(k)}={v:.2f})")
            for k, v in ed.items()
        ]
        parts.append("ED " + " ".join(items))
    return f"\n[step {step}] " + " | ".join(parts)


class LlavaLlamaConfig(LlavaConfig):
    model_type = "llava_llama"


# FIXME we will follow the convention to add a new class for CausalLM in the future
class LlavaLlamaModel(LlavaMetaModel, LlavaMetaForCausalLM, PreTrainedModel):
    config_class = LlavaLlamaConfig
    main_input_name = "input_embeds"
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True

    # Dump an L4P-prediction video every N distillation steps (rank 0 only).
    L4P_VIZ_EVERY = 10

    def __init__(self, config: LlavaLlamaConfig = None, *args, **kwargs) -> None:
        super().__init__(config)
        self.init_vlm(config=config, *args, **kwargs)

        self.l4p_step = 0

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs,
    ):
        if hasattr(cls, "load_pretrained"):
            return cls.load_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                config=config,
                cache_dir=cache_dir,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                use_safetensors=use_safetensors,
                **kwargs,
            )
        return super(LlavaLlamaModel, cls).from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            **kwargs,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        media: Optional[Dict[str, List[torch.Tensor]]] = None,
        images: Optional[torch.FloatTensor] = None,
        media_config: Optional[List] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        packing: bool = True,
        seqlens_in_batch: Optional[torch.LongTensor] = None,
        dpo_forward: bool = False,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        self.freezed_module_patch()

        if images is not None:
            if media is not None:
                raise ValueError("Both 'media' and 'images' are provided. Please provide only one.")
            logger.warning("The 'images' argument is deprecated. Please use 'media' instead.")
            media = {"image": images}

        if media_config is None:
            media_config = defaultdict(dict)

        if inputs_embeds is None:
            inputs_embeds, labels, attention_mask = self._embed(input_ids, media, media_config, labels, attention_mask)

        if packing and self.training and not dpo_forward:
            if seqlens_in_batch is None:
                seqlens_in_batch = torch.sum(attention_mask, dim=1)
            set_seqlens_in_batch(seqlens_in_batch)

            (inputs_embeds, attention_mask, position_ids, labels) = self.repack_multimodal_data(
                inputs_embeds, attention_mask, position_ids, labels
            )

        # logger.debug(self.llm) # Qwen2ForCausalLM
        # logger.debug(kwargs)
        kwargs.update({
            "output_hidden_states": True
        })
        # (lm_head): Linear(in_features=3584, out_features=151651, bias=False)
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            labels=labels,
            **kwargs,
        )

        if self.training:
            if getattr(self.config, "time_token_ids", []):
                outputs.loss = soft_cross_entropy(
                    outputs.logits,
                    labels,
                    soft_tokens=self.config.time_token_ids,
                    std=self.config.soft_ce_std,
                )

            if 'video' in media and media['video'] and not getattr(self.config, "disable_distillation", False):
                frame_feats = self.extract_image_region_hidden_states(outputs, input_ids, labels)
                distill_loss, breakdown = self.perception_distillation(frame_feats, media['video'])
                if distill_loss is not None:
                    # Paper Sec. 4.2: total = L_SFT + L_LD + L_ED.
                    logger.info(_fmt_loss_breakdown(
                        self.l4p_step,
                        outputs.loss.item(),
                        breakdown["LD"],
                        breakdown["ED"],
                        breakdown["ED_TRAINED"],
                    ))
                    outputs.loss = outputs.loss + distill_loss

        # Loss rescale for SP
        if get_pg_manager() is not None:
            loss_weight = calculate_loss_weight(labels)
            outputs.loss = outputs.loss * loss_weight

        if dpo_forward:
            return outputs.logits, labels

        return outputs

    def extract_image_region_hidden_states(self, outputs, input_ids, labels=None, num_frames=16):
        """
        Extract hidden states for each image region (all tokens between consecutive image tokens).

        Args:
            outputs: Model outputs from self.llm()
            input_ids: Input IDs tensor
            labels: Optional labels tensor

        Returns:
            List of tensors per batch, where each tensor is (num_images, num_tokens_per_image, hidden_dim)
        """
        from llava.constants import IGNORE_INDEX

        if not hasattr(outputs, 'hidden_states') or outputs.hidden_states is None:
            raise ValueError("outputs.hidden_states is None. Set output_hidden_states=True in model forward")

        hidden_states = outputs.hidden_states[-1]  # (batch_size, seq_len, hidden_dim)
        image_token_id = self.tokenizer.media_token_ids["video"]

        batch_size = len(input_ids)
        all_batch_image_hidden_states = []

        end_positions = torch.where(labels == self.tokenizer.added_tokens_encoder["<|im_end|>"])[1]
        assert(len(end_positions) == batch_size)

        for b in range(batch_size):
            video_positions = (input_ids[b] == image_token_id).nonzero(as_tuple=True)[0]
            # Image-only / no-video samples (e.g. *_dev sets without video) have no
            # video token to anchor the region. Skip them — L4P loss only applies
            # to video samples and `media['video']` excludes them too, so the
            # resulting stack still aligns with len(media['video']).
            if len(video_positions) == 0:
                continue

            h = w = int(math.sqrt(self.num_image_tokens))
            assert(h * w == self.num_image_tokens)

            # After repack_multimodal_data all samples concatenate along seq dim
            # into a single row; index hidden_states with [0, ...] and add a
            # per-sample offset here. The offset is needed for every b >= 1.
            start_pos = video_positions[0] + 1
            if b >= 1:
                start_pos += end_positions[b-1] + 1
            end_pos = start_pos + self.num_image_tokens * num_frames
            assert(labels[0, start_pos:end_pos].max() == IGNORE_INDEX)
            batch_image_hidden = hidden_states[0, start_pos:end_pos, :]

            batch_image_hidden = einops.rearrange(
                batch_image_hidden, "(n L) c -> n c L", n=num_frames)

            all_batch_image_hidden_states.append(
                einops.rearrange(batch_image_hidden[..., :self.num_image_tokens], "n c (h w) -> n c h w",  h=h, w=w))
        all_batch_image_hidden_states = torch.stack(all_batch_image_hidden_states, dim=0)
        return all_batch_image_hidden_states

    def perception_distillation(
        self,
        frame_feats: torch.Tensor,
        videos: list[torch.Tensor],
    ) -> Union[torch.Tensor, None]:
        from llava.utils import distributed as dist

        with torch.no_grad():
            l4p_videos = [T.functional.resize(
                video, [self.encoder_l4p.img_size, self.encoder_l4p.img_size]) for video in videos]
            l4p_videos = [einops.rearrange((video + 1) / 2, "t c h w -> c t h w") for video in l4p_videos]
            l4p_outputs = self.encoder_l4p.forward(l4p_videos)

        # Student (DistillationModel) and teacher (encoder_l4p) must distill from
        # the same encoder layers. The two hooks_idx lists are defined separately
        # in builder.py / base.py — this assert catches drift early.
        hooks_idx = list(self.distillation.hooks_idx)
        assert hooks_idx == list(self.encoder_l4p.hooks_idx), (
            f"hooks_idx mismatch: distillation={hooks_idx} vs "
            f"encoder_l4p={list(self.encoder_l4p.hooks_idx)}"
        )

        # Assumes --num_video_frames=16. Tied to: encoder_l4p.window_size=
        # [16,224,224] stride 8 (1 window/video) and VideoMAE tubelet_size=2
        # (16 frames → 8 temporal tokens). DistillationModel.forward does
        # `hidden_state[:, ::2]` to match. Move all four together if you change
        # num_video_frames.
        n_temporal = 8

        l4p_encs = []
        for l4p_output in l4p_outputs:
            feats_enc_hooked = [l4p_output.feats_enc[0, hook] for hook in hooks_idx]
            feats_enc_hooked = torch.stack(feats_enc_hooked, dim=1)  # 1, hook, 2048, 1408
            l4p_enc = einops.rearrange(
                feats_enc_hooked,
                "b hook (n h w) c -> b hook n c h w",
                n=n_temporal, h=16, w=16,
            )
            # 1, len(hooks_idx), n_temporal, 1408, 16, 16
            l4p_encs.append(l4p_enc)

        scale_factor = 1.
        l4p_encs = torch.cat(l4p_encs, dim=0) / scale_factor

        l4p_encs_pred = self.distillation.forward(frame_feats)

        l4p_projs_filled = []
        num_hooks = 41
        for l4p_proj in l4p_encs_pred:
            # Fill the full 41-hook stack — slots we don't distill from get a
            # sentinel of -10000 so the decoder treats them as masked-out.
            l4p_proj_fill = torch.stack(
                [l4p_proj[hooks_idx.index(i)] if i in hooks_idx else torch.zeros_like(l4p_proj[0]) - 10000 for i in range(num_hooks)],
                dim=0)[None]
            l4p_proj_fill = einops.rearrange(l4p_proj_fill, "b l n c h w -> b l (n h w) c")[:, :, None]
            l4p_projs_filled.append(l4p_proj_fill * scale_factor)

        l4p_projs_outputs = self.encoder_l4p.decode(l4p_projs_filled)

        # --distill_tasks selects which L_ED terms feed back into the loss.
        # Empty → use everything the encoder provides; we intersect with
        # encoder_l4p.tasks to silently drop typos / stale configs. Note that
        # encoder_l4p.decode runs ALL tasks regardless — this only gates the
        # L_ED loss contribution.
        configured = list(getattr(self.config, "distill_tasks", []) or [])
        available = list(self.encoder_l4p.tasks)
        tasks = [t for t in configured if t in available] if configured else available
        if dist.is_main() and self.l4p_step % self.L4P_VIZ_EVERY == 0:
            self._dump_l4p_viz(l4p_videos[0], l4p_projs_outputs[0], tasks)
        self.l4p_step += 1

        # `ed_weights` is a comma-separated "task=w" string; missing tasks → 1.0.
        ld_weight = float(getattr(self.config, "ld_weight", 1.0))
        ed_weights_raw = getattr(self.config, "ed_weights", "") or ""
        ed_weights: dict[str, float] = {}
        for pair in ed_weights_raw.split(","):
            pair = pair.strip()
            if "=" in pair:
                k, v = pair.split("=", 1)
                ed_weights[k.strip()] = float(v)

        ld_loss = proxy_loss(l4p_encs_pred, l4p_encs)
        new_loss = ld_weight * ld_loss
        ed_losses: dict[str, float] = {}

        # Compute (and log) L_ED for every available task even when distill_tasks
        # omits some — keeps the monitoring signal alive. Logged values are
        # *unweighted* so weights can be tuned against natural-scale magnitudes.
        for task in available:
            preds = torch.stack([getattr(l4p_output, f"pred_{task}") for l4p_output in l4p_projs_outputs], dim=0)
            targets = torch.stack([getattr(l4p_output, f"pred_{task}") for l4p_output in l4p_outputs], dim=0)
            if task == "depth":
                preds = torch.log(preds + 1e-8)
                targets = torch.log(targets + 1e-8)

            if task in tasks:
                ed_task = proxy_loss(preds, targets)
                new_loss = new_loss + ed_weights.get(task, 1.0) * ed_task
                ed_losses[task] = ed_task.detach().item()
            else:
                with torch.no_grad():
                    ed_losses[task] = proxy_loss(preds, targets).item()

        breakdown = {
            "LD": ld_loss.detach().item(),
            "LD_W": ld_weight,
            "ED": ed_losses,
            "ED_W": {t: ed_weights.get(t, 1.0) for t in tasks},
            "ED_TRAINED": list(tasks),
        }
        return new_loss, breakdown

    def _l4p_viz_dir(self) -> Path:
        """Per-run directory for L4P visualizations.

        `output_dir` is the HF trainer's checkpoint dir (e.g. `runs/train/<run>/model`);
        we drop viz alongside it under `runs/train/<run>/viz/` rather than inside the
        checkpoint folder. Falls back to `results/` if `output_dir` was never stashed
        (e.g. when the model is used outside training).
        """
        output_dir = getattr(self.config, "output_dir", None)
        return Path(output_dir).parent / "viz" if output_dir else Path("results")

    def _dump_l4p_viz(
        self,
        l4p_video: torch.Tensor,
        l4p_proj_output,
        tasks: List[str],
    ) -> None:
        from ..low_level_perception.viz.visualize import visualize_tasks

        visualize_tasks(
            l4p_video,
            l4p_proj_output,
            tasks,
            out_dir=str(self._l4p_viz_dir()),
            seq_name=f"pred-l4p-{self.l4p_step}",
        )


AutoConfig.register("llava_llama", LlavaLlamaConfig)
AutoModel.register(LlavaLlamaConfig, LlavaLlamaModel)
