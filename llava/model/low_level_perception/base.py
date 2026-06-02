# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Thin adapter wrapping the upstream NVlabs/L4P submodule (third_party/L4P) so
# 4D-RGPT can use it as the frozen perception teacher for explicit-distillation
# (paper L_ED) without rewriting downstream call sites.
#
# Upstream's `L4P_VideoMAE.forward(batch_dict, tasks)` is dict-based and takes
# a single batch dict per call. The rest of 4D-RGPT expects:
#   - `forward(videos: List[Tensor])`            iterates per-video
#   - `decode(l4p_projs: list[Tensor])`          student-side path, not upstream
#   - `LowLevelPerceptionOutput` dataclass with `.feats_enc` and `.pred_<task>`
# This file bridges the two without altering upstream behavior.

from dataclasses import dataclass
from typing import List, Optional

import einops
import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

from l4p.models.l4p_videomae import L4P_VideoMAE
from l4p.models.task_heads.dense_heads import (
    VideoMAEDepthDPTHead,
    VideoMAEDynMaskDPTHead,
    VideoMAEFlowDPTHead,
    VideoMAETraj3DDPTHead,
)
from l4p.utils.geometry_utils import normalize_intrinsics

from llava.utils.logging import logger


# ImageNet mean/std — upstream applies these inside its dataset transforms; we
# apply them in `_normalize` since 4D-RGPT passes raw [0,1] video tensors.
L4P_INPUT_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None, None]
L4P_INPUT_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None, None]

# DPT hook layer indices read out of the VideoMAE encoder.
HOOKS_IDX = (14, 21, 28, 36)


def default_intrinsics(h: int, w: int) -> torch.Tensor:
    """Identity-ish intrinsics: fx=fy=min(h,w), principal at (w/2, h/2). Shape (1,4,4)."""
    fx = fy = min(h, w)
    cx, cy = w / 2, h / 2
    return torch.tensor(
        [[[fx, 0, cx, 0],
          [0, fy, cy, 0],
          [0, 0, 1, 0],
          [0, 0, 0, 1]]],
        dtype=torch.float32,
    )


class LowLevelPerceptionConfig(PretrainedConfig):
    model_type = "low_level_perception"

    def __init__(self, low_level_perception_type: Optional[str] = None, **kwargs):
        super().__init__()
        self.low_level_perception_type = low_level_perception_type


@dataclass
class LowLevelPerceptionOutput:
    feats_enc: Optional[torch.Tensor] = None
    pred_dyn_mask: Optional[torch.Tensor] = None
    pred_depth: Optional[torch.Tensor] = None
    pred_flow_2d_backward: Optional[torch.Tensor] = None
    pred_camray: Optional[torch.Tensor] = None


def _build_task_heads(hooks_idx) -> nn.ModuleDict:
    """Construct task heads matching third_party/L4P/configs/model.yaml.

    `camray` slot defaults to VideoMAETraj3DDPTHead (the YAML default). For the
    distillation path, train.py swaps it to VideoMAECameraDPTHead post-build —
    see comment in llava/train/train.py. We keep the Traj3D default here so the
    pretrained ckpt loads cleanly into the unswapped state dict.

    track_2d is intentionally omitted — sparse-head loss interacts badly with
    our packed-batch forward and we don't distill against it.
    """
    hooks_idx = list(hooks_idx)
    return nn.ModuleDict({
        "depth": VideoMAEDepthDPTHead(
            task_name="depth", out_nchan=1, depth_fn="exp",
            hooks_idx=hooks_idx, align_window_overlap_fn="inverse",
        ),
        "dyn_mask": VideoMAEDynMaskDPTHead(
            task_name="dyn_mask", out_nchan=1, apply_fn="linear", hooks_idx=hooks_idx,
        ),
        "flow_2d_backward": VideoMAEFlowDPTHead(
            task_name="flow_2d_backward", out_nchan=2, hooks_idx=hooks_idx,
        ),
        "camray": VideoMAETraj3DDPTHead(
            task_name="traj3d", hooks_idx=hooks_idx,
            use_intrinsics=False, fixed_intrinsics=True,
        ),
    })


def _primary_tensor(task_out) -> torch.Tensor:
    """Pick the main estimate tensor out of a head's dict-of-tensors return.

    Upstream heads return `{f"{task_name}_est_{suffix}": tensor, ...}` —
    sometimes with additional aux outputs (e.g. estimated intrinsics). We
    take the first tensor value that doesn't contain `intrinsics` in its key.
    """
    if torch.is_tensor(task_out):
        return task_out
    for k, v in task_out.items():
        if torch.is_tensor(v) and "intrinsics" not in k:
            return v
    raise KeyError(f"no primary tensor in task_out keys={list(task_out)}")


class LowLevelPerceptionModel(PreTrainedModel):
    """Adapter exposing upstream L4P_VideoMAE through 4D-RGPT's interface."""

    config_class = LowLevelPerceptionConfig

    def __init__(self, low_level_perception_cfg: LowLevelPerceptionConfig, config: PretrainedConfig):
        super().__init__(config)
        self.img_size = 224
        self.window_size = (16, self.img_size, self.img_size)
        self.window_stride_T = 8
        self.model_dtype = eval(config.model_dtype)
        self.hooks_idx = list(HOOKS_IDX)

        # Default intrinsics, shape (1, 4, 4, T). Used when caller doesn't
        # supply real ones; train.py rebuilds these once num_video_frames is
        # finalized so rays_to_cameras sees matching T.
        intrinsics_b44 = default_intrinsics(self.img_size, self.img_size)
        self.intrinsics_b44t = einops.repeat(
            intrinsics_b44, "b h w -> b h w t", t=config.num_video_frames
        )
        self.intrinsics = normalize_intrinsics(
            self.intrinsics_b44t, self.img_size, self.img_size
        )

        # Upstream raw nn.Module (no Lightning wrapper).
        self.l4p_model = L4P_VideoMAE(
            task_heads=_build_task_heads(self.hooks_idx),
            video_encoder_ckpt_path=None,
            window_size=self.window_size,
            window_stride_T=self.window_stride_T,
            freeze_video_encoder=True,
            always_use_windowed_version=True,
            joint_alignment=False,
        )

        for p in self.parameters():
            p.requires_grad = False

    # ----- proxy properties so callers can keep using `encoder_l4p.tasks` etc. -----
    @property
    def task_heads(self) -> nn.ModuleDict:
        return self.l4p_model.task_heads

    @property
    def video_encoder(self):
        return self.l4p_model.video_encoder

    @property
    def tasks(self):
        return self.l4p_model.task_heads.keys()

    # ----- forward / decode -------------------------------------------------------
    def _normalize(self, video: torch.Tensor) -> torch.Tensor:
        return (video - L4P_INPUT_MEAN.to(video)) / L4P_INPUT_STD.to(video)

    def _encode_windows(self, rgb_b3thw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sliding-window encode → stacked features + matching time_strides.

        Returns:
            feats: (num_windows, num_blocks=41, B=1, num_tokens=2048, embed_dim=1408)
            time_strides: 1-D LongTensor of window start indices into the T axis.
        """
        T = rgb_b3thw.shape[2]
        time_strides = torch.arange(0, T - self.window_size[0] + 1, self.window_stride_T)
        intrinsics_norm = self.intrinsics.to(rgb_b3thw)

        per_window = []
        for curr in time_strides:
            end = int(curr) + self.window_size[0]
            feats_list = self.video_encoder(
                rgb_b3thw[:, :, int(curr):end],
                intrinsics_norm[:, :, :, int(curr):end],
                None,  # no extrinsics
            )
            per_window.append(torch.stack(feats_list, dim=0))
        return torch.stack(per_window, dim=0), time_strides

    def forward(self, videos: List[torch.Tensor]) -> List[LowLevelPerceptionOutput]:
        """Encode each video and run every task head. Caller wraps in no_grad."""
        outputs: List[LowLevelPerceptionOutput] = []
        intrinsics_b44t = self.intrinsics_b44t

        for video in videos:
            rgb_b3thw = self._normalize(video).unsqueeze(0)
            feats, time_strides = self._encode_windows(rgb_b3thw)

            output = LowLevelPerceptionOutput(feats_enc=feats)
            intr = intrinsics_b44t.to(rgb_b3thw)
            for task in self.tasks:
                task_out = self.task_heads[task].forward_windowed(
                    enc_features_bpc_2dlist=feats,
                    img_info=self.window_size,
                    time_strides=time_strides,
                    intrinsics_b44t=intr,
                )
                setattr(output, f"pred_{task}", _primary_tensor(task_out))
            outputs.append(output)
        return outputs

    def decode(
        self,
        l4p_projs: list[torch.Tensor],
        num_frames: int = 16,
    ) -> List[LowLevelPerceptionOutput]:
        """Decode pre-projected (student) features through every task head.

        Used by the explicit-distillation path: the student MLP projects LLM
        hidden states into the encoder's feature space; this method runs the
        frozen task heads on those projections to produce L_ED targets.
        """
        time_strides = torch.arange(
            0, num_frames - self.window_size[0] + 1, self.window_stride_T
        )
        outputs: List[LowLevelPerceptionOutput] = []
        for l4p_proj in l4p_projs:
            output = LowLevelPerceptionOutput()
            for task in self.tasks:
                task_out = self.task_heads[task].forward_windowed(
                    enc_features_bpc_2dlist=l4p_proj,
                    img_info=self.window_size,
                    time_strides=time_strides,
                    intrinsics_b44t=self.intrinsics_b44t,
                )
                setattr(output, f"pred_{task}", _primary_tensor(task_out))
            outputs.append(output)
        return outputs
