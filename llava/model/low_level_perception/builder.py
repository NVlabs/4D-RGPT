# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Builders for the L4P teacher and the student-side distillation projector.

import os
from pathlib import Path
from typing import Any, Dict

import einops
import torch
import torch.nn as nn
import torchvision.transforms as T
from transformers import PretrainedConfig, PreTrainedModel

from .base import LowLevelPerceptionConfig, LowLevelPerceptionModel
from llava.utils.logging import logger


# Default location for the upstream L4P checkpoint. The canonical filename
# matches `third_party/L4P/weights/download.sh` (the gdrive blob is published
# under this name). Override per call by passing `ckpt_path=`.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CKPT = (
    _REPO_ROOT / "third_party" / "L4P" / "weights"
    / "l4p_depth_flow_2d3dtrack_camray_dynseg_v1.ckpt"
)


def build_l4p_encoder_decoder(
    model_type_or_path: str,
    config: PretrainedConfig,
    ckpt_path: str | os.PathLike | None = None,
) -> LowLevelPerceptionModel | None:
    """Build (and weight-load) the L4P teacher.

    If `model_type_or_path` is a resumable directory, defer to `from_pretrained`.
    Otherwise instantiate a fresh adapter and seed it from the upstream ckpt.
    """
    if model_type_or_path is None:
        return None

    if config.resume_path and os.path.exists(model_type_or_path):
        return LowLevelPerceptionModel.from_pretrained(model_type_or_path, config)

    model = LowLevelPerceptionModel(LowLevelPerceptionConfig("l4p"), config)
    _load_upstream_ckpt(model, ckpt_path or _DEFAULT_CKPT)
    return model


def _load_upstream_ckpt(model: LowLevelPerceptionModel, ckpt_path: os.PathLike) -> None:
    """Load an upstream NVlabs/L4P checkpoint into our adapter.

    Upstream saves a LightningModule (`L4PLitModule`) whose state dict keys are
    `l4p_model.video_encoder.<…>` and `l4p_model.task_heads.<task>.<…>`. Our
    adapter wraps the raw `L4P_VideoMAE` under the same attribute name
    (`self.l4p_model`), so the keys map directly without renaming. We use
    `strict=False` because (a) we omit upstream's `track_2d` head and (b) our
    adapter adds buffers (`intrinsics`, `intrinsics_b44t`) not in the ckpt.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"L4P checkpoint not found at {ckpt_path}. Either pass `ckpt_path=` "
            "or run `bash third_party/L4P/weights/download.sh`."
        )
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict: Dict[str, Any] = blob["state_dict"]

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    # `missing` is expected to contain our adapter-owned buffers (intrinsics*)
    # and nothing else. `unexpected` is expected to contain upstream's
    # track_2d head + possibly aligner state. Surface anything else loudly.
    expected_missing = {"intrinsics", "intrinsics_b44t"}
    surprising_missing = [k for k in missing if not any(s in k for s in expected_missing)]
    if surprising_missing:
        logger.warning(
            f"L4P ckpt missing {len(surprising_missing)} expected keys, e.g.: "
            f"{surprising_missing[:5]}"
        )
    surprising_unexpected = [
        k for k in unexpected if "track_2d" not in k and "aligner" not in k
    ]
    if surprising_unexpected:
        logger.warning(
            f"L4P ckpt has {len(surprising_unexpected)} unexpected keys, e.g.: "
            f"{surprising_unexpected[:5]}"
        )


# ============================================================================
# Distillation projector (4D-RGPT specific — not part of upstream L4P).
# ============================================================================

class DistillationConfig(PretrainedConfig):
    model_type = "distillation"


class DistillationModel(PreTrainedModel):
    """MLP that projects LLM hidden states into the L4P encoder feature space.

    Per-hook linear projection followed by a spatial resize to (16, 16). The
    output is consumed by `LowLevelPerceptionModel.decode()` to produce the
    student-side targets for L_ED.
    """

    config_class = DistillationConfig

    def __init__(self, distillation_config: DistillationConfig, config: PretrainedConfig):
        super().__init__(config)

        in_features = config.llm_cfg["hidden_size"]
        out_features = 1408
        output_size = 16
        self.hooks_idx = [14, 21, 28, 36]

        self.proj = nn.ModuleList([
            nn.ModuleList([nn.Linear(in_features, out_features)])
            for _ in self.hooks_idx
        ])
        for hook_layers in self.proj:
            for layer in hook_layers:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, mean=0, std=1e-1)
                    nn.init.constant_(layer.bias, 0)

        self.resize = T.Resize([output_size, output_size])

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        hidden_state: (batch_size, num_frames, in_features, height, width)
        Returns:     (batch_size, num_hooks, num_frames//2, out_features, 16, 16)
        """
        x = hidden_state[:, ::2]  # temporal stride 2 to match L4P's tubelet_size=2
        b, n, *_ = x.shape
        x = einops.rearrange(x, "b n c h w -> (b n) c h w")
        x = self.resize(x)
        x = einops.rearrange(x, "bn c h w -> bn h w c")
        y = [self._apply_hook_layers(layers, x) for layers in self.proj]
        x = torch.stack(y, dim=1)  # (b*n, num_hooks, h, w, out_features)
        return einops.rearrange(x, "(b n) l h w c -> b l n c h w", b=b, n=n)

    @staticmethod
    def _apply_hook_layers(layers: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
        for layer in layers:
            x = layer(x)
        return x


def build_distillation(model_type_or_path: str, config: PretrainedConfig) -> DistillationModel | None:
    if model_type_or_path is None:
        return None
    if config.resume_path and os.path.exists(model_type_or_path):
        return DistillationModel.from_pretrained(model_type_or_path, config)
    return DistillationModel(DistillationConfig(), config)
