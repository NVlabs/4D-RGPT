import re
import einops
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

from llava.utils.logging import logger


class RegionExtractorConfig(PretrainedConfig):
    model_type = "region_extractor"

    def __init__(self, region_extractor_type: str = None, **kwargs):
        super().__init__()
        self.region_extractor_type = region_extractor_type


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MaskCatAndPool(nn.Module):
    def __init__(self, mask_threshold=0.5):
        super().__init__()
        self.mask_threshold = mask_threshold

    def forward(self, x, mask_list, block_lengths, return_list=False, return_mask=False):

        batch_size = x.size(0)
        if mask_list is None:
            mask_list = [None] * batch_size
        if len(mask_list) != batch_size:
            raise ValueError("Length of mask_list must match batch size.")
        if int(np.sum(block_lengths)) != batch_size:
            raise ValueError("Sum of block_lengths must equal the batch size.")
        # print("mask_list length: ", len(mask_list))
        fused_embeds = []  # fused output per block
        start_idx = 0
        for length in block_lengths:
            if length == 0:
                fused_embeds.append(None)
                continue

            # For each block, we accumulate features and masks from all samples
            features_cat = []  # to hold [L, C] tensors
            masks_cat = []  # to hold corresponding masks [M, L]
            # print(length)
            for i in range(start_idx, start_idx + length):
                # Here, we assume x[i] is of shape [L, C] and mask_list[i] is [M, IH, IW]
                mask = mask_list[i]
                if mask is None:
                    continue  # TODO: check this, if so should be none for all samples in block length
                L = x.size(1)
                mask_hw = mask.size(-1) * mask.size(-2)
                scale_factor = (L / mask_hw) ** 0.5

                # Resize mask as before:
                mask_resized = nn.functional.interpolate(
                    mask.detach().float()[None, ...], scale_factor=scale_factor, mode="bilinear", align_corners=False
                ).to(x.dtype)[0]

                # Flatten the spatial dimensions: shape becomes [M, L].
                mask_flat = mask_resized.flatten(start_dim=1)

                features_cat.append(x[i])  # shape: [L, C]
                masks_cat.append(mask_flat)  # shape: [M, L]

            if len(features_cat) == 0:
                fused_embeds.append(None)
            else:
                # Concatenate along the spatial dimension.
                features_cat = torch.cat(features_cat, dim=0)  # shape: [total_L, C]
                masks_cat = torch.cat(masks_cat, dim=1)  # shape: [M, total_L]
                # Compute normalization factor (for each mask channel M)
                # print("masks_cat shape: ", masks_cat.shape)
                denorm = masks_cat.sum(dim=-1, keepdim=True) + 1e-8  # shape: [M, 1]
                # Weighted pooling over the concatenated spatial dimension.
                fused_tensor = torch.einsum("lc,ml->mc", features_cat, masks_cat / denorm)
                fused_embeds.append(fused_tensor)
            start_idx += length

        # If returning a single tensor, ignore any fused blocks that are None.
        valid_fused = [embed for embed in fused_embeds if embed is not None]

        if return_list:
            return fused_embeds
        else:
            if len(valid_fused) > 0:
                fused_cat = torch.cat(valid_fused, dim=0)
                return fused_cat
            else:
                return None


def get_feature_refinement_module(vision_hidden_size, feature_refinement_type="deconv2x"):
    deconv_match = re.match(r"^deconv(\d+)x$", feature_refinement_type)
    if deconv_match:
        deconv_depth = int(deconv_match.group(1))
        modules = []
        for i in range(deconv_depth - 1):
            modules.append(nn.ConvTranspose2d(vision_hidden_size, vision_hidden_size, kernel_size=2, stride=2))
            modules.append(LayerNorm2d(vision_hidden_size))
            modules.append(nn.GELU())
        modules.append(nn.ConvTranspose2d(vision_hidden_size, vision_hidden_size, kernel_size=2, stride=2))
        modules.append(nn.GELU())

        return nn.Sequential(*modules)

    raise ValueError(f"Unknown feature refinement type: {feature_refinement_type}")


class RegionExtractor(PreTrainedModel):
    config_class = RegionExtractorConfig

    def __init__(self, region_extractor_cfg: RegionExtractorConfig, config: PretrainedConfig):
        super().__init__(region_extractor_cfg)
        region_extractor_type = region_extractor_cfg.region_extractor_type

        if region_extractor_type == "regiongpt":
            self.mask_pooling = MaskCatAndPool()
            self.feature_refinement_module = get_feature_refinement_module(config.mm_hidden_size)
            # TODO: hardcoded pooling size here, should be inside cfg
            self.ada_pooling = nn.AdaptiveAvgPool2d(27)
            self.projector = nn.Linear(config.mm_hidden_size, config.hidden_size)
        else:
            raise NotImplementedError(f"{region_extractor_type} not implemented")

    def feature_refinement(self, tower_features: torch.Tensor) -> torch.Tensor:
        logger.debug(tower_features)
        HW = tower_features.shape[1]
        tower_features = einops.rearrange(tower_features, "N (H W) C -> N C H W", H=int(HW**0.5))
        hres_tower_features = self.feature_refinement_module(tower_features)
        # region feature branch
        hres_tower_features_flatten = einops.rearrange(hres_tower_features, "N C H W -> N (H W) C")

        # global feature branch
        ada_image_feature = self.ada_pooling(hres_tower_features)
        lres_tower_features_flatten = einops.rearrange(ada_image_feature, "N C H W -> N (H W) C")
        return hres_tower_features_flatten, lres_tower_features_flatten

    def extract_region_features(self, hres_tower_features, masks, block_lengths, connector):
        # assume is already flattened -> 'N (H W) C'
        if self.config.region_extractor_type == "regiongpt":
            _mask_embeds = self.mask_pooling(hres_tower_features, masks, block_lengths, return_list=False)
            if _mask_embeds is not None:
                mask_embeds = connector(_mask_embeds)
            else:
                mask_embeds = None
        else:
            raise NotImplementedError(f"{self.config.region_extractor_type} not implemented")
        return mask_embeds

    def forward(self, image_features, masks, block_lengths, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        logger.debug(image_features)
        logger.debug(masks)
        logger.debug(block_lengths)
        mask_embeds = self.extract_region_features(image_features, masks, block_lengths, self.projector)

        # generate dummy mask feature
        dummy_image_feat = image_features[0].mean(0, keepdim=True)
        dummy_mask_embeds = self.projector(dummy_image_feat)

        return mask_embeds, dummy_mask_embeds
