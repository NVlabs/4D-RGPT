import os

import torch
from transformers import PretrainedConfig, PreTrainedModel

from .base_pe import PE, PEConfig


def build_pe(model_type_or_path: str, config: PretrainedConfig) -> PE | None:
    if model_type_or_path is None:
        return None

    if config.resume_path and os.path.exists(model_type_or_path):
        return PE.from_pretrained(model_type_or_path, config, torch_dtype=eval(config.model_dtype))
    pe_cfg = PEConfig(model_type_or_path)
    return PE(pe_cfg, config).to(eval(config.model_dtype))
