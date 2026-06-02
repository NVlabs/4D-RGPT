# This file is modified from https://github.com/haotian-liu/LLaVA/
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


import os
import warnings

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PretrainedConfig

from llava.model import LlavaLlamaModel
from llava.model.utils import is_mm_model
# TODO
from llava.utils.logging import logger



def load_pretrained_model(
    model_path,
    model_name,
    model_base=None,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
    **kwargs,
):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs["device_map"] = {"": device}

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.float16
        # kwargs["torch_dtype"] = torch.bfloat16

    if is_mm_model(model_path):
        # Load LLaVA model
        # LoRA detection: prefer the presence of `adapter_config.json` (the
        # PEFT artifact) over a substring match on `model_name`. Many of our
        # in-repo runs don't put "lora" in the run name (e.g.
        # NVILA-Lite-8B-4D-dev_v2), but their checkpoints ARE LoRA — the
        # name-based gate misses them and the full-model branch then crashes
        # in get_model_config when `<ckpt>/llm` doesn't exist.
        is_lora_ckpt = os.path.exists(os.path.join(model_path, "adapter_config.json"))
        if is_lora_ckpt and model_base is None:
            warnings.warn(
                f"adapter_config.json found at {model_path} but no `model_base` provided. "
                "LoRA checkpoints need the base model to load against."
            )
        if is_lora_ckpt and model_base is not None:
            print("Loading LLaVA from base model...")
            # Use the LoRA ckpt's own config.json (preserves training-time
            # architecture: region_extractor_cfg, pe_cfg, distillation_cfg).
            # Override _name_or_path/resume_path to model_base so the
            # per-submodule builders snapshot_download base weights from the
            # hub — by default HF sets _name_or_path to the LoRA ckpt dir
            # (which has no llm/, vision_tower/, mm_projector/ subdirs).
            # Constructing via __init__ (not from_pretrained) routes through
            # init_vlm and the per-subdir loaders, required because
            # NVILA-Lite-8B has a composite layout (no top-level model.safetensors).
            config = AutoConfig.from_pretrained(model_path)
            config._name_or_path = model_base
            config.resume_path = model_base
            prepare_config_for_eval(config, kwargs)
            model = LlavaLlamaModel(config=config, low_cpu_mem_usage=True, **kwargs)
            tokenizer = model.tokenizer
            token_num, tokem_dim = model.llm.lm_head.out_features, model.llm.lm_head.in_features
            if model.llm.lm_head.weight.shape[0] != token_num:
                model.llm.lm_head.weight = torch.nn.Parameter(
                    torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype)
                )
                model.llm.embed_tokens.weight = torch.nn.Parameter(
                    torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype)
                )

            print("Loading additional LLaVA weights...")
            if os.path.exists(os.path.join(model_path, "non_lora_trainables.bin")):
                non_lora_trainables = torch.load(
                    os.path.join(model_path, "non_lora_trainables.bin"),
                    map_location="cpu",
                )
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download

                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(repo_id=repo_id, filename=filename, subfolder=subfolder)
                    return torch.load(cache_file, map_location="cpu")

                non_lora_trainables = load_from_hf(model_path, "non_lora_trainables.bin")
            non_lora_trainables = {
                (k[11:] if k.startswith("base_model.") else k): v for k, v in non_lora_trainables.items()
            }
            if any(k.startswith("model.model.") for k in non_lora_trainables):
                non_lora_trainables = {
                    (k[6:] if k.startswith("model.") else k): v for k, v in non_lora_trainables.items()
                }
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel

            print("Loading LoRA weights...")
            model = PeftModel.from_pretrained(model, model_path)
            print("Merging LoRA weights...")
            model = model.merge_and_unload()
            print("Model is loaded...")
        else:
            config = AutoConfig.from_pretrained(model_path)
            config.resume_path = model_path
            prepare_config_for_eval(config, kwargs)
            model = LlavaLlamaModel(config=config, low_cpu_mem_usage=True, **kwargs)
            tokenizer = model.tokenizer
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel

            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print("Convert to FP16...")
            model.to(torch.float16)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, legacy=False)
            model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    model.eval()
    image_processor = None
    if is_mm_model(model_path):
        model.resize_token_embeddings(len(tokenizer))
        vision_tower = model.get_vision_tower()
        vision_tower.to(device=device, dtype=torch.float16)
        mm_projector = model.get_mm_projector()
        mm_projector.to(device=device, dtype=torch.float16)
        image_processor = vision_tower.image_processor

        # NOTE: 4D_RVILA
        pe = model.get_pe()
        if pe:
            pe.to(device=device, dtype=torch.float16)
        region_extractor = model.get_region_extractor()
        if region_extractor:
            region_extractor.to(device=device, dtype=torch.float16)


    if hasattr(model.llm.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len


def prepare_config_for_eval(config: PretrainedConfig, kwargs: dict):
    try:
        # compatible with deprecated config convention
        if getattr(config, "vision_tower_cfg", None) is None:
            config.vision_tower_cfg = config.mm_vision_tower
    except AttributeError:
        raise ValueError(f"Invalid configuration! Cannot find vision_tower in config:\n{config}")

    config.model_dtype = kwargs.pop("torch_dtype").__str__()
