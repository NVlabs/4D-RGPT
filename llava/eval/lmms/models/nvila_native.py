import copy
import json
import os
from typing import List, Tuple

import accelerate
import torch
import transformers
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from tqdm import tqdm
from transformers import GenerationConfig

import llava
from llava import conversation as conversation_lib
from llava.media import Image, Video
from llava.utils import distributed as dist
from llava.utils import io


# ---- v5 compat: patch LlavaMetaForCausalLM.default_generation_config ----------
# Sibling to llava/eval/main_without_split_tv5.py's monkey-patch, applied at
# import time so generate_until's `self.model.default_generation_config`
# (line below) works under transformers 5.
#
# Why: transformers>=5 only auto-sets self.generation_config when can_generate()
# is True. LlavaLlamaModel doesn't inherit GenerationMixin, so the attribute
# is unset, and the AttributeError raised inside the @property gets silently
# re-cast by Python as "no such attribute on the host class" — surfacing as
# `'LlavaLlamaModel' object has no attribute 'default_generation_config'`.
#
# Guarded on transformers major version so v4 (transformers 4.46) stays
# byte-untouched. The patched property is functionally identical to the v4
# original except it uses getattr() so missing generation_config doesn't raise.
if int(transformers.__version__.split(".")[0]) >= 5:
    from llava.model.llava_arch import LlavaMetaForCausalLM as _LlavaMetaForCausalLM

    @property
    def _tv5_default_generation_config(self) -> GenerationConfig:
        base_config = getattr(self, "generation_config", None) or GenerationConfig()
        generation_config = copy.deepcopy(base_config)
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must have an EOS token")
        if generation_config.max_length == GenerationConfig().max_length:
            generation_config.max_length = self.tokenizer.model_max_length
        if generation_config.pad_token_id is None:
            generation_config.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        if generation_config.bos_token_id is None:
            generation_config.bos_token_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        if generation_config.eos_token_id is None:
            generation_config.eos_token_id = self.tokenizer.stop_token_ids
        return generation_config

    _LlavaMetaForCausalLM.default_generation_config = _tv5_default_generation_config
# ---- end v5 compat -----------------------------------------------------------


def _resolve_lora_base(model_path: str) -> str | None:
    """If `model_path` is a LoRA checkpoint, return the base model path from its
    config.json. Otherwise return None.

    Detection: presence of `adapter_config.json` (the PEFT artifact). We read
    `model_name_or_path` (the original base, e.g. "Efficient-Large-Model/...")
    falling back to `resume_path`. Without this, eval of in-repo LoRA runs
    fails because the eval loader's full-model branch expects `<ckpt>/llm/`
    which LoRA saves don't produce.
    """
    if not os.path.exists(os.path.join(model_path, "adapter_config.json")):
        return None
    cfg_path = os.path.join(model_path, "config.json")
    if not os.path.exists(cfg_path):
        return None
    with open(cfg_path) as f:
        cfg = json.load(f)
    return cfg.get("model_name_or_path") or cfg.get("resume_path")


@register_model("nvila_native")
class NVILANative(lmms):
    def __init__(
        self,
        model_path: str,
        conv_mode: str,
        model_base: str = None,
        num_video_frames: int = 16,
        max_tiles: int = 12,
        batch_size: int = 1,
    ) -> None:
        super().__init__()
        # lmms-eval parses --batch_size as str (default "1"); coerce before compare.
        assert int(batch_size) == 1, "VILA only supports batch size of 1 at the moment."

        devices = range(dist.local_rank(), torch.cuda.device_count(), dist.local_size())
        torch.cuda.set_device(devices[0])

        # Auto-fill model_base for LoRA checkpoints if the caller didn't.
        if model_base is None:
            auto_base = _resolve_lora_base(model_path)
            if auto_base is not None:
                print(f"[nvila_native] LoRA detected; auto model_base={auto_base}", flush=True)
                model_base = auto_base

        self.model = llava.load(model_path, model_base=model_base, devices=devices)
        self.model.config.num_video_frames = num_video_frames
        context_length = num_video_frames * 512

        self.model.config.min_tiles = 1
        self.model.config.max_tiles = max_tiles
        self.model.llm.config.min_tiles = 1
        self.model.llm.config.max_tiles = max_tiles
        self.model.config.image_aspect_ratio = "resize"

        if max_tiles > 12:
            context_length = max(context_length, int(max_tiles / 12.0 * 4096))

        context_length = max(
            getattr(self.model.llm.config, "tokenizer_model_max_length", context_length), context_length
        )

        self.model.config.model_max_length = context_length
        self.model.config.tokenizer_model_max_length = context_length
        self.model.llm.config.model_max_length = context_length
        self.model.llm.config.tokenizer_model_max_length = context_length
        self.model.tokenizer.model_max_length = context_length

        conversation_lib.default_conversation = conversation_lib.conv_templates[conv_mode].copy()

        self.accelerator = accelerate.Accelerator()
        self.device = torch.device(f"cuda:{devices[0]}")
        self._world_size = dist.size()
        self._rank = dist.rank()

    def generate_until(self, requests: List[Instance]) -> List[str]:
        responses = []
        for request in tqdm(requests, disable=not dist.is_main()):
            prompt, generation_kwargs, doc_to_visual, doc_id, task, split = request.args
            doc = self.task_dict[task][split][doc_id]

            # Generate multimodal prompt
            medias = []
            for media in doc_to_visual(doc):
                if isinstance(media, str):
                    lower = media.lower()
                    if lower.endswith((".mp4", ".mkv", ".webm")):
                        media = Video(media)
                    elif lower.endswith((".png", ".jpg", ".jpeg", ".webp")):
                        media = Image(media)
                    else:
                        raise NotImplementedError(f"Unsupported media type: {media}")
                medias.append(media)
            prompt = medias + [prompt]

            # Override generation config. default_generation_config sets
            # max_length from tokenizer.model_max_length; if the task supplies
            # max_new_tokens, clear max_length so transformers doesn't warn
            # "Both max_new_tokens and max_length seem to have been set" on
            # every generate() call.
            generation_config = self.model.default_generation_config
            if "max_new_tokens" in generation_kwargs:
                generation_config.max_length = None
            generation_config.update(**generation_kwargs)

            # Generate and cache response
            cache_path = None
            if "CACHE_DIR" in os.environ:
                cache_path = os.path.join(os.environ["CACHE_DIR"], f"{task}_{split}_{doc_id}.txt")

            if cache_path is not None and os.path.exists(cache_path):
                response = io.load(cache_path)
            else:
                response = self.model.generate_content(prompt, generation_config=generation_config)
                if cache_path is not None:
                    io.save(cache_path, response)
            responses.append(response)
        return responses

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError
