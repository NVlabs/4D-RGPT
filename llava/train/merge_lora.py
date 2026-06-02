# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Merge a LoRA-trained NVILA checkpoint into a full-FT-shaped checkpoint so
# downstream loaders (llava.load, nvila_native lmms-eval backend) can pick it
# up without needing model_base or PEFT awareness.
#
# Usage:
#   python -m llava.train.merge_lora \
#       --lora_dir   runs/train/NVILA-Lite-8B-4D-lora/model \
#       --model_base Efficient-Large-Model/NVILA-Lite-8B \
#       --output_dir runs/train/NVILA-Lite-8B-4D-lora/merged
#
# Input  (lora_dir):  adapter_model.safetensors + non_lora_trainables.bin
# Output (output_dir): standard HF checkpoint — full LLM weights with LoRA
#                     deltas baked in, plus projector / region extractor /
#                     L4P heads from non_lora_trainables. Loadable via
#                     `llava.load(output_dir)` with no model_base.
#
# The actual load+merge happens inside `llava.load` → `load_pretrained_model`
# → `PeftModel.from_pretrained` → `merge_and_unload`, which already exists in
# llava/model/builder.py. We just re-save the merged in-memory model.
import argparse
import os

import torch

from llava import load


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a LoRA-trained NVILA checkpoint.")
    parser.add_argument("--lora_dir", required=True,
                        help="Path to the trained LoRA dir (contains adapter_model.safetensors).")
    parser.add_argument("--model_base", required=True,
                        help="HF id or local path of the base NVILA checkpoint LoRA was trained on.")
    parser.add_argument("--output_dir", required=True,
                        help="Where to save the merged checkpoint. Must NOT contain the substring 'lora'.")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
                        help="Dtype of the merged weights. Default bf16 to match training.")
    args = parser.parse_args()

    # builder.py:62 detects LoRA via `"lora" in model_name.lower()`. If the
    # output dir contains "lora", future loads will try to PEFT-rehydrate the
    # (already-merged) weights and fail. Guard early.
    if "lora" in os.path.basename(args.output_dir.rstrip("/")).lower():
        raise ValueError(
            f"output_dir basename ({args.output_dir!r}) contains 'lora'. "
            "Pick a different name (e.g. '<run>/merged') — builder.py treats "
            "any path containing 'lora' as an unmerged LoRA checkpoint."
        )

    # llava.load(lora_dir, model_base=...) routes through builder.py's
    # LoRA branch: loads base, applies non_lora_trainables.bin, attaches
    # PeftModel, calls merge_and_unload. Returns a regular LlavaLlamaModel.
    dtype = getattr(torch, args.dtype)
    print(f"Loading + merging {args.lora_dir} onto {args.model_base} ...")
    model = load(args.lora_dir, model_base=args.model_base, torch_dtype=dtype)

    print(f"Saving merged checkpoint to {args.output_dir} ...")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    # NVILA attaches the tokenizer to the model (`model.tokenizer`, see
    # builder.py:73). Save it alongside so the merged dir is self-contained.
    if hasattr(model, "tokenizer") and model.tokenizer is not None:
        model.tokenizer.save_pretrained(args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
