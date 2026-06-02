# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import json
import torch
import itertools
import argparse

from typing import Optional, Union
from tqdm.auto import tqdm
from pathlib import Path
from functools import partial
from transformers import AutoModel, AutoProcessor
from qwen_vl_utils import process_vision_info

from data import get_datamodule
from data.utils.data_helper import sample_video_frames, get_video_length
from utils import distributed as dist
from utils.logging import logger
from eval.utils.report import Report

def qwen_response(
    model,
    processor,
    text: str,
    video_path: Optional[Union[list[str], str]] = None,
    image_path: Optional[Union[list[str], str]] = None,
) -> str:
    content = []
    if image_path is not None:
        if isinstance(image_path, list):
            for path in image_path:
                content.append({
                    "type": "image",
                    "image": path,
                })
        else:
            content.append({
                "type": "image",
                "image": image_path,
            })
    if video_path is not None:
        frame_paths, fps = sample_video_frames(video_path, n_frames=16)
        content.append({
            "type": "video",
            "video": frame_paths,
            "fps": fps
        })
    video_sec = get_video_length(video_path)
    prompt = f"These images are sampled from a video of {video_sec} seconds." + "\n"
    prompt += text

    content.append({
        "type": "text",
        "text": prompt,
    })
    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    chat = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[chat],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return response

def get_qwen_model(model_path: str) -> tuple[torch.nn.Module, AutoProcessor]:
    if "qwen3" in model_path.lower() or "cosmos" in model_path.lower():
        from transformers import Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, dtype="auto", device_map="auto", attention_implementation="flash_attention_2"
        )
        processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
    else:
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="auto"
        )
        try:
            processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        except ValueError:
            print("Failed to load processor from model path, using Qwen2.5-VL-Instruct instead.")
            processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=True)
    return model, processor

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    # parser.add_argument("--model-base", type=str, default=None)
    # parser.add_argument("--lora-path", type=str, default=None)
    parser.add_argument("--benchmarks", type=str, default="vstibench")
    # parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    # parser.add_argument("--conv-mode", type=str, required=True)
    # parser.add_argument("--prompt-mode", type=str, default="direct")
    # parser.add_argument("--num-video-frames", type=int, default=-1)
    # parser.add_argument("--max-tiles", type=int)
    # parser.add_argument("--generation-config", type=json.loads)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="If true, only evaluate the result.json file without running inference.")
    args = parser.parse_args()

    dist.init()
    devices = range(dist.local_rank(), torch.cuda.device_count(), dist.local_size())
    torch.cuda.set_device(devices[0])

    model, processor = get_qwen_model(args.model_path)
    # model = model.to("cuda")

    os.makedirs(args.output_dir, exist_ok=True)
    output_dir = Path(args.output_dir) / args.benchmarks
    output_dir.mkdir(parents=True, exist_ok=True)
    result_json = output_dir / "result.json"

    datamodule = get_datamodule(args.benchmarks)
    if not args.eval_only:

        if dist.is_main():
            logger.info("[Init] Model ready.")

        datamodule.load()
        instances = datamodule.test_data[dist.rank() :: dist.size()]
        if dist.is_main():
            logger.info(f"[Data] {len(datamodule.test_data)} samples loaded.")
            logger.info(f"[Data] Each GPU will run {len(instances)} samples.")

        model_response = partial(
            qwen_response,
            model,
            processor,
        )

        outputs = datamodule.inference(
            instances,
            model_response,
            save_every=True,
            output_dir=output_dir.as_posix()
        )

        if dist.size() > 1:
            outputs = dist.gather(outputs, dst=0)
            if not dist.is_main():
                return
            outputs = list(itertools.chain(*outputs))
        with open(result_json, "w", encoding="utf-8") as f:
            json.dump(outputs, f, indent=4, ensure_ascii=False)

    scores = datamodule.eval(result_json.as_posix())

    report = Report(title=f"{args.benchmarks} Evaluation Report", headers=list(scores.keys()), floatfmt=".1%")
    report.add_row(list(scores.values()))
    report_txt = str(report)

    with open((output_dir / "report.txt").as_posix(), "a", encoding="utf-8") as f:
        f.write(str(args) + "\n")
        f.write(report_txt)
    logger.info("\n" + report_txt)

    if dist.is_main():
        logger.info(f"[Save] {len(outputs)} results → {result_json}")

if __name__ == '__main__':
    main()
