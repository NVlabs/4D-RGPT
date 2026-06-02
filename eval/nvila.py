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

import llava
from llava import conversation as conversation_lib

from typing import Optional, Union
from pathlib import Path
from functools import partial

from data import get_datamodule
from utils import distributed as dist
from utils.logging import logger
from eval.utils.report import Report


def nvila_response(
    model,
    generation_config,
    text: str,
    video_path: Optional[Union[list[str], str]] = None,
    image_path: Optional[Union[list[str], str]] = None,
) -> str:
    content = []
    if image_path is not None:
        paths = image_path if isinstance(image_path, list) else [image_path]
        for path in paths:
            content.append(llava.Image(path))
    if video_path is not None:
        # Pass the raw video path — llava's internal _load_video samples frames
        # via model.config.num_video_frames. The old `sample_video_frames` path
        # pre-extracted PNGs and returned a list, but llava.Video expects a
        # single str (video file OR frame directory), so the list broke
        # downstream `os.path.isdir(video_path)` checks.
        content.append(llava.Video(video_path))
    content.append(text)

    response = model.generate_content(content, generation_config=generation_config)
    return response


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str)
    parser.add_argument("--lora-path", type=str, default=None)
    parser.add_argument("--benchmarks", type=str, default="r4d")
    parser.add_argument("--conv-mode", type=str, default="auto")
    parser.add_argument("--num-video-frames", type=int, default=-1)
    parser.add_argument("--generation-config", type=json.loads)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="If true, only evaluate the result.json file without running inference.")
    args = parser.parse_args()

    dist.init()
    devices = range(dist.local_rank(), torch.cuda.device_count(), dist.local_size())
    torch.cuda.set_device(devices[0])

    os.makedirs(args.output_dir, exist_ok=True)
    output_dir = Path(args.output_dir) / args.benchmarks
    output_dir.mkdir(parents=True, exist_ok=True)
    result_json = output_dir / "result.json"

    datamodule = get_datamodule(args.benchmarks)
    if not args.eval_only:
        conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_mode].copy()

        if dist.is_main():
            logger.info(f"[Init] Loading model: {args.model_path}")

        if args.lora_path is None:
            model = llava.load(args.model_path, model_base=args.model_base, devices=devices)
        else:
            model = llava.load(args.lora_path, model_base=args.model_path, devices=devices)

        if args.num_video_frames > 0:
            model.config.num_video_frames = args.num_video_frames

        generation_config = model.default_generation_config
        if args.generation_config is not None:
            generation_config.update(**args.generation_config)

        model = model.to("cuda")

        if dist.is_main():
            logger.info(f"[Init] Model ready. num_video_frames={model.config.num_video_frames}")

        datamodule.load()
        instances = datamodule.test_data[dist.rank() :: dist.size()]
        if dist.is_main():
            logger.info(f"[Data] {len(datamodule.test_data)} samples loaded.")
            logger.info(f"[Data] Each GPU will run {len(instances)} samples.")

        model_response = partial(nvila_response, model, generation_config)

        outputs = datamodule.inference(
            instances,
            model_response,
            save_every=True,
            output_dir=output_dir.as_posix(),
        )

        if dist.size() > 1:
            outputs = dist.gather(outputs, dst=0)
            if not dist.is_main():
                return
            outputs = list(itertools.chain(*outputs))
        with open(result_json, "w", encoding="utf-8") as f:
            json.dump(outputs, f, indent=4, ensure_ascii=False)

    scores = datamodule.eval(result_json.as_posix())

    report = Report(
        title=f"{args.benchmarks} Evaluation Report",
        headers=list(scores.keys()),
        floatfmt=".1%",
    )
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
