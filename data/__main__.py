# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import glob
import json
import torch
import itertools
import argparse
import einops
from pathlib import Path

from .register import get_datamodule
# from transformers import PretrainedConfig
# from l4p.model import get_model, encode_l4p, decode_l4p


# def batch_to_cuda(batch_dict: dict) -> dict:
#     for k in batch_dict:
#         if hasattr(batch_dict[k], 'device'):
#             batch_dict[k] = batch_dict[k].cuda()
#     return batch_dict


# class Config(PretrainedConfig):
#     vision_tower_cfg = {"image_size": 224}
#     num_video_frames = 16
#     resume_path = False
#     model_dtype = "torch.bfloat16"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('benchmark', type=str, default="stibench")
    parser.add_argument('function', type=str, default='demo')
    # parse known and unknown separately
    args, unknown = parser.parse_known_args()

    kwargs = {}
    for i in range(0, len(unknown), 2):
        key = unknown[i].lstrip("-")
        value = unknown[i + 1]

        # try to cast value to int/float/bool automatically
        if value.isdigit():
            value = int(value)
        else:
            try:
                value = float(value)
            except ValueError:
                if value.lower() in ["true", "false"]:
                    value = value.lower() == "true"
        kwargs[key] = value

    datamodule = get_datamodule(args.benchmark)

    if hasattr(datamodule, args.function):
        func = getattr(datamodule, args.function)
        assert callable(func), f"{args.function} is not callable!"
        func(**kwargs)
    else:
        raise NotImplementedError(
            f"Function {args.function} not implemented in datamodule {args.benchmark}!")

if __name__ == '__main__':
    main()
