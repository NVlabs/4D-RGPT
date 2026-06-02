# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import cv2
from PIL import Image
from pathlib import Path
from typing import Optional

class BaseDatamodule:

    name: str = ""
    path: str = ""
    registry: dict = {}
    tasks: list[str] | dict[str, list[str]] = []

    def __init__(self):
        self._loaded = False
        self.train_data: Optional[list[dict]] = None
        self.val_data: Optional[list[dict]] = None
        self.test_data: Optional[list[dict]] = None

    def load(self) -> None:
        raise NotImplementedError()

    def download(self) -> None:
        raise NotImplementedError()

    def get_gt_mask(self, *args, **kwargs) -> tuple[Image.Image | None, dict | None]:
        return None, None

    def stats(self) -> None:
        raise NotImplementedError()

    def demo(self) -> None:
        raise NotImplementedError()

    def inference(self, *args, **kwargs) -> None:
        raise NotImplementedError()

    def eval(self, filename: str):
        raise NotImplementedError()

    def get_video_stats(self, split="train") -> tuple[int, int, float]:
        match split:
            case "train":
                assert(self.train_data is not None), "Train data not loaded!"
                data = self.train_data
            case "test":
                assert(self.test_data is not None), "Test data not loaded!"
                data = self.test_data
            case _:
                raise ValueError(f"Unknown split {split}!")
        num_qas = len(data)

        videos = set()
        fps = None
        for x in data:
            videos.add(x['video'])
            video_path = Path(self.media_dir) / x['video']
            assert(video_path.exists()), f"Video {video_path} not found!"
            if fps is None:
                cap = cv2.VideoCapture(video_path.as_posix())
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
        num_videos = len(videos)

        return num_videos, num_qas, fps
