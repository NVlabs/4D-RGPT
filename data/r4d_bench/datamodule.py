# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import cv2
import json
# import shutil
# import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from typing import Callable, Any

from ..base import BaseDatamodule
from ..register import register_datamodule
from ..config import DATASETS_ROOT, DATASETS
from ..utils.data_helper import mp42gif, print_stats, get_first_frame, extract_zip, get_video_length
from ..utils.data_helper import extract_video_clip, extract_video_frame, seconds_to_hhmmss_ms, sample_video_frames
from ..utils.prompts.vqa_format import FORMAT_PROMPTS
from ..metrics.multiple_choice import multipl_choice_accuracy
from utils import distributed as dist
from utils.logging import logger
from utils.eval_utils import parse_choice


@register_datamodule()
class R4DBench(BaseDatamodule):

    name = "R4D-Bench"
    dirname = 'r4d_bench'
    path = os.path.join(DATASETS_ROOT, name)
    registry = DATASETS[dirname]

    tasks = [
        '3D Video Grounding',
        'Dimensional Measurement',
        'Spatial Relation',
        'Rotational',
        'Counting',
        'Translational',
        'False Positives',
        'Speed & Acceleration',
        'Displacement & Path Length',
    ]
    fps: list[float] = [
        24, 10, 12, 30
    ]
    num_tasks = len(tasks)
    # raw_eval_path = registry["raw"]["eval_path"]
    # raw_media_dir = registry["raw"]["media_dir"]
    eval_path = registry["eval_path"]
    media_dir = Path(registry["media_dir"])
    os.makedirs(media_dir, exist_ok=True)

    def download(self):
        raise NotImplementedError

    @staticmethod
    def is_static(task: str) -> bool:
        return task in [
            "3D Video Grounding",
            "Dimensional Measurement",
            "Spatial Relation"
        ]

    @staticmethod
    def is_dynamic(task: str) -> bool:
        return not R4DBench.is_static(task)

    @staticmethod
    def task_abbr(task: str) -> str:
        match task:
            case "3D Video Grounding": return "VG"
            case "Dimensional Measurement": return "DM"
            case "Displacement & Path Length": return "DP"
            case "Spatial Relation": return "SR"
            case "Speed & Acceleration": return "SA"
            case 'False Positives': return "FP"
            case 'Rotational': return "R"
            case 'Counting': return "C"
            case 'Translational': return "T"
            case _: raise ValueError(f"Unknown task {task}")

    def load(self) -> None:
        if self.test_data:
            return
        with open(self.eval_path, 'r') as f:
            self.test_data = json.load(f)

    def stats(self) -> None:
        self.load()

        test_videos = set()
        if not self.fps:
            fps_list = set()
        if not self.tasks:
            tasks = set()
        for x in self.test_data:
            test_videos.add(x['video'])
            video_path = Path(self.media_dir) / x['video']
            assert(video_path.exists()), f"Video {video_path} not found!"
            if not self.fps:
                cap = cv2.VideoCapture(video_path.as_posix())
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                fps_list.add(fps)
            if not self.tasks:
                tasks.add(x['task'])
        if not self.fps:
            self.fps = list(fps_list)
        if not self.tasks:
            self.tasks = list(tasks)

        test_num_videos = len(test_videos)
        test_num_qas = len(self.test_data)

        print_stats(self.name, self.tasks, test_num_videos, test_num_qas, fps=self.fps)
        for task in self.tasks:
            task_data = [x for x in self.test_data if x['task'] == task]
            # num_videos = len(set([x['video'] for x in task_data]))
            num_qas = len(task_data)
            print(f"- {self.task_abbr(task)}: {num_qas} QAs")

    def eval(self, filename: str) -> dict[str, float]:
        with open(filename, "r", encoding="utf-8") as f:
            df = pd.DataFrame(json.load(f))

        df["output"] = df["response"].apply(lambda response: parse_choice(response, ["A", "B", "C", "D", "E"]))
        df["correct"] = df.apply(
            multipl_choice_accuracy,
            axis=1,
        )

        overall_acc = df["correct"].mean()
        task_acc = df.groupby("task")["correct"].mean()

        scores = {
            "Avg": overall_acc,
            "Sta": df[df['task'].apply(self.is_static)]['correct'].mean(),
            "Dyn": df[df['task'].apply(self.is_dynamic)]['correct'].mean(),
        }
        scores.update(
            { self.task_abbr(task): task_acc[task] for task in self.tasks if task in task_acc }
        )

        return scores

    def inference(
        self,
        instances: list[dict[str, Any]],
        model_response: Callable,
        save_every: bool = False,
        output_dir: str = "."
    ) -> list[dict[str, Any]]:

        outputs = []
        for entry in tqdm(instances, disable=not dist.is_main()):
            question = entry["question"]
            answer = entry["answer"]

            candidates = "\n".join([
                f"({chr(ord('A') + i)}) {v}" for i, v in enumerate(entry['options'])])

            image_path = self.media_dir / entry["som_image"]
            if not image_path.exists():
                logger.info(f"       ! image not found → {image_path}")
                continue

            video_path = self.media_dir / entry["video"]
            if not video_path.exists():
                logger.info(f"       ! video not found → {video_path}")
                continue
            # Paper §6.3 / Tab. 6 P4D+prompt trick: explicit temporal cue in the
            # question. Without this, accuracy on R4D drops ~3pp vs paper numbers
            # (matches the rvila path's `data_prepare/benchmarks/r4d/datamodule`
            # which is what scripts/nvila/eval.sh in 4d_rvila uses).
            video_sec = get_video_length(video_path.as_posix())
            question = f"These images are uniformly sampled from a video of {video_sec} seconds." + "\n" + question
            prompt = question + "\n" + "Options: \n" + candidates

            prompt += "\n" + FORMAT_PROMPTS['direct']

            response = model_response(
                image_path=image_path.as_posix(),
                video_path=video_path.as_posix(),
                text=prompt
            )

            output_letter = parse_choice(response, ["A", "B", "C", "D", "E"])

            outputs.append({
                "id": entry["id"],
                "video": entry["video"],
                "options": entry["options"],
                "question": question,
                "response": response,
                "output": output_letter,
                "task": entry["task"],
                "answer": "ABCDE"[entry['options'].index(answer)],
            })

            if save_every:
                temp_output_file = os.path.join(output_dir, f"temp_result_rank{dist.rank()}.json")
                with open(temp_output_file, "w", encoding="utf-8") as f:
                    json.dump(outputs, f, indent=4, ensure_ascii=False)
        return outputs
