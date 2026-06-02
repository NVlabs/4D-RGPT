import copy
import glob
import os
import torch
import random
from typing import Any, Dict, List, Optional

from llava.constants import DEFAULT_IMAGE_TOKEN
from llava.data.base import BaseDataset
from llava.data.dataset_impl.utils import _remove_media_tokens
from llava.media import Image, Video, Mask
from llava.utils import io, make_list
from llava.utils.logging import logger

__all__ = ["LLaVA4DVideoDataset"]

class LLaVA4DVideoDataset(BaseDataset):
    def __init__(self, data_path: str, media_dir: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.data_path = data_path
        self.media_dir = media_dir
        self.instances = io.load(self.data_path)

    def process(self, instance: Dict[str, Any]) -> List[Dict[str, Any]]:
        messages = instance["conversations"]

        medias = []
        if "image" in instance:
            for image_path in make_list(instance["image"]):
                medias.append(Image(os.path.join(self.media_dir, image_path)))

        if "video" in instance:
            for video_path in make_list(instance["video"]):
                medias.append(Video(os.path.join(self.media_dir, video_path)))
                if 'dataset_name' not in instance:
                    if 'final_version/' in video_path.lower():
                        instance['dataset_name'] = 'wolf'

        # if "rle" in instance or "segmentation" in instance or "bbox" in instance:
        #     region_modality_list = ["rle", "segmentation", "bbox"]
        #     region_modality_list = [_ for _ in region_modality_list if _ in instance.keys()]
        #     region_modality = random.choice(region_modality_list)
        #     for region in instance[region_modality]:
        #         # NOTE: for multi-vew setup, will be an empty list for content if no region
        #         # each Mask object's content is a list length 0~N masks
        #         medias.append(Mask(region_modality, region))

                # logger.debug(medias[-1])


        # if "region" in instance:
        #     img_flist = glob.glob(os.path.join(self.media_dir, instance["region"]) + "/*.jpeg")
        #     vpath = os.path.join(self.media_dir, instance["region"])

        #     assert len(img_flist) > 0, f"no images found in {vpath}"
        #     value = messages[0]["value"]
        #     img_list = [Image(img_path) for img_path in img_flist]
        #     new_value = [*img_list, value.replace(DEFAULT_IMAGE_TOKEN, "").strip()]
        #     messages[0]["value"] = new_value
        # Extract media from the instance

        # Remove media tokens from messages
        for message in messages:
            message["value"] = _remove_media_tokens(message["value"])

        # Add media to the beginning of the first message
        messages[0]["value"] = medias + [messages[0]["value"]]
        return messages

    # def __getitem__(self, index):
    #     sample, sample_str = self.get_dict_with_valid_vals(self.getitem_helper(index))

    #     # mirror and pad the sample in temporal dimension
    #     ori_video_len = sample["rgb_b3thw"].shape[-3]
    #     T_curr = ori_video_len
    #     crop_size = self.crop_size
    #     multiple = self.length_multiply_of
    #     if crop_size is None:
    #         T_new = ceil(max(T_curr, self.default_sample_size[0]) / multiple) * multiple
    #         crop_size = (T_new,) + self.default_sample_size[1:]

    #     if T_curr == 1:
    #         sample = self.repeat_single_frame(sample, crop_size[0])
    #     else:
    #         while T_curr < crop_size[0]:
    #             sample = self.mirror_and_pad(sample)
    #             T_curr = sample["rgb_b3thw"].shape[-3]

    #     # apply augmentations if any
    #     if self.use_augs and self.l4p_augs is not None:
    #         sample = self.l4p_augs.apply_augs(sample)

    #     # print(f"debug, after aug: {sample['rgb_b3thw'].shape}", flush=True)

    #     # perform spatial resizing
    #     if self.resize_size is not None:
    #         sample = self.resize(sample, self.resize_size, self.resize_mode)

    #     # print(f"debug, after resizing: {sample['rgb_b3thw'].shape}", flush=True)

    #     # perform spatio-temporal random cropping
    #     sample = self.crop(sample, crop_size)

    #     # print(f"debug, after cropping: {sample['rgb_b3thw'].shape}", flush=True)

    #     # perform track_2d sampling
    #     sample = self.sample_tracks(sample, sample_str, self.track_2d_repeat_traj)

    #     # final postprocessings and augmentations
    #     sample = self.post_process(sample)

    #     # get in proper form
    #     mean = self.input_mean[:, None, None, None]
    #     std = self.input_std[:, None, None, None]
    #     sample["rgb_mean_b3111"] = mean
    #     sample["rgb_std_b3111"] = std
    #     sample["rgb_b3thw"] = (sample["rgb_b3thw"] - mean) / std
    #     for key in sample.keys():
    #         if torch.is_tensor(sample[key]):
    #             sample[key] = sample[key].contiguous()
    #     sample.update(sample_str)
    #     sample["ori_video_len"] = ori_video_len  # type: ignore

    #     return sample


    # def getitem_helper(self, index: int) -> L4PData:

    #     rgbs = []
    #     instances = []
    #     video_name = os.path.basename(self.video_paths[index])

    #     with media.VideoReader(self.video_paths[index]) as reader:
    #         print(f"Video has {reader.num_images} images with shape={reader.shape} at {reader.fps}")
    #         if reader.num_images > self.max_frames:
    #             print(
    #                 f"Trimming video to {self.max_frames} frames.",
    #                 "This is the limitation of current implementation.",
    #             )
    #         count = 0
    #         for rgb in reader:
    #             rgb = media.resize_image(rgb, self.resize_size)
    #             rgb = F.to_tensor(rgb)[:3][:, None]
    #             rgbs.append(rgb)
    #             instance = torch.zeros_like(rgb[:1])
    #             instances.append(instance)

    #             count += 1
    #             if count == self.max_frames - 1:
    #                 break

    #     rgb_b3thw = torch.cat(rgbs, dim=1)
    #     instance_b1thw = (torch.mean(torch.cat(instances, dim=1), dim=0, keepdim=True) > 0).to(
    #         dtype=torch.float32
    #     )

    #     # generate sample for L4PData
    #     sample = {}

    #     # pass dummy intrinsics
    #     _, _, H, W = rgb_b3thw.shape
    #     fx = min(H, W)
    #     fy = min(H, W)
    #     cx = W / 2
    #     cy = H / 2
    #     intrinsics_b44 = torch.Tensor(
    #         [
    #             [fx, 0, cx, 0],
    #             [0, fy, cy, 0],
    #             [0, 0, 1, 0],
    #             [0, 0, 0, 1],
    #         ]
    #     )

    #     # pass to the base class
    #     sample = L4PData(
    #         rgb_b3thw=rgb_b3thw,
    #         intrinsics_b44t=einops.repeat(intrinsics_b44, "m n -> m n k", k=rgb_b3thw.shape[-3]).clone(),
    #         instanceseg_b1thw=instance_b1thw,
    #         seq_name=str(video_name),
    #     )

    #     return sample