# FOR FLOW VIS
# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-08-03

# FOR TRACK VIS
# COTRACKER KUBRIC

import os
import cv2
import numpy as np
import torch
import imageio
import einops
import matplotlib
import matplotlib.cm as pltcm
from pathlib import Path
from typing import Optional, Tuple, List, Any, Dict

from torch import Tensor

from .l4p_helper import colormap_image, flow_video_to_color_with_bounds
# from llava.model.low_level_perception.process import L4P_INPUT_MEAN, L4P_INPUT_STD

from llava.utils.logging import logger

from ..base import LowLevelPerceptionOutput

def visualize_depth(
    depth_pred: Tensor,
    vis_min_depth: float = 0.05,
    vis_max_depth: float = 20.0
) -> np.ndarray:
    depth_est_1thw = depth_pred[0]
    depth_range = (
        max(torch.min(depth_est_1thw[depth_est_1thw > 0]).item(), vis_min_depth),
        min(torch.max(depth_est_1thw[depth_est_1thw > 0]).item(), vis_max_depth),
    )
    depth_est_1thw = torch.clamp(depth_est_1thw, min=depth_range[0], max=depth_range[1])
    depth_est_vis, _, _ = colormap_image(depth_est_1thw, vmin=depth_range[0], vmax=depth_range[1])
    depth_est_vis = depth_est_vis.cpu().numpy().transpose((1, 2, 3, 0))
    return depth_est_vis

def visualize_flow_2d(
    flow_2d_pred: Tensor,
) -> np.ndarray:
    flow_2d_backward_est_b2thw = flow_2d_pred.cpu()
    bflow_est_vis_b3thw, _ = flow_video_to_color_with_bounds(
        flow_2d_backward_est_b2thw, None, max_flow_mag=25.0
    )
    bflow_est_vis_thw3 = bflow_est_vis_b3thw[0].numpy().transpose((1, 2, 3, 0))
    bflow_est_vis_thw3 = bflow_est_vis_thw3.astype(np.float32)
    return bflow_est_vis_thw3

def visualize_dyn_mask(
    dyn_mask_pred: Tensor,
    threhsold: float = 0.85
) -> np.ndarray:
    dyn_mask_est_1thw = dyn_mask_pred[0]
    dyn_mask_est_1thw = torch.nn.functional.sigmoid(dyn_mask_est_1thw)
    dyn_mask_est_1thw = (dyn_mask_est_1thw > threhsold).to(dtype=torch.float32)
    dyn_mask_est_thw3 = dyn_mask_est_1thw[0, ..., None].repeat(1, 1, 1, 3).cpu().numpy()
    dyn_mask_est_thw3 = dyn_mask_est_thw3.astype(np.float32)
    return dyn_mask_est_thw3

# def visualize_track_2d(

# ):
#     data = out | batch
#     for key, _ in data.items():
#         if torch.is_tensor(data[key]):
#             data[key] = data[key].to(device=torch.device("cpu"))
#     out_vis_2d_thw3, _ = visualize_2d_3d_tracks(
#         data,
#         os.path.join(out_path, seq_name),
#         "ours",
#         vis_gt=False,
#         depth_fn_est="linear",
#         vis_fn_est="sigmoid",
#         fix_scale=True,
#         tracks_leave_trace=64,
#         depth_subsample_factor=1,
#         get_point_cloud=True,  # False,
#         combine_pc_and_tracks=True,  # False,
#     )

def visualize_track_2d(
    video,
    batch,
    out,
    tracks_leave_trace=16,
    vis_thr=0.75,
) -> np.ndarray:
    """Function to visualize 2D tracks.

    Args:
        batch (Dict[str, Any]): Batch of data to process and visualize.
        out (Dict[str, Any]): Output of the model.
        vis_fn_est (str, optional): Function to apply to the visibility estimate. Defaults to "sigmoid".
        tracks_leave_trace (int, optional): Number of frames to leave the trace of the tracks. Defaults to 16.
        vis_thr (float, optional): Threshold for the visibility estimate. Defaults to 0.75.

    Returns:
        np.ndarray: Video visualization of the 2D tracks.
    """
    rgb_b3thw = video#batch["rgb_b3thw"] * batch["rgb_std_b3111"] + batch["rgb_mean_b3111"]
    sorted_indices = torch.argsort(batch["track_2d_traj_bn2t"][0, :, 1, 0])  # Sort points over height

    track_2d_vis_est_bn1t = torch.sigmoid(out["track_2d_vis_est_bn1t"])
    track_2d_vis_est_bn1t = track_2d_vis_est_bn1t[:, sorted_indices, :, :]
    track_2d_traj_est_bn2t = out["track_2d_traj_est_bn2t"][:, sorted_indices, :, :]

    vis_bnt = track_2d_vis_est_bn1t[..., 0, :]
    vis_bnt = vis_bnt.cpu().numpy() > vis_thr
    vis_tn = einops.rearrange(vis_bnt[0], "n t -> t n")

    rgb_thw3 = einops.rearrange(rgb_b3thw[0].clone(), "c t h w -> t h w c")
    rgb_thw3 = torch.repeat_interleave(torch.mean(rgb_thw3, dim=-1, keepdims=True), 3, dim=-1).cpu().numpy()  # type: ignore
    track_2d_traj_tn2 = einops.rearrange(track_2d_traj_est_bn2t[0], "n c t -> t n c").cpu().numpy()
    out_vis_2d_thw3 = plot_track_2d(
        rgb_thw3, track_2d_traj_tn2, vis_tn, tracks_leave_trace=tracks_leave_trace
    )
    return out_vis_2d_thw3


def plot_track_2d(
    video,
    points,
    visibles,
    infront_cameras=None,
    tracks_leave_trace=16,
    show_occ=False
) -> np.ndarray:
    """Visualize 2D point trajectories."""
    num_frames, num_points = points.shape[:2]

    # Precompute colormap for points
    # color_map = matplotlib.colormaps.get_cmap('hsv') # AB
    color_map = pltcm.get_cmap("hsv")  # AB
    cmap_norm = matplotlib.colors.Normalize(vmin=0, vmax=num_points - 1)
    point_colors = np.zeros((num_points, 3))
    for i in range(num_points):
        point_colors[i] = np.array(color_map(cmap_norm(i)))[:3]  # * 255

    if infront_cameras is None:
        infront_cameras = np.ones_like(visibles).astype(bool)

    frames = []
    for t in range(num_frames):
        frame = video[t].copy()

        # Draw tracks on the frame
        line_tracks = points[max(0, t - tracks_leave_trace) : t + 1]
        line_visibles = visibles[max(0, t - tracks_leave_trace) : t + 1]
        line_infront_cameras = infront_cameras[max(0, t - tracks_leave_trace) : t + 1]
        for s in range(line_tracks.shape[0] - 1):
            img = frame.copy()

            for i in range(num_points):
                if line_visibles[s, i] and line_visibles[s + 1, i]:  # visible
                    x1, y1 = int(round(line_tracks[s, i, 0])), int(round(line_tracks[s, i, 1]))
                    x2, y2 = int(round(line_tracks[s + 1, i, 0])), int(round(line_tracks[s + 1, i, 1]))
                    cv2.line(frame, (x1, y1), (x2, y2), point_colors[i], 1, cv2.LINE_AA)
                elif show_occ and line_infront_cameras[s, i] and line_infront_cameras[s + 1, i]:  # occluded
                    x1, y1 = int(round(line_tracks[s, i, 0])), int(round(line_tracks[s, i, 1]))
                    x2, y2 = int(round(line_tracks[s + 1, i, 0])), int(round(line_tracks[s + 1, i, 1]))
                    cv2.line(frame, (x1, y1), (x2, y2), point_colors[i], 1, cv2.LINE_AA)

            alpha = (s + 1) / (line_tracks.shape[0] - 1)
            frame = cv2.addWeighted(frame, alpha, img, 1 - alpha, 0)

        # Draw end points on the frame
        for i in range(num_points):
            if visibles[t, i]:  # visible
                x, y = int(round(points[t, i, 0])), int(round(points[t, i, 1]))
                cv2.circle(frame, (x, y), 2, point_colors[i], -1)
            elif show_occ and infront_cameras[t, i]:  # occluded
                x, y = int(round(points[t, i, 0])), int(round(points[t, i, 1]))
                cv2.circle(frame, (x, y), 2, point_colors[i], 1)

        frames.append(frame)
    frames = np.stack(frames)
    return frames




# def generate_4D_visualization(
#     video: torch.Tensor,
#     batch,
#     out,
#     tasks,
#     out_path
# ):
#     """Process and visualize 4D reconstruction.

#     Args:
#         batch (Dict[str, Any]): Batch of data to process and visualize.
#         out (Dict[str, Any]): Output of the model.
#         tasks (List[str]): Tasks to visualize.
#         out_path (str, optional): Path to save the visualizations.
#     """
#     B, _, T, H, W = batch["rgb_b3thw"].shape
#     assert "depth" in tasks and "camray" in tasks, "Tasks must include depth, camray"
#     assert B == 1, "Current implementation supports only batch size 1"
#     # seq_name = batch["seq_name"][0]
#     # device = batch["rgb_b3thw"].device
#     # dtype = batch["rgb_b3thw"].dtype

#     # out_path = os.path.join(out_path, seq_name)
#     # os.makedirs(out_path, exist_ok=True)

#     if "traj3d_est_b16t" in out.keys():
#         batch["intrinsics_b44t"] = out["traj3d_intrinsics_est_b16t"].reshape(1, 4, 4, T)

#     if "camray_est_b6thw" in out.keys():
#         intrinsics_norm_b44t = normalize_intrinsics(batch["intrinsics_b44t"], H, W).to(
#             dtype=dtype, device=device
#         )
#         extrinsics_est_b44t, _ = rays_to_cameras(
#             camray_b6thw=out["camray_est_b6thw"], intrinsics_b44t=intrinsics_norm_b44t, ctr_only=False
#         )
#     else:
#         extrinsics_est_b44t = torch.linalg.inv(
#             out["traj3d_est_b16t"].permute(0, 2, 1).reshape(1, T, 4, 4)
#         ).permute(0, 2, 3, 1)

#     extrinsics_est_b44t = get_cam_T_ref(extrinsics_est_b44t, ref_idx=0)

#     rgb_b3thw = batch["rgb_b3thw"] * batch["rgb_std_b3111"] + batch["rgb_mean_b3111"]

#     # track3d
#     if "track_2d" in tasks:
#         vis_thr = 0.75

#         fix_scale = True
#         track_2d_vis_est_bn1t = apply_fn(out["track_2d_vis_est_bn1t"], "sigmoid").clone()
#         track_2d_depth_est_bn1t = apply_fn(out["track_2d_depth_est_bn1t"], "linear").clone()
#         track_2d_traj_est_bn2t = out["track_2d_traj_est_bn2t"].clone()

#         if fix_scale:
#             traj_norm = out["track_2d_traj_est_bn2t"][0].clone()
#             traj_norm[:, 0, :] = traj_norm[:, 0, :] / (W - 1) * 2 - 1
#             traj_norm[:, 1, :] = traj_norm[:, 1, :] / (H - 1) * 2 - 1
#             traj_sampled_depth_nt = torch.nn.functional.grid_sample(
#                 out["depth_est_b1thw"][0].permute(1, 0, 2, 3).to(dtype=torch.float32),
#                 traj_norm.permute(2, 0, 1).unsqueeze(2),
#                 mode="nearest",
#                 align_corners=False,
#             )[:, 0, :, 0].permute(1, 0)

#             vis_est_nt = track_2d_vis_est_bn1t[0, :, 0] > vis_thr
#             traj_sampled_depth_good = traj_sampled_depth_nt[vis_est_nt > 0]
#             track_2d_depth_good = track_2d_depth_est_bn1t[0, :, 0][vis_est_nt > 0]
#             scale = torch.median(traj_sampled_depth_good / track_2d_depth_good)
#             track_2d_depth_est_bn1t = scale * track_2d_depth_est_bn1t

#         track_pc_list = generate_3d_track_point_clouds(
#             track_2d_traj_est_bn2t,
#             track_2d_depth_est_bn1t,
#             track_2d_vis_est_bn1t,
#             batch["intrinsics_b44t"].clone(),
#             extrinsics_est_b44t.clone(),
#             vis_thr=vis_thr,
#             tracks_leave_trace=16,
#             sort_points_by_height=True,
#         )[0]

#     camera_traj_mesh_list = generate_video_camera_trajectory(extrinsics_est_b44t.clone())[0]
#     point_clouds_list = generate_video_point_clouds(
#         rgb_b3thw.clone(),
#         out["depth_est_b1thw"].clone(),
#         batch["intrinsics_b44t"].clone(),
#         extrinsics_est_b44t.clone(),
#     )[0]

#     ply_paths = []
#     for index in range(T):
#         if "track_2d" not in tasks:
#             ply_path = os.path.join(out_path, f"{index}_world.ply")
#             o3d.io.write_point_cloud(ply_path, point_clouds_list[index])
#             ply_path_cam = ply_path.replace(".ply", "_cam_mesh.ply")
#             o3d.io.write_triangle_mesh(ply_path_cam, camera_traj_mesh_list[index])
#             ply_paths.append(
#                 {
#                     "name": f"{seq_name}_{index}",
#                     "pc_depth": ply_path,
#                     "mesh_cam": ply_path_cam,
#                 }
#             )
#         else:
#             ply_path = os.path.join(out_path, f"{index}_world.ply")
#             ply_path_cam = ply_path.replace(".ply", "_cam_mesh.ply")
#             o3d.io.write_triangle_mesh(ply_path_cam, camera_traj_mesh_list[index])
#             ply_path_track_depth = ply_path.replace(".ply", "_track_depth_pc.ply")
#             o3d.io.write_point_cloud(
#                 ply_path_track_depth,
#                 point_clouds_list[index] + track_pc_list[index],
#             )
#             ply_paths.append(
#                 {
#                     "name": f"{seq_name}_{index}",
#                     "pc_depth_track": ply_path_track_depth,
#                     "mesh_cam": ply_path_cam,
#                 }
#             )

#     return ply_paths




def visualize_tasks(
    video: torch.Tensor,
    outputs: LowLevelPerceptionOutput,
    tasks: List[str] = ["depth", "flow_2d_backward", "dyn_mask", "track_2d"],
    out_dir: str = "results",
    seq_name: str = "demo"
) -> None:
    """
    video: shape [3, T, H, W]. range [0~1]
    """
    # rgb_3thw = (video.cpu() * L4P_INPUT_STD + L4P_INPUT_MEAN)
    rgb_thw3 = video.cpu().float().numpy().transpose((1, 2, 3, 0))
    # logger.info(rgb_thw3.shape)
    out_video = [rgb_thw3]

    for task in tasks:
        match task:
            case "depth":
                out_vis = visualize_depth(outputs.pred_depth.float())
            case "flow_2d_backward":
                out_vis = visualize_flow_2d(outputs.pred_flow_2d_backward.float())
            case "dyn_mask":
                out_vis = visualize_dyn_mask(outputs.pred_dyn_mask.float())
            case "camray":
                logger.debug("not implemented yet")
                continue
            case "track_2d":
                logger.debug("not implemented yet")
                continue
            case _:
                raise ValueError(task)

        # logger.info(out_vis.shape)
        out_video.append(out_vis)

    out_video = (np.concatenate(out_video, axis=-2) * 255).astype(np.uint8)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_path /= f"{seq_name}.mp4"

    logger.debug(out_path)
    with imageio.get_writer(out_path, fps=15) as writer:
        for frame in out_video:
            writer.append_data(frame)
