# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import numpy as np
import torch

# import matplotlib

# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.cm as pltcm
from typing import Optional, Tuple, List, Any, Dict

import einops
# Dead imports (neither symbol used anywhere in this file). Dropped so the
# visualize-tasks path doesn't require these heavy/optional viz deps.
# import open3d as o3d
import cv2
# import mediapy as media
# import viser


# def apply_fn(x: torch.Tensor, fn_type: str = "linear"):
#     if fn_type == "log":
#         out = torch.log(x)
#     elif fn_type == "exp":
#         out = torch.exp(x)
#     elif fn_type == "sigmoid":
#         out = torch.nn.functional.sigmoid(x)
#     elif fn_type == "linear":
#         out = x
#     elif fn_type == "inverse":
#         eps = 1e-8
#         mask = x.abs() > eps
#         out = torch.zeros_like(x)
#         out[mask] = (1.0 / x[mask]).to(x.dtype)
#     else:
#         print(f"Not implemented {fn_type}")
#         raise NotImplementedError

#     return out.to(x.dtype)


# def unproject_2d_track_to_3d(
#     track_xy_bn2t: torch.Tensor, track_Z_bn1t: torch.Tensor, intrinsics_b44t: torch.Tensor
# ) -> torch.Tensor:
#     """Projects 2D track and depth into 3D in camera coordinate system.

#     Args:
#         track_xy_bn2t (torch.Tensor): 2d track with [x,y] positions
#         track_Z_bn1t (torch.Tensor): depth of track
#         intrinsics_b44t (torch.Tensor): camera intrinsics, assumes simple pinhole model [fx, fy, cx, cy]

#     Returns:
#         torch.Tensor: 3D tracking with [X,Y,Z] in camera coordinate system
#     """
#     track_X_bn1t = (
#         (track_xy_bn2t[:, :, 0:1, :] - intrinsics_b44t[:, 0:1, 2:3, :])
#         * track_Z_bn1t
#         / intrinsics_b44t[:, 0:1, 0:1, :]
#     )
#     track_Y_bn1t = (
#         (track_xy_bn2t[:, :, 1:2, :] - intrinsics_b44t[:, 1:2, 2:3, :])
#         * track_Z_bn1t
#         / intrinsics_b44t[:, 1:2, 1:2, :]
#     )
#     track_XYZ_bn3t = torch.cat([track_X_bn1t, track_Y_bn1t, track_Z_bn1t], dim=-2)

#     return track_XYZ_bn3t


def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[col : col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col : col + YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col : col + GC, 1] = 255
    colorwheel[col : col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[col : col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col : col + CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col : col + BM, 2] = 255
    colorwheel[col : col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[col : col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col : col + MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75  # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)
    return flow_image


# def flow_to_color(flow_uv, clip_flow=None, convert_to_bgr=False):
#     """
#     Expects a two dimensional flow image of shape.

#     Args:
#         flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
#         clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
#         convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

#     Returns:
#         np.ndarray: Flow visualization image of shape [H,W,3]
#     """
#     assert flow_uv.ndim == 3, "input flow must have three dimensions"
#     assert flow_uv.shape[2] == 2, "input flow must have shape [H,W,2]"
#     if clip_flow is not None:
#         flow_uv = np.clip(flow_uv, 0, clip_flow)
#     u = flow_uv[:, :, 0]
#     v = flow_uv[:, :, 1]
#     rad = np.sqrt(np.square(u) + np.square(v))
#     rad_max = np.max(rad)
#     epsilon = 1e-5
#     u = u / (rad_max + epsilon)
#     v = v / (rad_max + epsilon)
#     return flow_uv_to_colors(u, v, convert_to_bgr)


# def flow_video_to_color(flow_uv_b2thw, flow_bounds=None):

#     assert flow_bounds is None
#     B, _, T, H, W = flow_uv_b2thw.shape
#     flow_viz = np.zeros((B, 3, T, H, W))

#     for b in range(B):
#         for t in range(T):
#             flow_i = flow_uv_b2thw[b, :, t].permute(1, 2, 0).detach().cpu().numpy()
#             flow_viz_i = flow_to_color(flow_i)
#             flow_viz[b, :, t] = flow_viz_i.transpose(2, 0, 1)

#     flow_viz = torch.from_numpy(flow_viz).to(flow_uv_b2thw.device).to(flow_uv_b2thw.dtype) / 255.0

#     return flow_viz, flow_bounds


def flow_to_color_with_bounds(flow_uv, rad_max=None, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, "input flow must have three dimensions"
    assert flow_uv.shape[2] == 2, "input flow must have shape [H,W,2]"
    if clip_flow is not None:
        # flow_uv = np.clip(flow_uv, 0, clip_flow)
        flow_uv = np.clip(flow_uv, -clip_flow, clip_flow)
    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]
    rad = np.sqrt(np.square(u) + np.square(v))
    if rad_max is None:
        rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)


def flow_video_to_color_with_bounds(flow_uv_b2thw, flow_bounds=None, max_flow_mag=-1.0):

    # assert flow_bounds is None
    B, _, T, H, W = flow_uv_b2thw.shape
    flow_viz = np.zeros((B, 3, T, H, W))

    flow_bounds_new = []
    for b in range(B):
        rad_max = (
            torch.max(torch.sqrt(torch.square(flow_uv_b2thw[0, 0]) + torch.square(flow_uv_b2thw[0, 1])))
            .cpu()
            .item()
        )
        rad_max = min(max_flow_mag, rad_max) if max_flow_mag > 0 else rad_max
        rad_max = rad_max if flow_bounds is None else flow_bounds[b]
        for t in range(T):
            flow_i = flow_uv_b2thw[b, :, t].permute(1, 2, 0).detach().cpu().numpy()
            flow_viz_i = flow_to_color_with_bounds(flow_i, rad_max=rad_max, clip_flow=rad_max / np.sqrt(2))
            # flow_viz_i = flow_to_color(flow_i)
            flow_viz[b, :, t] = flow_viz_i.transpose(2, 0, 1)
        flow_bounds_new.append(rad_max)

    flow_viz = torch.from_numpy(flow_viz).to(flow_uv_b2thw.device).to(flow_uv_b2thw.dtype) / 255.0

    return flow_viz, flow_bounds_new


# def flow_video_to_color_1(flow_uv_b2thw, flow_bounds=None):
#     if flow_bounds is None:
#         flow_bounds = (
#             torch.amin(flow_uv_b2thw.view(flow_uv_b2thw.shape[0], -1), dim=1)[:, None, None, None, None],
#             torch.amax(flow_uv_b2thw.view(flow_uv_b2thw.shape[0], -1), dim=1)[:, None, None, None, None],
#         )

#     flow_vis = (flow_uv_b2thw - flow_bounds[0]) / (flow_bounds[1] - flow_bounds[0])
#     flow_vis = torch.cat([torch.zeros_like(flow_vis[:, 0:1]), flow_vis], dim=1)

#     return flow_vis, flow_bounds


def colormap_image(
    image_1thw,
    mask_1thw=None,
    invalid_color=(0.0, 0, 0.0),
    flip=True,
    vmin=None,
    vmax=None,
    return_vminvmax=False,
    colormap="turbo",
):
    """
    Colormaps a one channel tensor using a matplotlib colormap.

    Args:
        image_1thw: the tensor to colomap.
        mask_1thw: an optional float mask where 1.0 donates valid pixels.
        colormap: the colormap to use. Default is turbo.
        invalid_color: the color to use for invalid pixels.
        flip: should we flip the colormap? True by default.
        vmin: if provided uses this as the minimum when normalizing the tensor.
        vmax: if provided uses this as the maximum when normalizing the tensor.
            When either of vmin or vmax are None, they are computed from the
            tensor.
        return_vminvmax: when true, returns vmin and vmax.

    Returns:
        image_cm_3thw: image of the colormapped tensor.
        vmin, vmax: returned when return_vminvmax is true.


    """
    valid_vals = image_1thw if mask_1thw is None else image_1thw[mask_1thw.bool()]
    if vmin is None:
        vmin = valid_vals.min()
    if vmax is None:
        vmax = valid_vals.max()

    cmap = torch.Tensor(plt.cm.get_cmap(colormap)(torch.linspace(0, 1, 256))[:, :3]).to(  # type: ignore
        image_1thw.device
    )
    if flip:
        cmap = torch.flip(cmap, (0,))

    t, h, w = image_1thw.shape[1:]

    image_norm_1thw = (image_1thw - vmin) / ((vmax - vmin) * 1.05)
    image_int_1thw = (torch.clamp(image_norm_1thw * 255, 0, 255)).byte().long()

    image_cm_3thw = cmap[image_int_1thw.flatten(start_dim=1)].permute([0, 2, 1]).view([-1, t, h, w])

    if mask_1thw is not None:
        mask_1thw = mask_1thw.float()
        invalid_color = torch.Tensor(invalid_color).view(3, 1, 1, 1).to(image_1thw.device)
        image_cm_3thw = image_cm_3thw * mask_1thw + invalid_color * (1 - mask_1thw)

    return image_cm_3thw, vmin, vmax
