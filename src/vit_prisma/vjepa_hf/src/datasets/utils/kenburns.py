# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import cv2
import numpy as np
import torch


def zoom_pan(img, zoom=1, delta_xy=None):

    _, h, w = img.shape  # [C, H, W]
    img = np.transpose(img, (1, 2, 0))  # [H, W, C]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # convert torch img to cv2 (RGB->BGR)

    # Pan (translation)
    dx, dy = (delta_xy[0] * w, delta_xy[1] * h)
    M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)

    # Zoom
    cx, cy = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D((cx, cy), 0, zoom)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))

    return img


def apply_kenburns(img, nframe=16, max_zoom=2, end_xy=None, zoom_mode="in", dtype=None):

    # Checks to ensure zoom_mode is passed in as a valid option
    assert isinstance(zoom_mode, str), "Flag zoom_mode is not a sring"
    zoom_mode = zoom_mode.lower()
    assert zoom_mode in ["in", "out", "inout", "outin"], "Unrecognized zoom_mode"

    og_nframe = nframe
    if zoom_mode in ["inout", "outin"]:
        nframe = nframe // 2

    # Sample start/end coordinates for panning
    start_xy = (0.5, 0.5)  # start location is img center
    if end_xy is None:
        end_xy = torch.rand(2).tolist()
        end_xy = tuple([v / 4 for v in end_xy])  # only drift from center by 1/4 img width/height

    # -- delta-x,y in each frame (for panning)
    dx = (end_xy[0] - start_xy[0]) / nframe
    dy = (end_xy[1] - start_xy[1]) / nframe

    # -- delta-zoom in each frame
    dz = (max_zoom - 1.0) / nframe

    # Create zoom and pan animation
    prev_frame = img
    video = [prev_frame]
    for _ in range(nframe - 1):
        zoom = 1.0 + dz
        prev_frame = zoom_pan(prev_frame, zoom=zoom, delta_xy=(dx, dy))
        video += [prev_frame]

    if zoom_mode == "in":
        pass  # Nothing to do; already zoomed in :)
    elif zoom_mode == "out":
        video = video[::-1]
    elif zoom_mode == "inout":
        video = video + video[::-1]
    elif zoom_mode == "outin":
        video = video[::-1] + video

    # Repeat last frame until video length is the request number of frames,
    # there may be a mismatch if 'zoom_mode' is 'inout' or 'outin' and
    # nframes is an odd number
    if len(video) < og_nframe:
        video += [video[-1]]

    if dtype is not None:
        video = torch.cat([torch.tensor(f, dtype=dtype).unsqueeze(1) for f in video], dim=1)

    return video


def batch_apply_kenburns(batch, nframe=16, max_zoom=2, end_xy=None, zoom_mode="in", dtype=None):
    dtype = batch.dtype
    device = batch.device
    batch = [i.numpy(force=True) for i in batch]  # tensor->numpy
    batch = [apply_kenburns(i, nframe=nframe, dtype=dtype) for i in batch]
    batch = torch.cat([i.unsqueeze(0) for i in batch], dim=0).to(device)
    return batch
