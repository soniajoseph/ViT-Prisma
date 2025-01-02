# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# The data loader for MVBench
# For more context on the code checkout https://github.com/OpenGVLab/Ask-Anything/blob/main/video_chat2/MVBENCH.md
# Part of this code is taked from their notebook here:
# https://github.com/OpenGVLab/Ask-Anything/blob/main/video_chat2/mvbench.ipynb

import json
import os

import cv2
import imageio
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

from evals.mvbench.mvbench_transforms import GroupCenterCrop, GroupNormalize, GroupScale, Stack, ToTorchFormatTensor
from vit_prisma.vjepa_hf.src.utils.logging import get_logger

_GLOBAL_SEED = 0
logger = get_logger("MVBench Data Loader")


DATA_LIST = {
    "Action Sequence": (
        "action_sequence.json",
        "star/Charades_v1_480/",
        "video",
        True,
    ),  # has start & end
    "Action Prediction": (
        "action_prediction.json",
        "star/Charades_v1_480/",
        "video",
        True,
    ),  # has start & end
    "Action Antonym": ("action_antonym.json", "ssv2_video/", "video", False),
    "Fine-grained Action": (
        "fine_grained_action.json",
        "Moments_in_Time_Raw/videos/",
        "video",
        False,
    ),
    "Unexpected Action": ("unexpected_action.json", "FunQA_test/test/", "video", False),
    "Object Existence": (
        "object_existence.json",
        "clevrer/video_validation/",
        "video",
        False,
    ),
    "Object Interaction": (
        "object_interaction.json",
        "star/Charades_v1_480/",
        "video",
        True,
    ),  # has start & end
    "Object Shuffle": ("object_shuffle.json", "perception/videos/", "video", False),
    "Moving Direction": (
        "moving_direction.json",
        "clevrer/video_validation/",
        "video",
        False,
    ),
    "Action Localization": (
        "action_localization.json",
        "sta/sta_video/",
        "video",
        True,
    ),  # has start & end
    "Scene Transition": ("scene_transition.json", "scene_qa/video/", "video", False),
    "Action Count": ("action_count.json", "perception/videos/", "video", False),
    "Moving Count": ("moving_count.json", "clevrer/video_validation/", "video", False),
    "Moving Attribute": (
        "moving_attribute.json",
        "clevrer/video_validation/",
        "video",
        False,
    ),
    "State Change": ("state_change.json", "perception/videos/", "video", False),
    "Fine-grained Pose": ("fine_grained_pose.json", "nturgbd/", "video", False),
    "Character Order": ("character_order.json", "perception/videos/", "video", False),
    "Egocentric Navigation": ("egocentric_navigation.json", "vlnqa/", "video", False),
    "Episodic Reasoning": (
        "episodic_reasoning.json",
        "tvqa/frames_fps3_hq/",
        "frame",
        True,
    ),  # has start & end, read frame
    "Counterfactual Inference": (
        "counterfactual_inference.json",
        "clevrer/video_validation/",
        "video",
        False,
    ),
}


class MVBenchDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, num_segments=8, resolution=224, transform=None):
        self.data_dir = data_dir
        self.data = []
        for k, v in DATA_LIST.items():
            json_fpath = os.path.join(data_dir, "json", v[0])
            with open(json_fpath, "r") as f:
                json_data = json.load(f)
            for data in json_data:
                self.data.append(
                    {
                        "task_type": k,
                        "prefix": v[1],
                        "data_type": v[2],
                        "bound": v[3],
                        "data": data,
                    }
                )

        self.decord_method = {
            "video": self.read_video,
            "gif": self.read_gif,
            "frame": self.read_frame,
        }

        self.num_segments = num_segments

        # transform
        crop_size = resolution
        scale_size = resolution
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]
        if transform is None:
            self.transform = T.Compose(
                [
                    GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
                    GroupCenterCrop(crop_size),
                    Stack(),
                    ToTorchFormatTensor(),
                    GroupNormalize(input_mean, input_std),
                ]
            )
        else:
            self.transform = transform

    def __len__(self):
        return len(self.data)

    def get_index(self, bound, fps, max_frame, first_idx=0):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / self.num_segments
        frame_indices = np.array(
            [int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(self.num_segments)]
        )
        return frame_indices

    def read_video(self, video_path, bound=None):
        video_full_path = os.path.join(self.data_dir, "video", video_path)
        vr = VideoReader(video_full_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())

        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0)
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy())
            images_group.append(img)
        torch_imgs = self.transform(images_group)
        return torch_imgs

    def read_gif(self, video_path, bound=None, fps=25):
        video_full_path = os.path.join(self.data_dir, "video", video_path)
        gif = imageio.get_reader(video_full_path)
        max_frame = len(gif) - 1

        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0)
        for index, frame in enumerate(gif):
            if index in frame_indices:
                img = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                img = Image.fromarray(img)
                images_group.append(img)
        torch_imgs = self.transform(images_group)
        return torch_imgs

    def read_frame(self, video_path, bound=None, fps=3):
        video_full_path = os.path.join(self.data_dir, "video", video_path)
        max_frame = len(os.listdir(video_full_path))
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=1)  # frame_idx starts from 1
        for frame_index in frame_indices:
            img = Image.open(os.path.join(video_full_path, f"{frame_index:05d}.jpg"))
            images_group.append(img)
        torch_imgs = self.transform(images_group)
        return torch_imgs

    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        answer = data["answer"].strip()
        answer_idx = -1
        for idx, c in enumerate(data["candidates"]):
            c = c.strip()
            question += f"({chr(ord('A') + idx)}) {c}\n"
            if c == answer:
                answer_idx = idx
        question = question.rstrip()
        answer = f"({chr(ord('A') + answer_idx)}) {answer}"
        return question, answer

    def __getitem__(self, idx):
        decord_method = self.decord_method[self.data[idx]["data_type"]]
        bound = None
        if self.data[idx]["bound"]:
            bound = (
                self.data[idx]["data"]["start"],
                self.data[idx]["data"]["end"],
            )
        video_path = os.path.join(self.data[idx]["prefix"], self.data[idx]["data"]["video"])
        torch_imgs = decord_method(video_path, bound)
        question, answer = self.qa_template(self.data[idx]["data"])

        return {
            "video": torch_imgs,
            "video_path": video_path,
            "question": question,
            "answer": answer,
            "task_type": self.data[idx]["task_type"],
        }


def make_mvbench_dataset(
    batch_size,
    num_workers,
    world_size,
    rank,
    root_path,
    image_folder,
    num_frames,
    transform,
):
    dataset = MVBenchDataset(
        os.path.join(root_path, image_folder),
        num_segments=num_frames,
        transform=transform,
    )
    logger.info(f"MVBench dataset created successfully. ({len(dataset)} examples)")
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers)
    return dataset, dataloader, sampler
