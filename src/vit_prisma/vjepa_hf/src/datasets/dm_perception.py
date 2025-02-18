import json
import os

import decord
import numpy as np
import torch
from torchvision.transforms import v2

from evals.video_classification_frozen.utils import make_transforms
from vit_prisma.vjepa_hf.src.utils.logging import get_logger

DATA_SPLITS = ("train", "valid", "test")
TASKS = (
    "action_localisation",
    "grounded_question",
    "mc_question",
    "object_tracking",
    "point_tracking",
    "sound_localisation",
)
OPTIONS_DELIM = " :: "

logger = get_logger("DM Perception Dataset")


def dm_perception_make_transforms(data_split, resolution=224, normalize=True):
    # Computed by ./scripts/mean_std.py
    # Resolution set to 224, and 64 frames are sampled.
    # NOTE: After some discussion on PR#54, it was decided that this is not the correct way of doing the normalization.
    # Reverting to the commonly used make_transforms in JEPA. But keeping this here for future reference (maybe a
    # single trial later).
    MEANS = {
        "train": (0.4892, 0.4403, 0.4156),
        "valid": (0.4875, 0.4381, 0.4127),
        "test": (0.4936, 0.4424, 0.4201),
    }
    STDS = {
        "train": (0.1935, 0.1958, 0.1974),
        "valid": (0.1937, 0.1946, 0.1955),
        "test": (0.1940, 0.1957, 0.1958),
    }

    transforms_list = []

    if resolution > 0:
        transforms_list.append(v2.RandomResizedCrop(size=(resolution, resolution), antialias=True))

    transforms_list.append(v2.ToDtype(torch.float32, scale=True))

    if normalize:
        transforms_list.append(v2.Normalize(mean=MEANS[data_split], std=STDS[data_split]))

    return v2.Compose(transforms_list)


def mvbench_qa_template(question, candidates, answer_idx):
    question = f"Question: {question}\n"
    question += "Options:\n"
    answer = None
    for idx, c in enumerate(candidates):
        question += f"({chr(ord('A') + idx)}) {c}\n"
        if idx == answer_idx:
            answer = c
    assert answer
    question = question.rstrip()
    answer = f"({chr(ord('A') + answer_idx)}) {answer}"
    return question, answer


class DMPerceptionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        data_split: str = "train",
        data_subcategory: str = "",
        transform=None,
        frames_per_clip=16,
    ):
        assert data_split in DATA_SPLITS, f"Unknown data split: {data_split}\nAvailable values are {DATA_SPLITS}"

        self.data_format = None
        if not data_subcategory:
            task = "mc_question"
        elif ":" in data_subcategory:
            task, self.data_format = data_subcategory.split(":")
        else:
            task = data_subcategory

        assert task in TASKS, f"Unknown task: {task}\nAvailable values are {TASKS}"
        self.video_dir = os.path.join(data_dir, f"{data_split}_videos/videos")
        self.annot_fpath = os.path.join(data_dir, f"{data_split}_annotations/all_{data_split}.json")
        logger.info(f"Started reading {data_split} from {self.annot_fpath}")
        self.data = self.read_annotations(task)
        if not transform:
            self.transform = make_transforms(data_split == "train")
        else:
            self.transform = transform
        self.frames_per_clip = frames_per_clip

    def extract_example(self, annotation, meta_data):
        example = {"annotation": annotation, "metadata": meta_data}
        example["annotation"]["tag"] = ";".join(example["annotation"]["tag"])
        return example

    def read_annotations(self, task):
        with open(self.annot_fpath) as fi:
            raw_data = json.load(fi)

        ret = []
        for _, rd in raw_data.items():
            task_data = rd.get(task, None)
            if task_data:
                md = rd["metadata"]
                if isinstance(task_data, list):
                    for task_query in task_data:
                        ret.append(self.extract_example(task_query, md))
                else:
                    ret.append(self.extract_example(task_data, md))

        logger.info(f"Detected {len(ret)} examples.")
        return ret

    def format_data(self, example):
        if self.data_format is None:
            # We do this to avoid the weird default data collation by pytorch
            example["annotation"]["options"] = OPTIONS_DELIM.join(example["annotation"]["options"])
            return example
        elif self.data_format.lower() == "mvbench":
            # Converting the video tensor [T, C, H, W] --> [TC, H, W]
            vt = example["video"]
            if isinstance(vt, list):
                assert len(vt) == 1
                vt = vt.pop()
            new_shape = [-1] + [int(i) for i in vt.shape[-2:]]
            vt = vt.reshape(new_shape)
            converted_example = {
                "video": vt,
                "video_path": os.path.join(self.video_dir, example["metadata"]["video_id"]),
            }
            annot = example["annotation"]
            q, a = mvbench_qa_template(
                question=annot["question"],
                candidates=[o.strip() for o in annot["options"]],  # some option strings have trailing space
                answer_idx=annot["answer_id"],
            )
            converted_example["question"] = q
            converted_example["answer"] = a
            converted_example["task_type"] = annot["area"]
            return converted_example
        else:
            raise ValueError(f"Invalid format to convert: {self.data_format}")

    def load_video(self, filename: str):
        with open(filename, "rb") as fin:
            vr = decord.VideoReader(fin)

        # The linspace might cause slow motion if the number of frames is less than the requested number of frames.
        frame_ids = [round(fi) for fi in np.linspace(0, len(vr) - 1, self.frames_per_clip)]
        return vr.get_batch(frame_ids)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        annot = data_item["annotation"]
        metadata = data_item["metadata"]

        video_file_path = os.path.join(self.video_dir, f"{metadata['video_id']}.mp4")
        vid_frames = self.load_video(video_file_path)  # [T, H, W, C]
        if isinstance(vid_frames, decord.ndarray.NDArray):
            # No transformation is set via decord.bridge. Converting to numpy array.
            vid_frames = vid_frames.asnumpy()
        if self.transform:
            vid_frames = self.transform(vid_frames)

        return self.format_data({"metadata": metadata, "annotation": annot, "video": vid_frames})


def make_dm_perception_dataset(
    batch_size,
    transform,
    data_split,
    num_workers,
    world_size,
    rank,
    root_path,
    image_folder,
    data_subcategory,
):

    dataset = DMPerceptionDataset(
        data_dir=os.path.join(root_path, image_folder),
        data_split=data_split,
        data_subcategory=data_subcategory,
        transform=transform,
    )
    logger.info(f"DM Perception dataset created successfully. {len(dataset)} examples for {data_subcategory} task.")
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers)
    return dataset, dataloader, sampler
