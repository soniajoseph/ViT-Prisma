import bisect
import io
import json
import mmap
import os
import pathlib
from typing import Any, Callable, List, Optional

import numpy as np
import torch
import torchvision
from braceexpand import braceexpand
from PIL import Image
from tqdm import tqdm

from vit_prisma.vjepa_hf.src.datasets.utils.dataloader import MonitoredDataset, NondeterministicDataLoader
from vit_prisma.vjepa_hf.src.utils.cluster import get_dataset_path
from vit_prisma.vjepa_hf.src.utils.logging import get_logger

logger = get_logger("LVD images dataset")

# This list is based on the available directories at (RSC) /checkpoint/dino/datasets/LaViDa-20221031-blurred
AVAILABLE_SUBSETS = (
    "LaViDa-patchwork-v3-part2a-ImageNet22k-retrieval",
    "LaViDa-patchwork-v3-part2b-ImageNet22k-retrieval",
    "LaViDa-patchwork-v3-part2c-ImageNet22k-retrieval",
    "LaViDa-patchwork-v3-part3b-GLDv2-retrieval",
    "LaViDa-patchwork-v3-part4a-ADE20KChallengeData2016-TRAIN-retrieval",
    "LaViDa-patchwork-v3-part4b-AmsterTime-NEW-retrieval",
    "LaViDa-patchwork-v3-part4c-AmsterTime-OLD-retrieval",
    "LaViDa-patchwork-v3-part4d-Caltech101_Split-TRAIN-retrieval",
    "LaViDa-patchwork-v3-part4e-Cityscapes-TRAIN-retrieval",
    "LaViDa-patchwork-v3-part4f-CUB_200_2011-TRAIN-retrieval",
    "LaViDa-patchwork-v3-part4g-DTD-TRAIN1-retrieval",
    "LaViDa-patchwork-v3-part4h-FGVC_Aircraft-TRAIN-retrieval",
    "LaViDa-patchwork-v3-part4i-Flowers102-TRAIN-retrieval",
    "LaViDa-patchwork-v3-part4j-Food101-TRAIN-retrieval",
    "LaViDa-patchwork-v3-part4k-Met-TRAIN-retrieval",
    "LaViDa-patchwork-v3-part4l-NYU-TRAIN-retrieval",
    "LaViDa-patchwork-v3-part4m-Pets-TRAINVAL-retrieval",
    "LaViDa-patchwork-v3-part4n-RevisitingOxford-BASE-retrieval",
    "LaViDa-patchwork-v3-part4o-RevisitingParis-BASE-retrieval",
    "LaViDa-patchwork-v3-part4p-StanfordCars-TRAIN-retrieval",
    "LaViDa-patchwork-v3-part4q-SUN397-TRAIN1-retrieval",
    "LaViDa-patchwork-v3-part4r-SUNRGBD-TRAIN-retrieval",
    "LaViDa-patchwork-v3-part4s-VOC2007-TRAIN-retrieval",
    "LaViDa-patchwork-v3-part4t-VOC2012Seg-TRAINAUG-retrieval",
    "LaViDa-patchwork-v3-part4u-KITTI-EIGEN_TRAIN-retrieval",
    "LaViDa-patchwork-v3-part5-ImageNet-retrieval_41M",
    "LaViDa-patchwork-v3-part1-ImageNet22k",
    "LaViDa-patchwork-v3-part3a-GLDv2",
    "LaViDa-patchwork-v3-part3c-Mapillary_SLS",
    "LaViDa-patchwork-v3-part3-GLDv2+GLDv2-retrieval+Mapillary_SLS",
)


def read_josn_file(fpath: str):
    with open(fpath, "r") as fi:
        return json.load(fi)


def make_lvd_dataset(
    subsets=None,
    frames_per_clip=8,
    batch_size=4,
    transform=None,
    shared_transform=None,
    rank=0,
    world_size=1,
    collator=None,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
    deterministic=True,
    log_dir=None,
):
    root = get_dataset_path("LVD")

    dataset = LVDImageDataset(
        root=root,
        frames_per_clip=frames_per_clip,
        shared_transform=shared_transform,
        transform=transform,
        subsets=subsets,
    )
    logger.info("LVD Image dataset created")

    log_dir = pathlib.Path(log_dir) if log_dir else None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        # Worker ID will replace '%w'
        resource_log_filename = log_dir / f"resource_file_{rank}_%w.csv"
        dataset = MonitoredDataset(
            dataset=dataset,
            log_filename=str(resource_log_filename),
            log_interval=10.0,
            monitor_interval=5.0,
        )

    # We suppose there is no weighted sampling here.
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )

    if deterministic:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=collator,
            sampler=dist_sampler,
            batch_size=batch_size,
            drop_last=drop_last,
            pin_memory=pin_mem,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0) and persistent_workers,
        )
    else:
        data_loader = NondeterministicDataLoader(
            dataset,
            collate_fn=collator,
            sampler=dist_sampler,
            batch_size=batch_size,
            drop_last=drop_last,
            pin_memory=pin_mem,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0) and persistent_workers,
        )
    logger.info("LVD Image unsupervised data loader created")

    return dataset, data_loader, dist_sampler


class LVDImageDataset(torchvision.datasets.VisionDataset):
    def __init__(
        self,
        root: str,
        frames_per_clip: int = 4,
        shared_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        subsets: Optional[List[str]] = None,
    ):
        super().__init__(root=root, transform=transform)
        self.shared_transform = shared_transform
        self._frames_per_clip = frames_per_clip

        self._index_subset_dir = []
        self._tarball_shard_names = []
        self._entries = []
        self._index_cum_sum = []
        if subsets is None or (len(subsets) == 1 and subsets[0].lower() == "all"):
            subsets = AVAILABLE_SUBSETS
        self._validate_subset_by_directory(subset_names=subsets)
        self.populate_data_indices(subsets)

    def _validate_subset_by_directory(self, subset_names: List[str]):
        for ssn in subset_names:
            assert ssn in AVAILABLE_SUBSETS, f"Subset name {ssn} is not valid."
            assert os.path.exists(os.path.join(self.root, ssn)), f"Can find the directory for {ssn} data subset."

    def get_metadata(self, dirname: str):
        metadata_fpath = os.path.join(self.root, dirname, "metadata.json")
        return read_josn_file(metadata_fpath)

    def populate_data_indices(self, subsets: List[str]):
        for subset_dir_name in tqdm(subsets, desc="Populating LVD datset"):
            self._index_subset_dir.append(subset_dir_name)

            metadata = self.get_metadata(subset_dir_name)
            self._tarball_shard_names.append(list(braceexpand(metadata["shards"])))
            entries_path = os.path.join(self.root, subset_dir_name, metadata["entries"])
            self._entries.append(np.load(entries_path))
            num_samples_in_subset = metadata["samples"]
            if len(self._index_cum_sum) > 0:
                new_cum_sum = self._index_cum_sum[-1] + num_samples_in_subset
                self._index_cum_sum.append(new_cum_sum)
            else:
                self._index_cum_sum.append(num_samples_in_subset)

    def subset_index(self, index: int):
        datset_idx = bisect.bisect_right(self._index_cum_sum, index)
        assert datset_idx < len(self._index_cum_sum), "Index requsted for outside array."
        if datset_idx == 0:
            sample_idx_in_dataset = index
        else:
            sample_idx_in_dataset = index - self._index_cum_sum[datset_idx - 1]
        return datset_idx, sample_idx_in_dataset

    def _get_tarball_path(self, index: int) -> str:
        entry = self.get_entry(index=index)
        tarball_id = entry["tarball_id"]
        return os.path.join(self._shard_paths[tarball_id])

    def _get_tarball_image_data(self, index: int) -> bytes:
        datset_idx, sample_idx = self.subset_index(index=index)
        entry = self._entries[datset_idx][sample_idx]

        tarball_id = entry["tarball_id"]
        tarball_path = os.path.join(
            self.root,
            self._index_subset_dir[datset_idx],
            self._tarball_shard_names[datset_idx][tarball_id],
        )

        jpg_start_offset = entry["jpg_start_offset"]
        jpg_end_offset = entry["jpg_end_offset"]

        with open(tarball_path) as f:
            with mmap.mmap(fileno=f.fileno(), length=0, access=mmap.ACCESS_READ) as m:
                return m[jpg_start_offset:jpg_end_offset]

    def __getitem__(self, index: int) -> Any:
        img_data = self._get_tarball_image_data(index)
        img = Image.open(io.BytesIO(img_data)).convert(mode="RGB")
        buffer = np.stack([img for _ in range(self._frames_per_clip)], axis=0)
        clip_indices = np.arange(self._frames_per_clip)

        if self.shared_transform:
            buffer = self.shared_transform(buffer)

        if self.transform:
            buffer = self.transform(buffer)

        # Enable backwards compatibilty for sampling multiple clips per video
        buffer = [buffer]
        clip_indices = [clip_indices]

        return buffer, 0, clip_indices

    def __len__(self) -> int:
        assert len(self._index_cum_sum) > 0, "No data subset was added."
        return self._index_cum_sum[-1]
