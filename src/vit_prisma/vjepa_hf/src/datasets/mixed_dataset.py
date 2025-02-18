import pathlib
import pprint
import time
from bisect import bisect_right
from collections import defaultdict
from typing import List, Optional

import psutil
import torch

from vit_prisma.vjepa_hf.src.datasets.utils.dataloader import MonitoredDataset, NondeterministicDataLoader, get_worker_info
from vit_prisma.vjepa_hf.src.datasets.utils.weighted_sampler import DistributedWeightedSampler, MemoryEfficientDistributedWeightedSampler
from vit_prisma.vjepa_hf.src.utils.cluster import get_dataset_paths
from vit_prisma.vjepa_hf.src.utils.logging import get_logger

logger = get_logger("Mixed dataset")
pp = pprint.PrettyPrinter(indent=4)


def _log_memory_usage():
    process = psutil.Process()
    mem_mb = round(process.memory_info().rss / 1024 / 1024, 2)
    logger.info(f"Memory usage: {mem_mb} MB")


def get_dataset(dataset_type, **kwargs):

    # This is for convenience, we used to pass the dataset names and find their relative paths based on the cluster
    # This makes it easier for the user to specify the dataset by name (rather than path based on cluster)
    # as it is needed for the Dataset class.
    datasets = kwargs.pop("datasets", None)
    assert datasets is not None, "Dataset must be specified (needed for finding its path on cluster)"
    if dataset_type.lower() != "lvd":
        kwargs["data_paths"] = get_dataset_paths(datasets)

    logger.info(f"Creating dataset of type {dataset_type}")
    pp.pprint(kwargs)

    if dataset_type.lower() == "videodataset":
        from vit_prisma.vjepa_hf.src.datasets.video_dataset import VideoDataset

        return VideoDataset(**kwargs)
    elif dataset_type.lower() == "airstore_videodataset":
        from vit_prisma.vjepa_hf.src.datasets.airstore_dataset import AIRStoreDataset

        return AIRStoreDataset(**kwargs)
    elif dataset_type.lower() == "blobstore_videodataset":
        from vit_prisma.vjepa_hf.src.datasets.blobstore_dataset import BlobStoreDataset

        return BlobStoreDataset(**kwargs)
    elif dataset_type.lower() == "lvd":
        from vit_prisma.vjepa_hf.src.datasets.lvd_images import LVDImageDataset

        lvd_kwargs = {
            "root": get_dataset_paths(["LVD"])[0],
            "frames_per_clip": kwargs["frames_per_clip"],
            "transform": kwargs["transform"],
            "shared_transform": kwargs["shared_transform"],
            "subsets": datasets,
        }
        return LVDImageDataset(**lvd_kwargs)
    else:
        raise ValueError(f"Unknown dataset {dataset_type}")


def make_mixed_dataset(
    data_paths,
    datasets_weights=None,
    batch_size=1,
    frames_per_clip=8,
    dataset_fpcs=None,
    num_clips=1,
    frame_step=None,
    duration=None,
    fps=None,
    random_clip_sampling=True,
    allow_clip_overlap=False,
    shared_transform=None,
    transform=None,
    collator=None,
    rank=0,
    world_size=1,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
    deterministic=True,
    use_memory_efficient_weighted_sampler=False,
    log_dir=None,
    log_resource_util_data=False,
    log_stats_intervals=0,
):
    if not all(":" in dp for dp in data_paths):
        raise ValueError("All data paths be formatted as <dataset_type>:<dataset>")

    if datasets_weights is not None and len(datasets_weights) != len(data_paths):
        raise ValueError("Number of datasets and weights must match")

    if dataset_fpcs is not None:
        if frames_per_clip is not None:
            raise ValueError(f"Specify dataset_fpcs {dataset_fpcs} or frames_per_clip {frames_per_clip}, not both")
        if len(dataset_fpcs) != len(data_paths):
            raise ValueError("Number of datasets and frames per clip must match")

    datasets = []
    for i, dp in enumerate(data_paths):
        data_type, dataset_names = dp.split(":")

        frames_per_clip = dataset_fpcs[i] if dataset_fpcs is not None else frames_per_clip

        dataset_args = {
            "dataset_type": data_type,
            "datasets": dataset_names.split(","),
            "frames_per_clip": frames_per_clip,
            "frame_step": frame_step,
            "duration": duration,
            "fps": fps,
            "num_clips": num_clips,
            "random_clip_sampling": random_clip_sampling,
            "allow_clip_overlap": allow_clip_overlap,
            "transform": transform,
            "shared_transform": shared_transform,
        }
        if data_type.lower() in ("blobstore_videodataset", "airstore_videodataset"):
            dataset_args["rank"] = rank
            dataset_args["world_size"] = world_size
            dataset_args["fps"] = fps
            ds_fpcs = dataset_fpcs[i] if dataset_fpcs is not None else None
            if ds_fpcs is not None and isinstance(ds_fpcs, int):
                ds_fpcs = [ds_fpcs]
            elif ds_fpcs is not None:
                assert isinstance(ds_fpcs, list)
                assert len(ds_fpcs) == len(dataset_names.split(","))
            dataset_args["dataset_fpcs"] = ds_fpcs

        datasets.append(get_dataset(**dataset_args))

    log_dir_path = pathlib.Path(log_dir) if log_dir else None
    if log_dir_path:
        log_dir_path.mkdir(parents=True, exist_ok=True)

    mixed_dataset = MixedDataset(
        datasets=datasets,
        data_paths=data_paths,
        weights=datasets_weights,
        log_stats_intervals=log_stats_intervals,
        world_size=world_size,
        rank=rank,
        memory_efficient_sampler=use_memory_efficient_weighted_sampler,
        log_stats_dir=log_dir_path / "stats" if log_dir_path else None,
    )

    if log_resource_util_data:
        if log_dir_path is None:
            raise ValueError(
                "log_dir must be specified when log_resource_util_data is True"
                " (this happens in the train script, check there)"
            )

        resource_monitoring_dir = log_dir_path / "resource_monitoring"
        resource_monitoring_dir.mkdir(parents=True, exist_ok=True)

        # Worker ID will replace '%w'
        resource_log_filename = resource_monitoring_dir / f"resource_file_{rank}_%w.csv"
        mixed_dataset = MonitoredDataset(
            dataset=mixed_dataset,
            log_filename=str(resource_log_filename),
            log_interval=10.0,
            monitor_interval=5.0,
        )

    logger.info("Mixed dataset created")
    _log_memory_usage()
    if datasets_weights is not None:
        if use_memory_efficient_weighted_sampler:
            logger.info("Using memory efficient weighted sampler")
            dist_sampler = MemoryEfficientDistributedWeightedSampler(
                mixed_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
            )
        else:
            dist_sampler = DistributedWeightedSampler(
                mixed_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
            )
    else:
        dist_sampler = torch.utils.data.distributed.DistributedSampler(
            mixed_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )

    if deterministic:
        data_loader = torch.utils.data.DataLoader(
            mixed_dataset,
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
            mixed_dataset,
            collate_fn=collator,
            sampler=dist_sampler,
            batch_size=batch_size,
            drop_last=drop_last,
            pin_memory=pin_mem,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0) and persistent_workers,
        )
    logger.info("Mixed dataset unsupervised data loader created")
    _log_memory_usage()

    return mixed_dataset, data_loader, dist_sampler


class MixedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        datasets: List[torch.utils.data.Dataset],
        data_paths: List[str],
        weights: Optional[List[float]] = None,
        memory_efficient_sampler=False,
        log_stats_dir: Optional[pathlib.Path] = None,
        log_stats_intervals: Optional[int] = 0,
        world_size: Optional[int] = 1,
        rank: Optional[int] = 0,
    ):
        assert len(datasets) > 0, "No dataset provided"
        assert weights is None or len(datasets) == len(weights), "Number of datasets and weights must match"
        self.datasets = datasets
        self.data_paths = data_paths
        self.world_size = world_size
        self.rank = rank
        self.log_stats_dir = log_stats_dir if log_stats_dir is not None else None
        if self.log_stats_dir:
            self.log_stats_dir.mkdir(parents=True, exist_ok=True)

        # Sample indices will keep tuples of (dataset_idx, sample_idx)
        # The first index is the dataset index, the second index is the sample index inside the latter dataset.
        self.cum_sample_indices = [0]
        if memory_efficient_sampler:
            self.dataset_weights = [] if weights is not None else None
        else:
            self.sample_weights = [] if weights is not None else None

        for dataset_idx in range(len(datasets)):
            st = time.time()
            curr_dataset = datasets[dataset_idx]
            assert isinstance(
                curr_dataset, torch.utils.data.Dataset
            ), f"Dataset {dataset_idx} is instance of {type(curr_dataset)}; not a torch dataset"
            self.cum_sample_indices.append(self.cum_sample_indices[-1] + len(curr_dataset))

            if weights is not None:
                logger.info(f"Setting dataset {dataset_idx} weights {weights[dataset_idx]}")
                if memory_efficient_sampler:
                    # Memory efficient sampler first chooses a dataset based on the weights; and then samples from it.
                    # Therefore, we do NOT need to adjust the dataset weight by its size.
                    self.dataset_weights.append(weights[dataset_idx])
                else:
                    curr_dataset_weight = weights[dataset_idx] / len(curr_dataset)
                    self.sample_weights.extend([curr_dataset_weight] * len(curr_dataset))

            # Logging info
            logger.info(
                f"dataset {dataset_idx + 1}/{len(datasets)} ({self.data_paths[dataset_idx]})"
                f" with {len(curr_dataset)} examples added in {time.time() - st} seconds."
            )

        self.log_stats_intervals = log_stats_intervals or 0
        self.subdataset_counts = defaultdict(int)
        self.subdataset_loading_times = defaultdict(float)

    def __len__(self):
        return self.cum_sample_indices[-1]

    def _report_dataset_stats(self):
        # Generating the count of the exampls used from each dataset.
        num_workers, worker_id = get_worker_info()
        log_lines = [f"Number of processed examples {sum(self.subdataset_counts.values())}"]
        for dataset_idx, ds_name in enumerate(self.data_paths):
            examples_count = self.subdataset_counts[dataset_idx]
            avg_loading_time = self.subdataset_loading_times[dataset_idx] / examples_count
            log_lines.append(
                f"\tDataset {dataset_idx} ({ds_name}): {examples_count},"
                f" average loading time: {avg_loading_time} seconds"
            )

        if self.log_stats_dir:
            # Dumping the stats to its corresponding file
            log_filename = self.log_stats_dir / f"stats_{self.rank}_{worker_id}"
            log_lines = log_lines
            with open(log_filename, "a") as log_file:
                log_file.write("\n".join(log_lines))
                log_file.write("\n")
        else:
            # Dumping in the main log stream
            workers_info = f"job rank {self.rank} / {self.world_size}, dataloader worker {worker_id} / {num_workers}"
            log_lines = [f"Number of examples from each dataset for {workers_info}:"] + log_lines
            logger.info("\n".join(log_lines))

    def get_dataset_index(self, index):
        # Getting the index of the dataset that the sample belongs to
        assert index < self.cum_sample_indices[-1], f"{index=}, {self.cum_sample_indices[-1]=}"
        insertion_point = bisect_right(self.cum_sample_indices, index)
        return insertion_point - 1

    def __getitem__(self, index):
        # First getting the dataset that we want to sample from
        dataset_idx = self.get_dataset_index(index)
        assert dataset_idx < len(self.datasets)
        dataset = self.datasets[dataset_idx]

        sample_idx_in_dataset = index - self.cum_sample_indices[dataset_idx]
        assert sample_idx_in_dataset < len(dataset)

        # Then getting the sample from its corresponding dataset
        if self.log_stats_intervals:
            self.subdataset_counts[dataset_idx] += 1
            fetch_start_time = time.time()

            sample = dataset[sample_idx_in_dataset]

            self.subdataset_loading_times[dataset_idx] += time.time() - fetch_start_time
            if sum(self.subdataset_counts.values()) % self.log_stats_intervals == 0:
                self._report_dataset_stats()
        else:
            sample = dataset[sample_idx_in_dataset]

        return sample
