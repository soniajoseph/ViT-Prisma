# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
import os
import random
import tempfile

import numpy as np
import torch
from airstore.client.airstore_tabular import AIRStorePathHandler
from decord import VideoReader, cpu
from iopath.common.file_io import PathManager
from PIL import Image

from vit_prisma.vjepa_hf.src.datasets.utils.dataloader import get_worker_info
from vit_prisma.vjepa_hf.src.utils.logging import get_logger
from vit_prisma.vjepa_hf.src.utils.profiler import create_profiler
from vit_prisma.vjepa_hf.src.utils.temp_ssd_dir import temp_ssd_dir

path_manager = PathManager()
path_manager.register_handler(AIRStorePathHandler())


logger = get_logger("Airstore Data Loader")

NUM_EXAMPLES = {
    "airstore://jepa_howto100m_no_user_data": 1_238_912,
    "airstore://jepa_kinetics_710_train_no_user_data": 733_191,
    "airstore://jepa_ssv2_no_user_data": 220_847,
    "airstore://yttemporal1b_no_user_data": 17_269_549,
    "airstore://ego4d_egoexo4d_no_user_data_rsc": 688_746,
    "airstore://jepa_ek100_256px_train_no_user_data": 67_217,
    "airstore://jepa_ek100_256px_valid_no_user_data": 9_668,
    "airstore://jepa_imagenet_no_user_data": 1_431_169,
    "airstore://muvigen_shutterstock_video_derived_resample_v1_no_user_data": 34_280_884,
}


class AIRStoreSampler(torch.utils.data.SequentialSampler):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.dataset = dataset

    def set_epoch(self, epoch, rebuild_iterators=False):
        self.dataset.set_epoch(epoch, lazy_shuffle=not rebuild_iterators)

    # For convenience in the train script
    def increase_epoch(self, rebuild_iterators=False):
        self.dataset.set_epoch(self.dataset.epoch + 1, lazy_shuffle=not rebuild_iterators)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    try:
        import accimage

        return accimage.Image(path)
    except OSError:  # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)


def make_airstore_dataset(
    data_paths,
    batch_size,
    split="train",
    frames_per_clip=8,
    fps=4,
    frame_step=4,
    duration=None,
    num_clips=1,
    random_clip_sampling=True,
    allow_clip_overlap=False,
    transform=None,
    shared_transform=None,
    world_size=1,
    rank=0,
    log_dir=None,
    datasets_weights=None,
    num_workers=8,
    pin_mem=True,
    persistent_workers=True,
    collator=None,
    drop_last=True,
    return_dict=False,
    dataloader_profiler_conf=None,
    dataset_fpcs=None,
):
    dataset = AIRStoreDataset(
        data_paths,
        split="train",
        transform=transform,
        shared_transform=shared_transform,
        frames_per_clip=frames_per_clip,
        dataset_fpcs=dataset_fpcs,
        fps=fps,
        frame_step=frame_step,
        duration=duration,
        num_clips=num_clips,
        random_clip_sampling=random_clip_sampling,
        allow_clip_overlap=allow_clip_overlap,
        world_size=world_size,
        rank=rank,
        log_dir=log_dir,
        datasets_weights=None,
        return_dict=return_dict,
        profiler_conf=dataloader_profiler_conf,
    )

    logger.info(f"Dataset in {split} split for rank {rank} / {world_size} created.")

    # Using the AIRStore as the source of data, we don't have control over sampling.
    # Shuffling is done in the AIRStore client. Thus we just process them sequentially.
    sampler = AIRStoreSampler(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )

    return dataset, data_loader, sampler


class AIRStoreDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_paths,
        split="train",
        transform=None,
        shared_transform=None,
        frames_per_clip=16,
        fps=None,
        frame_step=None,
        duration=None,
        num_clips=1,
        random_clip_sampling=True,
        allow_clip_overlap=False,
        world_size=1,
        rank=0,
        log_dir=None,
        datasets_weights=None,
        dataset_fpcs=None,
        return_dict=False,
        profiler_conf=None,
    ):

        if (int(fps is not None) + int(duration is not None) + int(frame_step is not None)) == 0:
            raise ValueError("Either fps or duration or frame_step must be specified")

        elif (int(fps is not None) + int(duration is not None) + int(frame_step is not None)) > 1:
            raise ValueError("Only one of fps or duration or frame_step can be specified")

        if isinstance(data_paths, str):
            data_paths = [data_paths]

        if dataset_fpcs is None:
            dataset_fpcs = [frames_per_clip for _ in data_paths]

        assert len(dataset_fpcs) == len(data_paths), "Frames per clip not properly specificed for AIRStore data paths"

        self.dataset_fpcs = {}
        for _path, _fpc in zip(data_paths, dataset_fpcs):
            self.dataset_fpcs[_path] = _fpc

        self.fps = fps
        self.frame_step = frame_step
        self.duration = duration
        self.num_clips = num_clips
        self.random_clip_sampling = random_clip_sampling
        self.allow_clip_overlap = allow_clip_overlap

        self.data_paths = data_paths
        self.split = split
        self.transform = transform
        self.shared_transform = shared_transform
        self.world_size = world_size
        self.rank = rank
        self.log_dir = log_dir
        self.return_dict = return_dict
        self.profiler = None
        self.profiler_conf = profiler_conf

        if datasets_weights is not None:
            assert len(datasets_weights) == len(data_paths)
            self.datasets_weights = datasets_weights
        else:
            self.datasets_weights = []
            for d in self.data_paths:
                assert d in NUM_EXAMPLES, f"Can not find dataset {d} in NUM_EXAMPLES"
                self.datasets_weights += [NUM_EXAMPLES[d] / len(self)]

        self.data_iters = dict()
        self.epoch = 0

    def get_profiler(self, profiler_conf):
        if profiler_conf is None:
            return None

        if self.profiler is not None:
            # Already created a profiler
            return self.profiler

        profiler_log_dir = os.path.join(self.log_dir, "profiler")
        profiler = create_profiler(profiler_conf, profiler_log_dir, self.rank)
        profiler.start()
        return profiler

    def get_data_iter(self, data_path: str, epoch: int = 0):
        num_workers, worker_id = get_worker_info()

        # split the dataset for each worker
        airstore_world_size = self.world_size * num_workers
        # each worker takes its split by its parent process rank and worker id
        airstore_rank = self.rank * num_workers + worker_id

        self.profiler = self.get_profiler(self.profiler_conf)

        data = path_manager.opent(
            path=data_path,
            world_size=airstore_world_size,
            rank=airstore_rank,
            use_decryption_server=True,
            seed=epoch,
        )
        logger.info(f"AIRStore worker initialized in job {self.rank}/{self.world_size} for {data_path}")
        data_iter = iter(data)
        self.data_iters[data_path] = data_iter
        return data_iter

    def set_epoch(self, epoch, lazy_shuffle=True):
        """Initiates a new data iterator with the epoch number as the seed.

        See this for setting epoch using seed on AIRStore: https://www.internalfb.com/intern/wiki/?fbid=653694039107701
        """
        self.epoch = epoch
        if not lazy_shuffle:
            # Also reset the data iterators with the new epoch number to shuffle them
            for data_path, data_iter in self.data_iters.items():
                data_iter.close()
                data_iter = self.get_data_iter(data_path, epoch)
                self.data_iters[data_path] = data_iter

    def __len__(self):
        n_total = 0
        for d in self.data_paths:
            assert d in NUM_EXAMPLES, f"Can not find dataset {d} in NUM_EXAMPLES"
            n_total += NUM_EXAMPLES[d]

        return n_total

    def load_video(self, video_path, fpc, is_img=False):

        if is_img:
            img = default_loader(video_path)
            buffer = np.stack([img for _ in range(fpc)], axis=0)
            indices = [np.arange(fpc) for _ in range(self.num_clips)]
            return buffer, indices

        try:
            vr = VideoReader(video_path, num_threads=-1, ctx=cpu(0))
            vr.seek(0)  # Go to start of video before sampling frames
        except Exception:
            return None, None

        fstp = self.frame_step
        if self.duration is not None or self.fps is not None:
            try:
                video_fps = math.ceil(vr.get_avg_fps())
            except Exception as e:
                logger.warning(e)

            if self.duration is not None:
                assert self.fps is None
                fstp = int(self.duration * video_fps / fpc)
            else:
                assert self.duration is None
                fstp = video_fps // self.fps

        assert fstp is not None and fstp > 0
        clip_len = int(fpc * fstp)

        # Partition video into equal sized segments and sample each clip
        # from a different segment
        partition_len = len(vr) // self.num_clips

        all_indices, clip_indices = [], []
        for i in range(self.num_clips):

            if partition_len > clip_len:
                # If partition_len > clip len, then sample a random window of
                # clip_len frames within the segment
                end_indx = clip_len
                if self.random_clip_sampling:
                    end_indx = np.random.randint(clip_len, partition_len)
                start_indx = end_indx - clip_len
                indices = np.linspace(start_indx, end_indx, num=fpc)
                indices = np.clip(indices, start_indx, end_indx - 1).astype(np.int64)
                # --
                indices = indices + i * partition_len
            else:
                # If partition overlap not allowed and partition_len < clip_len
                # then repeatedly append the last frame in the segment until
                # we reach the desired clip length
                if not self.allow_clip_overlap:
                    indices = np.linspace(0, partition_len, num=partition_len // fstp)
                    indices = np.concatenate(
                        (
                            indices,
                            np.ones(fpc - partition_len // fstp) * partition_len,
                        )
                    )
                    indices = np.clip(indices, 0, partition_len - 1).astype(np.int64)
                    # --
                    indices = indices + i * partition_len

                # If partition overlap is allowed and partition_len < clip_len
                # then start_indx of segment i+1 will lie within segment i
                else:
                    sample_len = min(clip_len, len(vr)) - 1
                    indices = np.linspace(0, sample_len, num=sample_len // fstp)
                    indices = np.concatenate(
                        (
                            indices,
                            np.ones(fpc - sample_len // fstp) * sample_len,
                        )
                    )
                    indices = np.clip(indices, 0, sample_len - 1).astype(np.int64)
                    # --
                    clip_step = 0
                    if len(vr) > clip_len:
                        clip_step = (len(vr) - clip_len) // (self.num_clips - 1)
                    indices = indices + i * clip_step

            clip_indices.append(indices)
            all_indices.extend(list(indices))

        try:
            buffer = vr.get_batch(all_indices).asnumpy()
        except Exception:
            return None, None

        return buffer, clip_indices

    def _get_next(self):
        next_iteration_data_path = random.choices(self.data_paths, weights=self.datasets_weights)[0]
        fpc = self.dataset_fpcs[next_iteration_data_path]
        data_iter = self.data_iters.get(next_iteration_data_path, None)
        if not data_iter:
            data_iter = self.get_data_iter(next_iteration_data_path, self.epoch)

        example = next(data_iter, None)
        if example is None:
            # If we reachthe end of the iterator, it means we finished a data epoch
            # This enables "infinite" iterator to avoid the compute of rebuilding them
            self.epoch += 1
            data_iter = self.get_data_iter(next_iteration_data_path, self.epoch)
            example = next(data_iter, None)
        assert example is not None
        return example, fpc

    def get_label(self, example):
        # TODO add label pulling later, either in AIRStore data or from matching with the annotation file.
        return 0

    def __getitem__(self, idx):
        buffer = None
        video_loading_tries = 0
        while buffer is None:
            example, fpc = self._get_next()
            video_id = example["video_id"] if "video_id" in example else "NA"

            temp_vid_file = tempfile.NamedTemporaryFile(dir=temp_ssd_dir())
            try:
                if "video" in example:
                    temp_vid_file.write(example["video"])
                elif "data" in example:
                    temp_vid_file.write(example["data"])
                elif "blob" in example:
                    temp_vid_file.write(example["blob"])
                temp_vid_file.flush()
                buffer, clip_indices = self.load_video(temp_vid_file.name, fpc=fpc, is_img="blob" in example)
            except Exception:
                buffer = None
            finally:
                temp_vid_file.close()

            if buffer is None:
                logger.warning(f"Failed to load video {video_id} on try {video_loading_tries}")
                video_loading_tries += 1

            NUM_RETRIES = 10
            if video_loading_tries > NUM_RETRIES:
                raise ValueError(f"Was not able to load video after {NUM_RETRIES} tries.")

        if self.shared_transform:
            buffer = self.shared_transform(buffer)

        # Splitting into n=self.num_clips clips
        buffer = [buffer[i * fpc : (i + 1) * fpc] for i in range(self.num_clips)]

        if self.transform:
            buffer = [self.transform(clip) for clip in buffer]

        label = self.get_label(example)

        if self.profiler is not None:
            self.profiler.step()

        if self.return_dict:
            return {
                "clips": buffer,
                "video_id": video_id,
                "label": label,
                # the following fields are for debugging; can be removed later
                "job_rank": self.rank,
            }
        else:
            # This is the format used by the src.datasets.video_dataset.VideoDataset
            return buffer, label, clip_indices
