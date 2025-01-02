# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import multiprocessing
import random
import warnings
from dataclasses import dataclass
from multiprocessing import Value

import numpy as np
import pandas as pd
import torch
import webdataset as wds
from decord import VideoReader, cpu
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle

multiprocessing.set_start_method("spawn", force=True)


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def get_dataset_size(shards_list):
    num_shards = len(shards_list)
    total_size = num_shards
    return total_size, num_shards


def filter_video(sample):
    condition = ("mp4" in sample) or ("webm" in sample)
    if not condition:
        warnings.warn("found sample with wrong extension")
    return condition


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, isssue a warning, and continue."""
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


class detshuffle(wds.PipelineStage):
    def __init__(
        self,
        bufsize=1000,
        initial=100,
        epoch=-1,
        seed=0,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        epoch = self.epoch.get_value()
        rng = random.Random()
        seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class TextVideoDecoder:
    def __init__(
        self,
        tokenizer=None,
        video_decoder=None,
    ):
        self.text_tokenizer = tokenizer
        self.video_decoder = video_decoder

    def __call__(self, sample):
        videos, offsets = self.video_decoder(sample)
        return {"video": videos, "indices": offsets, "text": "No caption"}


class SimpleVideoDecoder:
    def __init__(
        self,
        frames_per_clip,
        frame_step=4,
        duration=None,
        transform=None,
        discard_short_videos=False,
        decode_one_clip=False,
    ):
        self.frames_per_clip = frames_per_clip
        self.frame_step = frame_step
        self.duration = duration
        self.transform = transform
        self.discard_short_videos = discard_short_videos
        self.decode_one_clip = decode_one_clip

    def __call__(self, path):
        try:
            vr = VideoReader(path, num_threads=-1, ctx=cpu(0))
        except Exception as e:
            raise Exception(f"Error initializing VideoReader {path=} {e}")

        fpc = self.frames_per_clip
        fstp = self.frame_step
        duration = self.duration
        if duration is not None:
            try:
                fps = vr.get_avg_fps()
                fstp = int(duration * fps / fpc)
            except Exception as e:
                raise Exception(f"Could not get FPS for video {path=} {e}")
        clip_len = fpc * fstp

        discard_short = self.discard_short_videos
        if discard_short and (len(vr) < clip_len):
            raise Exception(f"Video is too short, will not decode frames {path=}")

        vr.seek(0)  # Go to start of video before sampling frames

        # If only decoding one clip and clip_len < video_len, then sample a
        # clip from a random window of the video
        if self.decode_one_clip and (clip_len < len(vr)):
            end_indx = np.random.randint(clip_len, len(vr))
            start_indx = end_indx - clip_len
            all_indices = np.arange(start_indx, end_indx, fstp).astype(np.int64)
        else:
            all_indices = np.arange(0, len(vr), fstp).astype(np.int64)

        buffer = vr.get_batch(all_indices).asnumpy()
        if self.transform is not None:
            buffer = self.transform(buffer)
        return buffer, all_indices


class VideoToClips(wds.PipelineStage):
    def __init__(self, frames_per_clip, transform=None):
        self.frames_per_clip = frames_per_clip
        self.transform = transform

    def run(self, dataset):
        fpc = self.frames_per_clip
        for d in dataset:
            video = d.get("video")
            indices = d.get("indices")
            text = d.get("text")
            num_clips = len(video) // fpc
            for i in range(num_clips):
                clip = video[i * fpc : (i + 1) * fpc]
                clip_indices = indices[i * fpc : (i + 1) * fpc]
                if self.transform is not None:
                    clip = self.transform(clip)
                yield {"video": [clip], "indices": clip_indices, "text": text}


class ResampledShards(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(self, urls, epoch):
        super().__init__()
        self.epoch = epoch
        self.urls = np.array(urls)
        logging.info("Done initializing ResampledShards")

    def __iter__(self):
        """Return an iterator  ver the shards."""
        epoch = self.epoch.get_value()
        gen = torch.Generator()
        gen.manual_seed(epoch)
        yield from self.urls[torch.randperm(len(self.urls), generator=gen)]


def get_video_wds_dataset(
    batch_size,
    collator,
    input_shards,
    video_decoder,
    video_to_clips,
    ipe=None,
    repeat=False,
    epoch=0,
    floor=False,
    world_size=1,
    num_workers=1,
):

    assert input_shards is not None
    num_samples, num_shards = get_dataset_size(input_shards)
    logging.info(f"Total number of shards across all data is {num_shards=}")

    global_batch_size = int(world_size * batch_size)

    video_sample_decoder = TextVideoDecoder(tokenizer=None, video_decoder=video_decoder)

    epoch = SharedEpoch(epoch=epoch)
    pipeline = [
        ResampledShards(input_shards, epoch=epoch),
        wds.shuffle(bufsize=int(2 * num_shards), initial=num_shards),
        wds.split_by_node,
        wds.split_by_worker,
        # at this point, we have an iterator over the shards assigned to each worker at each node
        wds.shuffle(bufsize=10000, initial=2950),
        wds.select(filter_video),
        wds.map(video_sample_decoder, handler=log_and_continue),
        video_to_clips,
        wds.to_tuple("video", "indices", "text"),
        wds.batched(batch_size, partial=False, collation_fn=collator),
    ]
    dataset = wds.DataPipeline(*pipeline)

    if ipe is not None:
        num_samples = int(ipe * global_batch_size)
    else:
        ipe = num_samples // global_batch_size

    if repeat:
        dataset = dataset.repeat()

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=False,
    )

    dataloader.num_batches = num_samples // global_batch_size
    dataloader.num_samples = num_samples

    return dataset, DataInfo(dataloader=dataloader, shared_epoch=epoch)


def make_webvid(
    data_paths,
    batch_size,
    transform,
    shared_transform=None,
    ipe=None,
    repeat=False,
    collator=None,
    num_frames=16,
    sampling_rate=4,
    duration=None,  # duration in seconds
    rank=0,
    num_workers=8,
    world_size=1,
    filter_short_videos=False,
    decode_one_clip=True,
):

    input_shards = []
    for d in data_paths:
        if d[-4:] == ".csv":
            data = pd.read_csv(d, header=None, delimiter=" ")
            input_shards += list(data.values[:, 0])
        elif d[-4:] == ".npy":
            data = np.load(d, allow_pickle=True)
            input_shards += list(map(lambda x: repr(x)[1:-1], data))

    video_decoder = SimpleVideoDecoder(
        frames_per_clip=num_frames,
        frame_step=sampling_rate,
        transform=shared_transform,
        duration=duration,
        decode_one_clip=decode_one_clip,
        discard_short_videos=filter_short_videos,
    )
    video_to_clips = VideoToClips(frames_per_clip=num_frames, transform=transform)

    dataset, datainfo = get_video_wds_dataset(
        batch_size=batch_size,
        collator=collator,
        input_shards=input_shards,
        ipe=None,
        repeat=repeat,
        epoch=0,
        world_size=world_size,
        num_workers=num_workers,
        video_decoder=video_decoder,
        video_to_clips=video_to_clips,
    )

    return dataset, datainfo.dataloader, datainfo
