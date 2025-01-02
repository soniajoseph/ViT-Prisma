# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import logging
import os
import random
from dataclasses import dataclass
from multiprocessing import Value

import braceexpand
import webdataset as wds
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, tar_file_expander, url_opener, valid_sample

from vit_prisma.vjepa_hf.src.datasets.utils.tokenizers import tokenize as open_clip_tokenizer


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


def count_samples(dataloader):
    logging.info("Counting samples in dataset from scratch")
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for data, _, _ in dataloader:
        images, texts = data
        n_batches += 1
        n_elements += len(images)
        if n_batches % 100 == 0:
            logging.info(f"num-batches counted [{n_batches}], elements [{n_elements}]")
        assert len(images) == len(texts)
    return n_elements, n_batches


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


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, isssue a warning, and continue."""
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


def filter_no_caption_or_no_image(sample):
    return ("txt" in sample) and ("png" in sample or "jpg" in sample)


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """
    Return function over iterator that groups key, value pairs into samples.
    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


class TextTransform:

    def __init__(self, tokenize_txt=False):
        self.tokenize_txt = tokenize_txt

    def __call__(self, text):
        if self.tokenize_txt:
            return open_clip_tokenizer([str(text)])[0]
        else:
            return str(text)


class ResampledShards(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(self, urls):
        super().__init__()
        urls = wds.shardlists.expand_urls(urls)
        random.shuffle(urls)
        self.urls = urls

    def __iter__(self):
        """Return an iterator  ver the shards."""
        for u in self.urls:
            yield dict(url=u)


def get_wds_dataset(
    batch_size,
    collator,
    url_input_shards,
    preprocess_img=None,
    epoch=0,
    world_size=1,
    num_workers=1,
    tokenize_txt=True,
    ipe=None,
    repeat=False,
):

    preprocess_txt = TextTransform(tokenize_txt)

    num_shards = len(list(braceexpand.braceexpand(url_input_shards)))
    assert num_shards >= num_workers * world_size, "number of shards must be >= total workers"
    global_batch_size = int(world_size * batch_size)

    epoch = SharedEpoch(epoch)
    dataset = wds.DataPipeline(
        ResampledShards(url_input_shards),
        # wds.SimpleShardList(url_input_shards),
        detshuffle(bufsize=2000, initial=1250, epoch=epoch),
        wds.split_by_node,
        wds.split_by_worker,
        # at this point we have an iterator over all the shards assigned to each worker at each node
        tarfile_to_samples_nothrow,
        wds.shuffle(bufsize=5000, initial=2250),
        wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png;jpeg;webp", text="txt"),
        wds.map_dict(image=preprocess_img, text=preprocess_txt),
        wds.to_tuple("image", "text"),
        wds.batched(batch_size, partial=False, collation_fn=collator),
    )

    # Manually set the number of iterations-per-epoch (ipe)
    if ipe is not None:
        num_samples = int(ipe * global_batch_size)
        if not repeat:
            dataset = dataset.with_epoch(ipe // num_workers)

    if repeat:
        dataset = dataset.repeat()

    dataloader = wds.WebLoader(
        dataset, batch_size=None, shuffle=False, num_workers=num_workers, persistent_workers=num_workers > 0
    )

    # If user has not manually set ipe, need to count samples in dataset
    if ipe is None:
        num_samples, _ = count_samples(dataloader)
        logging.info(f"Number of samples in dataset: {num_samples}")

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_samples // global_batch_size
    dataloader.num_samples = num_samples

    return dataset, DataInfo(dataloader=dataloader, shared_epoch=epoch)


def make_laion(
    root_path,
    image_folder,
    transform,
    batch_size,
    world_size,
    num_workers=10,
    collator=None,
    tokenize_txt=True,
    repeat_wds=False,
    rank=0,
    ipe=650,  # iterations-per-epoch (will be rounded down to multiple of 10)
):
    input_shards = os.path.join(root_path, image_folder)
    dataset, datainfo = get_wds_dataset(
        ipe=ipe,
        batch_size=batch_size,
        repeat=repeat_wds,
        collator=collator,
        url_input_shards=input_shards,
        preprocess_img=transform,
        tokenize_txt=tokenize_txt,
        world_size=world_size,
        num_workers=num_workers,
    )
    return dataset, datainfo.dataloader, datainfo
