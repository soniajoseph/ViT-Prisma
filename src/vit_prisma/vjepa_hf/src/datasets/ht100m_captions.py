import json
import random
import sqlite3
import tempfile
from functools import lru_cache

import numpy as np
import torch
from decord import VideoReader, cpu

from vit_prisma.vjepa_hf.src.datasets.airstore_dataset import NUM_EXAMPLES, AIRStoreDataset, AIRStoreSampler, temp_ssd_dir
from vit_prisma.vjepa_hf.src.utils.logging import get_logger

logger = get_logger("HT100M captions Data Loader (AIRStore)")

VIDEO_DATA_PATH = "airstore://howto100m_no_user_data"
RAW_CAPTIONS_PATH = {
    "rsc": "/checkpoint/jepa/data/HowTo100M/ht100_raw_captions.db",
    "h2": "/checkpoint/komeili/HT100M/raw_caption.db",
}


# We are caching this because it is not an object that could be pickled.
# Therefore the dataloader can not pickle it for multiple workers.
# Getting the db connection everytime and caching that at this level avoids the pickling issue.
@lru_cache
def connect_captions_db(fpath):
    logger.info(f"Connrecting to the captions DB at {fpath}")
    db = sqlite3.connect(f"file:{fpath}?mode=ro", uri=True)
    logger.info("Captions DB connetio succcessful.")
    return db


def collator_fn(batch):
    batch = torch.utils.data.default_collate(batch)
    # For some reason the default collate puts some of the values in a list.
    # Here we pop them out to make sure we got better view.
    # NOTE: we have "video_buffer" twice here because we have it inside a nested list.
    for k in ("clip_frames", "captions", "video_buffer", "video_buffer"):
        assert len(batch[k]) == 1, f"{k} should be a single value, but got {len(batch[k])} values."
        batch[k] = batch[k][0]
    return batch


def make_ht100m_captions_airstore(
    batch_size,
    split="train",
    frames_per_clip=8,
    frame_step=4,
    num_clips=1,
    random_clip_sampling=True,
    transform=None,
    shared_transform=None,
    world_size=1,
    rank=0,
    num_workers=8,
    pin_mem=True,
    persistent_workers=True,
    collator=None,
    drop_last=True,
):
    dataset = HowTo100MCaptionsAIRStoreDataset(
        VIDEO_DATA_PATH,
        split="train",
        transform=transform,
        shared_transform=shared_transform,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        num_clips=num_clips,
        random_clip_sampling=random_clip_sampling,
        world_size=world_size,
        rank=rank,
    )

    logger.info(f"Dataset in {split} split for rank {rank} / {world_size} created.")

    # Using the AIRStore as the source of data, we don't have control over sampling.
    # Shuffling is done in the AIRStore client. Thus we just process them sequentially.
    sampler = AIRStoreSampler(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator or collator_fn,
        sampler=sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )

    return dataset, data_loader, sampler


class HowTo100MCaptionsAIRStoreDataset(AIRStoreDataset):
    def __init__(
        self,
        data_path,
        split="train",
        transform=None,
        shared_transform=None,
        frames_per_clip=16,
        frame_step=4,
        num_clips=1,
        random_clip_sampling=True,
        world_size=1,
        rank=0,
        duration=None,  # duration in seconds
    ):
        self.data_paths = [data_path]
        self.split = split
        self.transform = transform
        self.shared_transform = shared_transform
        self.frames_per_clip = frames_per_clip
        self.frame_step = frame_step
        self.num_clips = num_clips
        self.duration = duration
        self.random_clip_sampling = random_clip_sampling
        self.world_size = world_size
        self.rank = rank
        random.seed(42 + self.rank)
        self.epoch = 0
        self.data_iters = dict()
        self.datasets_weights = [1]

    def __len__(self):
        return NUM_EXAMPLES[VIDEO_DATA_PATH]

    def load_video_with_captions(self, vpath, captions):
        try:
            vr = VideoReader(vpath, num_threads=-1, ctx=cpu(0))
        except Exception:
            return None, None

        fpc = self.frames_per_clip
        fps = vr.get_avg_fps()
        fstp = self.frame_step
        if self.duration is not None:
            fstp = int(self.duration * fps / fpc)
        clip_dur = fpc * fstp / fps

        vr.seek(0)  # Go to start of video before sampling frames

        num_captions = len(captions["start"])
        assert num_captions == len(captions["end"]) and num_captions == len(
            captions["text"]
        ), "Mismatch between text and timespans in captions."

        if self.random_clip_sampling:
            span_ids = random.choices(list(range(num_captions)), k=self.num_clips)
        else:
            # Choosing clips from equally spaced intervals between them
            span_ids = np.linspace(start=0, stop=num_captions, num=self.num_clips + 1)
            # Shifting sample clip to the next one in each epcoh, with rotation to the beginning from the end.
            span_ids = [(self.epoch + round(sp)) % num_captions for sp in span_ids]

        # Making sure there is enough frames for each clip
        frame_indices = []
        texts = []
        for spi in span_ids:
            curr_text = captions["text"][spi]
            start_time, end_time = captions["start"][spi], captions["end"][spi]
            offset = 0
            while (end_time - start_time) < clip_dur and offset < (num_captions - spi):
                # Keep expanding into the next span of subtitle until we reach the desired clip_dur or the end of
                # video. This may introduce some overlap between the videos, if the requested clip_dur is long.
                offset += 1
                new_end_idx = spi + offset
                end_time = captions["end"][new_end_idx]
                curr_text = f"{curr_text} {captions['text'][new_end_idx]}"

            # This frame spaceing may result in some slow down in frame rate,
            # but we do this to keep the frames relevant to the caption
            frame_indices.append(
                np.round(np.linspace(start=start_time * fstp, stop=end_time * fstp, num=fpc, endpoint=False))
            )
            texts.append(curr_text)
        try:
            buffer = vr.get_batch(sum([indices.tolist() for indices in frame_indices], [])).asnumpy()
        except Exception:
            return None, None, None

        return buffer, frame_indices, texts

    def get_caption_data(self, vidoe_id):
        db = connect_captions_db(RAW_CAPTIONS_PATH["rsc"])
        query = f"SELECT video_id, caption_data FROM captions WHERE video_id='{vidoe_id}'"
        query_output = db.execute(query)
        _, caption_data = next(query_output)
        assert next(query_output, None) is None, "There must be one row of caption data per video."
        return json.loads(caption_data)

    def __getitem__(self, idx):
        buffer = None
        video_loading_tries = 0
        while buffer is None:
            example = self._get_next()
            video_id = example["video_id"]
            captions = self.get_caption_data(video_id)

            temp_vid_file = tempfile.NamedTemporaryFile(dir=temp_ssd_dir())
            try:
                temp_vid_file.write(example["video"])
                temp_vid_file.flush()
                buffer, frame_indices, texts = self.load_video_with_captions(temp_vid_file.name, captions)
            except Exception:
                buffer = None
            finally:
                temp_vid_file.close()

            if buffer is None:
                logger.warning(f"Failed to load video {video_id} on try {video_loading_tries}")
                video_loading_tries += 1

            NUM_RETRIES = 3
            if video_loading_tries > NUM_RETRIES:
                raise ValueError(f"Was not able to load video after {NUM_RETRIES} tries.")

        # Video Transforms
        if self.shared_transform:
            buffer = self.shared_transform(buffer)

        fpc = self.frames_per_clip
        buffer = [buffer[i * fpc : (i + 1) * fpc] for i in range(self.num_clips)]

        if self.transform:
            buffer = [self.transform(clip) for clip in buffer]

        return {
            "video_buffer": buffer,
            "clip_frames": frame_indices,
            "captions": texts,
            "video_id": video_id,
        }
