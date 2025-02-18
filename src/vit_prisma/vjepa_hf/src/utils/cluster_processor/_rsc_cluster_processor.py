import os
import shutil
import time
from pathlib import Path
from typing import Optional

from airstore.blobstore import BlobStore
from airstore.client.airstore_tabular import AIRStorePathHandler
from iopath.common.file_io import PathManager

from datasets.airstore_dataset import temp_ssd_dir
from vit_prisma.vjepa_hf.src.datasets.blobstore_dataset import (
    VIDEO_PATH_DB,
    DBEntry,
    get_db,
    get_item_from_rowid,
    get_num_examples_from_sqlite_db,
)
from vit_prisma.vjepa_hf.src.utils.logging import get_logger

from ._cluster_processor import ClusterProcessor
from ._local_file_dataset import LocalFileDataset

DB_WAIT_TIME = 10.0


class BlobstoreIterator:
    """Taken from vit_prisma.vjepa_hf.src.datasets.blobstore_dataset.BlobstoreDataset"""

    def __init__(
        self,
        data_path,
        offset: int,
        limit: int,
        blobstore_client,
        id_key: str = "video_id",
        data_key: str = "video",
        bucket: str = "jepa",
        rank: int = 0,
    ):
        self.logger = get_logger(self.__class__.__name__)
        self._blobstore_client = blobstore_client
        self.id_key = id_key
        self.data_key = data_key
        self.bucket = bucket

        orig_db_path = VIDEO_PATH_DB[data_path]
        basename = os.path.basename(orig_db_path)
        db_path = os.path.join(temp_ssd_dir(), basename)
        touch_name = os.path.join(temp_ssd_dir(), basename + ".touch")
        if rank == 0:
            self.logger.info(f"Copying db file {orig_db_path} to local {db_path}")
            if os.path.exists(db_path):
                raise RuntimeError(f"DB file {db_path} already exists. Duplicate name?")
            shutil.copyfile(orig_db_path, db_path)
            # use a touch file to prevent other ranks from reading copied file too early
            Path(touch_name).touch()
        else:
            # buffer for copying db
            while not os.path.exists(touch_name):
                time.sleep(DB_WAIT_TIME)
                self.logger.debug(f"Waiting for {db_path} to be copied")

        num_examples = get_num_examples_from_sqlite_db(db_path)
        db_entry = DBEntry(db_path=db_path, count=num_examples)
        self.samples_chunks = [db_entry]

        # we'll just set these in a super hacky way, store all the chunks and do an offset
        if limit < 0:
            self.indices = list(range(offset, num_examples))
        else:
            self.indices = list(range(offset, min(offset + limit, num_examples)))

        self.logger.info(
            f"Initializing BlobStoreIterator with offset {offset}, limit {limit}, num_examples {num_examples}, "
            f"yielding length {len(self.indices)}"
        )

    def __len__(self):
        return len(self.indices)

    def _get_blobstore_path(self, idx):
        # 1) find the index of the chunk that the video belongs to, and the index of the video in that chunk.
        chunk_start_idx = 0
        for chunk in self.samples_chunks:
            if isinstance(chunk, DBEntry):
                chunk_length = chunk.count
            else:
                # Probably an overkill to assert here multiple times,
                # specially since it happens in __len__ as well; but just in case.
                assert isinstance(chunk, list)
                chunk_length = len(chunk)

            chunk_end_idx = chunk_start_idx + chunk_length
            if idx < chunk_end_idx:
                break

            chunk_start_idx = chunk_end_idx

        # 2) return the path of the video in the chunk
        item_index_in_chunk = idx - chunk_start_idx
        if isinstance(chunk, DBEntry):
            db = get_db(chunk.db_path)
            db_row_id = item_index_in_chunk + 1  # row_id starts from 1
            return get_item_from_rowid(db, db_row_id)
        else:
            return chunk[item_index_in_chunk]

    def __getitem__(self, local_idx):
        idx = self.indices[local_idx]

        path = self._get_blobstore_path(idx)
        ret = self._blobstore_client.list_objects(
            bucket=self.bucket,
            path=path,
            continuation_token=None,
            max_keys=100,
        )
        blob = ret.blobs[0]
        return {
            self.id_key: path.split("/")[-1],
            self.data_key: self._blobstore_client.get(key=f"{path}/{blob.name}", bucket=self.bucket),
        }


class RSCClusterProcessor(ClusterProcessor):
    """Generic class for parallel cluster processing tasks on the RSC

    RSCClusterProcessor is an extension to ClusterProcessor with extra
    utilities for running on the RSC. Primarily, it builds your AIRStore or
    BlobStore source dataset for you based on ``dataset_type``.

    See src.utils.cluster_processor.ClusterProcessor for all details on
    parameters not documented below.

    Args:
        dataset_type: One of "airstore" or "blobstore". "local" is not fully
            implemented and tested at the moment.
        input_dataset: A string specifying the AIRStore or BlobStore source
            path.
        blobstore_bucket: Which BlobStore bucket to use. Only utilized if
            "dataset_type" is "blobstore".
    """

    dataset_type: str
    input_dataset: str
    blobstore_client: Optional[BlobStore] = None
    blobstore_bucket: str = "jepa"

    def __init__(
        self,
        job_offset: int,
        job_limit: int,
        dataset_type: str,
        input_dataset: str,
        blobstore_bucket: str = "jepa",
        progress_offset: int = 0,
        task_sleep_interval: float = 60.0,
        min_time_per_iter: float = 0.0,
    ):
        super().__init__(
            job_offset=job_offset,
            job_limit=job_limit,
            progress_offset=progress_offset,
            task_sleep_interval=task_sleep_interval,
            min_time_per_iter=min_time_per_iter,
        )
        self.dataset_type = dataset_type
        self.input_dataset = input_dataset
        self.blobstore_client = None
        self.blobstore_bucket = blobstore_bucket

    def _get_airstore_iterator(self, data_path: str, offset: int = 0, limit: int = -1):
        self.path_manager = PathManager()
        self.path_manager.register_handler(AIRStorePathHandler())
        self.logger.info(f"Limiting to {limit} videos for the AIRStore iterator")
        return self.path_manager.opent(
            path=data_path,
            world_size=1,
            rank=0,
            offset=offset,
            limit=limit,
            use_decryption_server=True,
            seed=0,  # no need to shuffle
        )

    def _get_blobstore_client(self):
        blobstore = BlobStore()
        blobstore.start()
        if blobstore is None:
            raise RuntimeError("Failed to start BlobStore client")
        return blobstore

    def __del__(self) -> None:
        if self.blobstore_client is not None:
            self.blobstore_client.cleanup()

    def setup_data(self, rank_offset: int, rank_limit: int):
        if self.dataset_type is None:
            raise RuntimeError(
                f"dataset_type not set. You must define dataset_type in __init__ of {self.__class__.__name__}"
            )
        if self.input_dataset is None:
            raise RuntimeError(
                f"input_dataset not set. You must define input_dataset in __init__ of {self.__class__.__name__}"
            )

        self.blobstore_client = None
        if self.dataset_type == "airstore":
            data_iter = self._get_airstore_iterator(
                self.input_dataset,
                offset=rank_offset + self.progress_offset,
                limit=rank_limit - self.progress_offset,
            )
        elif self.dataset_type == "blobstore":
            assert self.rank is not None
            self.blobstore_client = self._get_blobstore_client()
            data_iter = BlobstoreIterator(
                self.input_dataset,
                offset=rank_offset + self.progress_offset,
                limit=rank_limit - self.progress_offset,
                blobstore_client=self.blobstore_client,
                bucket=self.blobstore_bucket,
                rank=self.rank,
            )
        elif self.dataset_type == "local":
            raise NotImplementedError("Local file dataset not tested, please verify implementation.")
            data_iter = LocalFileDataset(
                self.input_dataset,
                offset=rank_offset + self.progress_offset,
                limit=rank_limit - self.progress_offset,
            )
        else:
            raise RuntimeError(f"Unknown dataset type {self.dataset_type}")

        # self.data_iterator = data_iter
        self.update_data_iterator(data_iter)
