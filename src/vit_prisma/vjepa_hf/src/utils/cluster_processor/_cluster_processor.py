import os
import time
from abc import ABC
from typing import Iterable, Optional

import submitit
from filelock import FileLock

from vit_prisma.vjepa_hf.src.utils.logging import get_logger
from vit_prisma.vjepa_hf.src.utils.temp_ssd_dir import temp_ssd_dir


class ClusterProcessor(ABC):
    """Generic class for parallel cluster processing tasks.

    ClusterProcessor wraps a data iterator and a processing function in a way
    designed for highly-parallelized processing tasks on SLURM clusters. At a
    high level, ClusterProcessor takes care of task-level sharding and job
    submission, while the user is expected to handle sharding across job
    arrays and write the processing function.

    Using ClusterProcessor requires a few steps:
    * You must overwrite ``setup_data_iterator`` with your data iterator setup
        process. ``setup_data_iterator`` must call ``update_data_iterator``
        with your instantiated iterator.
    * You must overwrite ``setup_iterations`` with any other setup tasks
        (such as initializing log files).
    * You must overwrite ``process_example`` with your pipeline for processing
        a single retrieved example from your iterator.
    * You must overwrite ``rebuild_processor`` with a pipeline for rebuilding
        ClusterProcessor for job resubmission based on the class attributes you
        have stored.

    At a high level, the following is the order of operations:
    1. ``setup_data_iterator`` called.
    2. ``setup_iterations`` called.
    3. During loop, ``process_example`` is called for each example
    4. On job resubmission, ``rebuild_processor`` is called for instantiating
        the ClusterProcessor to be used for the next job.

    The ClusteProcessor will process the data from `local_offset` to
    `job_offset` + `job_limit`. Progress is tracked via the
    `progress_offset` variable across SLURM subtasks. The minimal
    `progress_offset` among all subtasks is passed to `rebuild_processor` on
    job resubmission.

    Args:
        job_offset: The offset for this ClusterProcessor within the global
            dataset. The offset is "local" in the sense that it is typically
            assigned for a single ClusterProcessor within a SLURM job array.
        job_limit: The number of examples to process for a single SLURM job.
            On instantiation, if the number of SLURM tasks is greater than 1,
            the examples between ``job_offset`` and ``job_limit`` will be
            sharded between the tasks.
        progress_offset: A tracking variable for the progress for each task.
            On job resubmission, the minimum ``progress_offset`` among all
            tasks will be passed to ``rebuild_processor``.
        task_sleep_interval: The tasks will wait for the other tasks to
            finish prior to triggering resubmission. This variable governs how
            often the rank-0 task checks to see if the other tasks are running.
        min_time_per_iter: A minimum time for processing each example. Used to
            avoid triggering IOPS limits on the cluster.
    """

    job_offset: int
    job_limit: int
    progress_offset: int
    task_sleep_interval: float
    min_time_per_iter: float

    world_size: Optional[int] = None
    rank: Optional[int] = None
    temp_dir: Optional[str] = None
    task_file: Optional[str] = None
    progress_offset_file: Optional[str] = None
    data_iterator: Optional[Iterable] = None

    def __init__(
        self,
        job_offset: int,
        job_limit: int,
        progress_offset: int = 0,
        task_sleep_interval: float = 60.0,
        min_time_per_iter: float = 0.0,
    ):
        self.logger = get_logger(self.__class__.__name__)

        self.job_offset = job_offset
        self.job_limit = job_limit
        self.progress_offset = progress_offset
        self.task_sleep_interval = task_sleep_interval
        self.min_time_per_iter = min_time_per_iter

    def _get_world_size_and_rank(self) -> tuple[int, int]:
        world_size = int(os.environ.get("SLURM_NTASKS", 1))
        rank = int(os.environ.get("SLURM_PROCID", 0))

        return world_size, rank

    def _add_task_count(self):
        assert self.task_file is not None
        with FileLock(f"{self.task_file}.lock"):
            if os.path.exists(self.task_file):
                with open(self.task_file, "r") as f:
                    num = int(f.read())

                num += 1

                with open(self.task_file, "w") as f:
                    f.write(str(num))
            else:
                with open(self.task_file, "w") as f:
                    num = f.write("1")

    def _decrement_task_count(self):
        assert self.task_file is not None
        with FileLock(f"{self.task_file}.lock"):
            with open(self.task_file, "r") as f:
                num = int(f.read())

            num -= 1
            if num < 0:
                raise RuntimeError(f"Decremented value is {num}, but it should not be less than 1.")

            with open(self.task_file, "w") as f:
                f.write(str(num))

    def _read_task_count(self) -> int:
        assert self.task_file is not None
        with open(self.task_file, "r") as f:
            num = int(f.read())

        return num

    def _update_local_offset(self, offset: int):
        assert self.progress_offset_file is not None
        with open(self.progress_offset_file, "w") as f:
            f.write(str(offset))

    def _read_local_offset(self, fname: str) -> int:
        if fname is None:
            assert self.progress_offset_file is not None
            fname = self.progress_offset_file

        with open(fname, "r") as f:
            num = int(f.read())

        return num

    def update_data_iterator(self, iterator):
        self.data_iterator = iterator

    def setup_data(self, rank_offset: int, rank_limit: int):
        raise NotImplementedError

    def setup_iterations(self):
        pass

    def process_example(self, example):
        raise NotImplementedError

    def rebuild_processor(self, progress_offset: int):
        raise NotImplementedError

    def end_loop_processing(self):
        pass

    def __call__(self, checkpointpath=None):
        self.temp_dir = temp_ssd_dir()
        self.task_file = os.path.join(self.temp_dir, "task_count")
        self._add_task_count()

        self.world_size, self.rank = self._get_world_size_and_rank()
        self.progress_offset_file = os.path.join(self.temp_dir, f"progress_offset_{self.rank}")
        self._update_local_offset(self.progress_offset)

        if self.job_limit == -1:
            local_rank_limit = self.job_limit
            local_rank_offset = self.job_offset
        else:
            local_rank_limit = self.job_limit // self.world_size
            local_rank_offset = self.job_offset + self.rank * local_rank_limit

        self.setup_data(local_rank_offset, local_rank_limit)
        if self.data_iterator is None:
            raise RuntimeError(
                "data_iterator not set in setup_data_iterator(). You must update the data_iterator "
                "attribute with your dataloader during the execution of this function using update_data_iterator()"
            )
        assert self.data_iterator is not None

        self.setup_iterations()
        self.logger.info(
            f"Starting up rank {self.rank}, job_offset: {self.job_offset} job_world_size: {self.world_size}, "
            f"local_rank_offset: {local_rank_offset}, local_rank_limit: {local_rank_limit}, "
            f"progress_offset: {self.progress_offset}"
        )

        for example in self.data_iterator:
            start = time.time()
            if (self.progress_offset) % 100 == 0:
                self._update_local_offset(self.progress_offset)
                self.logger.info(f"Processed {self.progress_offset} examples")

            self.process_example(example)
            self.progress_offset += 1

            end = time.time()
            if end - start < self.min_time_per_iter:
                time.sleep(self.min_time_per_iter - (end - start))

        self.logger.info(f"Rank {self.rank} finished processing {self.progress_offset} examples")
        self.end_loop_processing()
        self._decrement_task_count()

        task_count = self._read_task_count()
        while task_count > 0:
            gram_task = "task" if task_count == 1 else "tasks"
            self.logger.info(f"Waiting for {task_count} other {gram_task} to finish ...")
            time.sleep(self.task_sleep_interval)
            task_count = self._read_task_count()

    def checkpoint(self, checkpointpath=None):
        assert self.world_size is not None
        assert self.temp_dir is not None

        offset_file = os.path.join(self.temp_dir, f"progress_offset_{0}")
        progress_offset = self._read_local_offset(offset_file)
        for ind in range(1, self.world_size):
            offset_file = os.path.join(self.temp_dir, f"progress_offset_{ind}")
            cur_offset = self._read_local_offset(offset_file)
            if cur_offset < progress_offset:
                progress_offset = cur_offset

        self.logger.info(f"Restarting with progress offset {progress_offset}")
        return submitit.helpers.DelayedSubmission(self.rebuild_processor(progress_offset=progress_offset))
