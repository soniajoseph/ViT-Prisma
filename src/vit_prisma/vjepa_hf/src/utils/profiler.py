import os

import torch

from vit_prisma.vjepa_hf.src.datasets.utils.dataloader import get_worker_info
from vit_prisma.vjepa_hf.src.utils.logging import get_logger

logger = get_logger("Profiler utils")


def get_profiler_trace_handler(output_dir, rank, print_number_rows=30):

    def trace_handler(prof):
        if print_number_rows > 0:
            print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=30))
            print(prof.key_averages(group_by_stack_n=5).table(sort_by="cuda_time_total", row_limit=30))
            print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cpu_time_total", row_limit=30))
            print(prof.key_averages(group_by_stack_n=5).table(sort_by="cpu_time_total", row_limit=30))

        if output_dir is None:
            return

        # Save the trace to a file
        profiler_output_dir = os.path.join(output_dir, "profiler_trace")
        os.makedirs(profiler_output_dir, exist_ok=True)

        _, worker_id = get_worker_info()
        output_fpath = os.path.join(profiler_output_dir, f"rank_{rank}_worker{worker_id}.json")
        logger.info(f"Saving DATALOADER trace for rank {rank} to {output_fpath}")
        prof.export_chrome_trace(output_fpath)
        logger.info(f"Successfully saved DATALOADER profiler trace to {output_fpath}")

    return trace_handler


def get_profiler_scheduler(scheduler_conf):
    """
    Creates a profiler schedule based on the config.

    For details of the conif, please refer to:
    https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html#using-profiler-to-analyze-long-running-jobs
    """
    if not scheduler_conf:
        return None

    logger.info(f"Profiler scheduler config: {scheduler_conf}")

    skip_first = scheduler_conf.get("skip_first", 4)
    wait = scheduler_conf.get("wait", 1)
    warmup = scheduler_conf.get("warmup", 1)
    active = scheduler_conf.get("active", 2)
    repeat = scheduler_conf.get("repeat", 2)
    return torch.profiler.schedule(skip_first=skip_first, warmup=warmup, wait=wait, active=active, repeat=repeat)


def create_profiler(profiler_conf, folder, rank):
    if profiler_conf is None:
        return None

    profiler_activities = []
    profiler_scheduler_conf = profiler_conf.get("scheduler", {})

    if profiler_conf.get("cpu", False):
        profiler_activities.append(torch.profiler.ProfilerActivity.CPU)
    if profiler_conf.get("cuda", False):
        profiler_activities.append(torch.profiler.ProfilerActivity.CUDA)

    if folder:
        job_id = os.environ.get("SLURM_JOB_ID", None)
        folder = os.path.join(folder, f"dataloader_job_{job_id}")

    profiler = torch.profiler.profile(
        activities=profiler_activities,
        schedule=get_profiler_scheduler(profiler_scheduler_conf),
        on_trace_ready=get_profiler_trace_handler(folder, rank),
        record_shapes=profiler_conf.get("record_shapes", False),
        profile_memory=profiler_conf.get("profile_memory", True),
        use_cuda=profiler_conf.get("use_cuda", True),
        with_stack=profiler_conf.get("with_stack", True),
    )
    return profiler
