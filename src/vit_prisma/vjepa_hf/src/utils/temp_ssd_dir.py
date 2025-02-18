import os
from functools import lru_cache

from vit_prisma.vjepa_hf.src.utils.logging import get_logger

LOGGER = get_logger("temp_ssd_dir")


@lru_cache
def temp_ssd_dir():
    job_id = os.environ.get("SLURM_JOB_ID", None)
    if job_id:
        slurm_ssd_temp_path = os.path.join("/scratch/slurm_tmpdir", job_id)
        if os.path.exists(slurm_ssd_temp_path):
            LOGGER.info(f"Using existing slurm temp dir: {slurm_ssd_temp_path}")
            return slurm_ssd_temp_path
    LOGGER.info("Using the default OS path for temp dir")
