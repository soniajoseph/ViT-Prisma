import os
import socket
import subprocess
from functools import lru_cache

from vit_prisma.vjepa_hf.src.utils.logging import get_logger

logger = get_logger("Cluster utils")


AWS_CLUSTER = "aws"
H2_CLUSTER = "h2"
RSC_CLUSTER = "rsc"
COREWEAVE_CLUSTER = "coreweave"


SUPPORTED_CLUSTERS = {
    "fair-a100": AWS_CLUSTER,
    "rsc": RSC_CLUSTER,
    "h2learnfair": H2_CLUSTER,
    "learnfair": H2_CLUSTER,
    "dm1-control": COREWEAVE_CLUSTER,
}


@lru_cache()
def get_cluster() -> str:
    # If the node is assigned by slurm, this is easy
    where = os.environ.get("SLURM_CLUSTER_NAME")
    if where is not None:
        if where in SUPPORTED_CLUSTERS:
            return SUPPORTED_CLUSTERS[where]
        else:
            # return the cluster name so the user knows to add support for it
            return where
    # If we have an /fsx partition, it's AWS
    if os.path.exists("/fsx-jepa"):
        return AWS_CLUSTER
    # If we're still here, maybe we can back it out from the hostname
    hostname = socket.gethostname()
    if hostname.startswith("rsc"):
        return RSC_CLUSTER
    elif hostname.startswith("devfair") or hostname.startswith("learnfair"):
        return H2_CLUSTER
    else:
        result = subprocess.run(["uname", "-a"], stdout=subprocess.PIPE)
        result = result.stdout.decode()
        if COREWEAVE_CLUSTER in result:
            return COREWEAVE_CLUSTER
    # default: return the hostname so the user knows to update this logic
    return hostname


# Gets slurm job vars, to launch another job with the same vars
def slurm_account_partition_and_qos(low_pri: bool) -> str:
    account = os.environ.get("SLURM_JOB_ACCOUNT")
    partition = os.environ.get("SLURM_JOB_PARTITION")
    qos = os.environ.get("SLURM_JOB_QOS")
    assert None not in (account, partition, qos), "This function should only be called by a job scheduled by slurm"
    if low_pri:
        # FAIR cluster doesn't use QoS-based scheduling yet
        if get_cluster() == H2_CLUSTER:
            partition = "devlab,learnlab,learnfair"
        else:
            qos = "lowest"
    return account, partition, qos


# TODO: UPDATE PATHS BEFORE RELEASE
DATASET_PATHS_BY_CLUSTER = {
    AWS_CLUSTER: {
        "AVA": "/fsx-jepa/abardes/datasets/AVA/ava_train_paths.csv",
        "Shutterstock_stopwordfiltered": "/fsx-jepa/ballasn/data/shutterstock/stopwords_filtered/files.npy",
        "Shutterstock_imnet": "/fsx-jepa/ballasn/data/shutterstock/stopwords_synsets_balance_imnet/files.npy",
        "Shutterstock_imnet2000": "/fsx-jepa/ballasn/data/shutterstock/stopwords_synsets_balance_imnet2000/files.npy",
        "Shutterstock_k710": "/fsx-jepa/ballasn/data/shutterstock/stopwords_synsets_balance_k710/files.npy",
        "Ego4d_track2": "/fsx-jepa/ballasn/data/ego4d/track2/files.csv",
        # -- Old VideoMix2M
        "K400": "/fsx-jepa/abardes/datasets/Kinetics400/annotations/k400_train_paths.csv",
        "K700": "/fsx-jepa/abardes/datasets/Kinetics700/annotations/k700_train_paths.csv",
        "K710": "/fsx-jepa/ballasn/datasets/Kinetics710/annotations/k710_train_paths.csv",  # labels are incorrect
        "SSv2": "/fsx-jepa/abardes/datasets/SSv2_annotations/ssv2_train.csv",
        "HowTo100M": "/fsx-jepa/abardes/datasets/HowTo100M/howto100m_paths.csv",
        # -- Validation data
        "K400_val": "/fsx-jepa/abardes/datasets/Kinetics400/annotations/k400_val_paths.csv",
        "K710_val": "/fsx-jepa/abardes/datasets/Kinetics400/annotations/k400_val_paths.csv",
        "SSv2_val": "/fsx-jepa/abardes/datasets/SSv2_annotations/ssv2_val.csv",
        "ImageNet": "/datasets01/imagenet_full_size/061417/",
        # -- Reindexed data
        "K710v2": "/fsx-jepa/massran/k710_fullres_paths.csv",
        "SSv2v2": "/fsx-jepa/massran/ssv2_fullres_paths.csv",
        "HowTo100M-FullRes": "/fsx-jepa/massran/howto100m_fullres_paths.csv",
        # -- Sharded datasets
        "SSv2-Sharded": "/fsx-jepa/massran/SSv2/data/{00000..00220}.tar",
        "K710-Sharded": "/fsx-jepa/massran/K710/data/{00000..01189}.tar",
        # -- Webdatasets
        "videomix2m-wds": "/fsx-jepa/rhowes/data/videomix2m-wds/{000000..009999}.tar",
        "howto100m-wds": "/fsx-jepa/rhowes/data/howto-wds/{00000..03999}.tar",
    },
    RSC_CLUSTER: {
        # -- Datasets migrated from AWS and stored on NFS
        "K400": "/checkpoint/jepa/ballasn/data/index/k400_train.csv",
        "K710": "/checkpoint/jepa/ballasn/data/index/k710_train.csv",  # labels are incorrect
        "SSv2": "/checkpoint/jepa/ballasn/data/index/ssv2_train.csv",
        "HowTo100M": "/checkpoint/jepa/ballasn/data/index/howto_320p.csv",
        "HowTo100M-FullRes": "/checkpoint/jepa/ballasn/data/index/howto.csv",
        "COIN": "/checkpoint/jepa/abardes/data/COIN/han_train_paths.csv",
        "EK100": "/checkpoint/jepa/hanlin/data/EpicKitchens100/annotations/EPIC_100_train.csv",
        "videomix2m-wds": "/checkpoint/jepa/data/videomix2m-wds/{000000..009999}.tar",
        "howto100m-wds": "/checkpoint/jepa/data/howto100m-wds/{00000..03999}.tar",
        # -- Validation data
        "K400_val": "/checkpoint/jepa/ballasn/data/index/k400_val.csv",
        "SSv2_val": "/checkpoint/jepa/ballasn/data/index/ssv2_val.csv",
        "COIN_val": "/checkpoint/jepa/abardes/data/COIN/han_val_paths.csv",
        "EK100_val": "/checkpoint/jepa/hanlin/data/EpicKitchens100/annotations/EPIC_100_validation.csv",
        "ImageNet": "/checkpoint/jepa/data/imagenet_full_size/061417/",
        # -- AIRStore datasets
        "HowTo100M-AIRStore": "airstore://jepa_howto100m_no_user_data",
        "Kinetics-AIRStore": "airstore://jepa_kinetics_710_train_no_user_data",
        "SSv2-AIRStore": "airstore://jepa_ssv2_no_user_data",
        "YT-Temporal-1B-AIRStore": "airstore://yttemporal1b_no_user_data",
        "Ego4d_egoexo4d-AIRStore": "airstore://ego4d_egoexo4d_no_user_data_rsc",
        "Muvigen-AIRStore": "airstore://muvigen_shutterstock_video_derived_resample_v1_no_user_data",
        "EK100-Train-AIRStore": "airstore://jepa_ek100_256px_train_no_user_data",
        "EK100-Valid-AIRStore": "airstore://jepa_ek100_256px_valid_no_user_data",
        "IN1K-AIRStore": "airstore://jepa_imagenet_no_user_data",
        # -- BlobStore datasets
        "yttemporal1b_512": "yttemporal1b_512",
        "yttemporal1b_360px_120s": "yttemporal1b_360px_120s",
        "yttemporal1b_resized360p_chunked120": "yttemporal1b_resized/res360_chunklength120",
        "yttemporal1b_resized480p_chunked120": "yttemporal1b_resized/res480_chunklength120",
        "yttemporal1b_resized720p_chunked120": "yttemporal1b_resized/res720_chunklength120",
        "yt1b_kinetics_curated_k10_v1": "yt1b_kinetics_curated_k10_v1",
        "howto100m": "howto100m",
        "howto100_resized360p_chunked120": "howto100m_resized/res360_chunklength120",
        "ssv2": "ssv2",
        "jepa_kinetics_710_train": "jepa_kinetics_710_train",
        "ego4d_track2_v1_full_512": "ego4d_track2_v1_full_512",
        "ego4d_chunked_512": "ego4d_chunked_512",
        "ego4d_full_length": "ego4d_full_length",
        "ego4d_resized360p_chunked120": "ego4d_resized/res360_chunklength120",
        "ego4d_resized480p_chunked120": "ego4d_resized/res480_chunklength120",
        "ego4d_resized720p_chunked120": "ego4d_resized/res720_chunklength120",
        # -- Shared datasets
        "LVD": "/checkpoint/dino/datasets/LaViDa-20221031-blurred",
    },
    COREWEAVE_CLUSTER: {
        "videomix2m-wds": "/checkpoint/amaia/video/data/videomix2m-wds/{000000..009999}.tar",
        "K710": "/checkpoint/amaia/video/davidfan/k710_train_paths.csv",
        "SSv2": "/checkpoint/amaia/video/abardes/data/index/ssv2_train_paths.csv",
        "K400": "/checkpoint/amaia/video/abardes/data/index/k400_train_paths.csv",
        "K710": "/checkpoint/amaia/video/davidfan/k710_train_paths.csv",
        "HowTo100M": "/checkpoint/amaia/video/massran/howto_320p.csv",
        "EK100": "/checkpoint/amaia/video/massran/EPIC_100_train.csv",
        "SSv2_val": "/checkpoint/amaia/video/abardes/data/index/ssv2_val_paths.csv",
        "K400_val": "/checkpoint/amaia/video/abardes/data/index/k400_val_paths.csv",
        "EK100_val": "/checkpoint/amaia/video/massran/EPIC_100_validation.csv",
        "ImageNet": "/checkpoint/amaia/video/massran/in1k_all.csv",
        "HowTo100M-FR": "/checkpoint/amaia/video/davidfan/howto100m_paths.csv",
        "YouTubeVOS": "/checkpoint/amaia/video/data/youtube_vos/2018/train/",
        "YouTubeVOS_val": "/checkpoint/amaia/video/data/youtube_vos/2018/valid/",
        "DAVISVOS": "/checkpoint/amaia/video/data/DAVIS/2017",
        "DAVISVOS_val": "/checkpoint/amaia/video/data/DAVIS/2017",
        "EgoSchema": "/checkpoint/amaia/video/data/EgoSchema",
    },
    H2_CLUSTER: {
        "DROID": "/checkpoint/basileterv/data/datasets/droid/droid_train_paths_h2_datasets01.csv",
        "METAWORLD": "/checkpoint/basileterv/results/tdmpc2/224_rgb_state_all_tasks/train_paths.csv",
        # -- Old VideoMix2M
        "K400": "/checkpoint/abardes/datasets/Kinetics400/annotations/k400_train_paths.csv",
        "K700": "/checkpoint/abardes/datasets/Kinetics700/annotations/k700_train_paths.csv",
        "SSv2": "/checkpoint/abardes/datasets/SSv2_annotations/ssv2_train.csv",
        # -- Validation data
        "K400_val": "/checkpoint/abardes/datasets/Kinetics400/annotations/k400_val_paths.csv",
        "K710_val": "/checkpoint/abardes/datasets/Kinetics700/annotations/k400_val_paths.csv",
        "SSv2_val": "/checkpoint/abardes/datasets/SSv2_annotations/ssv2_val.csv",
        "ImageNet": "/datasets01/imagenet_full_size/061417/",
    },
}


def get_dataset_path(dataset: str, cluster=None) -> str:
    if cluster is None:
        cluster = get_cluster()

    print(cluster)

    return DATASET_PATHS_BY_CLUSTER[cluster][dataset]


def get_dataset_paths(datasets: list[str], is_train: bool = True) -> list[str]:
    cluster = get_cluster()
    assert cluster in DATASET_PATHS_BY_CLUSTER, f"No data paths for environment {cluster}!"
    paths = []
    for dataset in datasets:
        if not is_train:
            dataset = f"{dataset}_val"
        try:
            path = get_dataset_path(dataset, cluster)
        except Exception:
            raise Exception(f"Could not find dataset {dataset} for cluster {cluster}")
        paths.append(path)
    logger.info(f"Datapaths {paths}")
    return paths


# TODO: remove?
def dataset_paths() -> dict[str, str]:
    cluster = get_cluster()
    assert cluster in DATASET_PATHS_BY_CLUSTER, f"No data paths for environment {cluster}!"
    return DATASET_PATHS_BY_CLUSTER[cluster]
