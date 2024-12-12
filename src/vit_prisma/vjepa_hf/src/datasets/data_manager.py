# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger

_GLOBAL_SEED = 0
logger = getLogger()


def get_data_split_name(training: bool, is_test_split: bool) -> str:
    if training:
        data_split = "train"
        assert not is_test_split, "Can not use the test split at train time."
    elif is_test_split:
        data_split = "test"
    else:
        data_split = "valid"
    return data_split

    return data_split


def init_data(
    batch_size,
    transform=None,
    shared_transform=None,
    data="ImageNet",
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder=None,
    training=True,
    copy_data=False,
    drop_last=True,
    tokenize_txt=True,
    subset_file=None,
    sub_datasets=None,
    clip_len=None,
    dataset_fpcs=None,
    imageAsVideo_clip_len=None,
    frame_sample_rate=None,
    duration=None,
    fps=None,
    num_clips=1,
    random_clip_sampling=True,
    allow_clip_overlap=False,
    filter_short_videos=False,
    filter_long_videos=int(1e9),
    decode_one_clip=True,
    datasets_weights=None,
    persistent_workers=False,
    repeat_wds=False,
    ipe=300,
    deterministic=True,
    log_dir=None,
    data_subcategory=None,
    is_test_split=False,
    return_type=None,
    dataloader_profiler_conf=None,
    pad_frames: bool = False,
    log_resource_util_data: bool = False,
    log_stats_intervals: int = 0,
    chunked_db: bool = False,
    use_memory_efficient_weighted_sampler: bool = False,
):
    if data.lower() == "imagenet":
        from vit_prisma.vjepa_hf.src.datasets.imagenet1k import make_imagenet1k

        dataset, data_loader, dist_sampler = make_imagenet1k(
            transform=transform,
            batch_size=batch_size,
            collator=collator,
            pin_mem=pin_mem,
            training=training,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            image_folder=image_folder,
            persistent_workers=persistent_workers,
            copy_data=copy_data,
            drop_last=drop_last,
            subset_file=subset_file,
        )

    elif data.lower() == "imagenet22k":
        from vit_prisma.vjepa_hf.src.datasets.imagenet22k import make_imagenet22k

        dataset, data_loader, dist_sampler = make_imagenet22k(
            root_path=root_path,
            ipe=ipe,
            repeat_wds=repeat_wds,
            image_folder=image_folder,
            transform=transform,
            batch_size=batch_size,
            collator=collator,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
        )
        logger.info("ImageNet22k dataset and loader created")

    elif data.lower() == "lvd":
        from vit_prisma.vjepa_hf.src.datasets.lvd_images import make_lvd_dataset

        dataset, data_loader, dist_sampler = make_lvd_dataset(
            subsets=data_subcategory,
            frames_per_clip=clip_len,
            batch_size=batch_size,
            transform=transform,
            shared_transform=shared_transform,
            rank=rank,
            world_size=world_size,
            collator=collator,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_mem=pin_mem,
            persistent_workers=persistent_workers,
            deterministic=deterministic,
            log_dir=log_dir,
        )

    elif data.lower() == "laion":
        from vit_prisma.vjepa_hf.src.datasets.laion import make_laion

        dataset, data_loader, dist_sampler = make_laion(
            root_path=root_path,
            repeat_wds=repeat_wds,
            image_folder=image_folder,
            transform=transform,
            tokenize_txt=tokenize_txt,
            batch_size=batch_size,
            collator=collator,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
        )
        logger.info("LAION dataset and loader created")

    elif data.lower() == "videodataset":
        from vit_prisma.vjepa_hf.src.datasets.video_dataset import make_videodataset

        dataset, data_loader, dist_sampler = make_videodataset(
            data_paths=root_path,
            batch_size=batch_size,
            frames_per_clip=clip_len,
            imageAsVideo_frames_per_clip=imageAsVideo_clip_len,
            frame_step=frame_sample_rate,
            duration=duration,
            num_clips=num_clips,
            random_clip_sampling=random_clip_sampling,
            allow_clip_overlap=allow_clip_overlap,
            filter_short_videos=filter_short_videos,
            filter_long_videos=filter_long_videos,
            shared_transform=shared_transform,
            transform=transform,
            datasets_weights=datasets_weights,
            collator=collator,
            num_workers=num_workers,
            pin_mem=pin_mem,
            persistent_workers=persistent_workers,
            world_size=world_size,
            rank=rank,
            deterministic=deterministic,
            log_dir=log_dir,
        )

    elif data.lower() == "airstore_videodataset":
        from vit_prisma.vjepa_hf.src.datasets.airstore_dataset import make_airstore_dataset

        return_dict = isinstance(return_type, str) and return_type.lower() == "dict"

        dataset, data_loader, dist_sampler = make_airstore_dataset(
            data_paths=root_path,
            batch_size=batch_size,
            frames_per_clip=clip_len,
            frame_step=frame_sample_rate,
            duration=duration,
            fps=fps,
            dataset_fpcs=dataset_fpcs,
            num_clips=num_clips,
            random_clip_sampling=random_clip_sampling,
            allow_clip_overlap=allow_clip_overlap,
            shared_transform=shared_transform,
            transform=transform,
            datasets_weights=datasets_weights,
            collator=collator,
            num_workers=num_workers,
            pin_mem=pin_mem,
            persistent_workers=persistent_workers,
            world_size=world_size,
            rank=rank,
            return_dict=return_dict,
            log_dir=log_dir,
            dataloader_profiler_conf=dataloader_profiler_conf,
        )

    elif data.lower() == "blobstore_videodataset":
        from vit_prisma.vjepa_hf.src.datasets.blobstore_dataset import make_blobstore_dataset

        return_dict = isinstance(return_type, str) and return_type.lower() == "dict"

        dataset, data_loader, dist_sampler = make_blobstore_dataset(
            data_paths=root_path,
            batch_size=batch_size,
            frames_per_clip=clip_len,
            dataset_fpcs=dataset_fpcs,
            frame_step=frame_sample_rate,
            duration=duration,
            fps=fps,
            num_clips=num_clips,
            random_clip_sampling=random_clip_sampling,
            allow_clip_overlap=allow_clip_overlap,
            shared_transform=shared_transform,
            transform=transform,
            datasets_weights=datasets_weights,
            collator=collator,
            num_workers=num_workers,
            pin_mem=pin_mem,
            persistent_workers=persistent_workers,
            world_size=world_size,
            rank=rank,
            deterministic=deterministic,
            return_dict=return_dict,
            log_dir=log_dir,
            dataloader_profiler_conf=dataloader_profiler_conf,
            chunked_db=chunked_db,
        )

    elif data.lower() == "mixed_dataset":
        from vit_prisma.vjepa_hf.src.datasets.mixed_dataset import make_mixed_dataset

        if frame_sample_rate is not None:
            import warnings

            warnings.warn(
                "`frame_sample_rate` will be ignored. Use `duration` and `frames_per_clip` instead", DeprecationWarning
            )

        dataset, data_loader, dist_sampler = make_mixed_dataset(
            data_paths=root_path,
            datasets_weights=datasets_weights,
            batch_size=batch_size,
            frames_per_clip=clip_len,
            dataset_fpcs=dataset_fpcs,
            num_clips=num_clips,
            frame_step=frame_sample_rate,
            duration=duration,
            fps=fps,
            random_clip_sampling=random_clip_sampling,
            allow_clip_overlap=allow_clip_overlap,
            shared_transform=shared_transform,
            transform=transform,
            collator=collator,
            rank=rank,
            world_size=world_size,
            num_workers=num_workers,
            drop_last=True,
            pin_mem=pin_mem,
            persistent_workers=persistent_workers,
            log_dir=log_dir,
            log_resource_util_data=log_resource_util_data,
            log_stats_intervals=log_stats_intervals,
            use_memory_efficient_weighted_sampler=use_memory_efficient_weighted_sampler,
        )

    elif data.lower() == "audiovideodataset":
        from vit_prisma.vjepa_hf.src.datasets.audio_video_dataset import make_videodataset

        dataset, data_loader, dist_sampler = make_videodataset(
            data_paths=root_path,
            batch_size=batch_size,
            frames_per_clip=clip_len,
            frame_step=frame_sample_rate,
            duration=duration,
            num_clips=num_clips,
            random_clip_sampling=random_clip_sampling,
            allow_clip_overlap=allow_clip_overlap,
            filter_short_videos=filter_short_videos,
            filter_long_videos=filter_long_videos,
            shared_transform=shared_transform,
            transform=transform,
            datasets_weights=datasets_weights,
            collator=collator,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
        )

    elif data.lower() == "iterable-videodataset":
        from vit_prisma.vjepa_hf.src.datasets.iterable_video_dataset import make_webvid

        dataset, data_loader, dist_sampler = make_webvid(
            data_paths=root_path,
            batch_size=batch_size,
            transform=transform,
            shared_transform=shared_transform,
            filter_short_videos=filter_short_videos,
            decode_one_clip=decode_one_clip,
            repeat=repeat_wds,
            collator=collator,
            num_frames=clip_len,
            sampling_rate=frame_sample_rate,
            duration=duration,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
        )

    elif data.lower() == "ht100m_captions":
        from vit_prisma.vjepa_hf.src.datasets.ht100m_captions import make_ht100m_captions_airstore

        data_split = get_data_split_name(training, is_test_split)

        dataset, data_loader, dist_sampler = make_ht100m_captions_airstore(
            batch_size=batch_size,
            split=data_split,
            frames_per_clip=clip_len,
            frame_step=frame_sample_rate,
            num_clips=num_clips,
            random_clip_sampling=random_clip_sampling,
            transform=transform,
            shared_transform=shared_transform,
            world_size=world_size,
            rank=rank,
            num_workers=num_workers,
            pin_mem=pin_mem,
            persistent_workers=persistent_workers,
            collator=collator,
            drop_last=drop_last,
        )

    elif data.lower() == "webvid":
        from vit_prisma.vjepa_hf.src.datasets.webvid import make_webvid

        dataset, data_loader, dist_sampler = make_webvid(
            data_path=root_path,
            batch_size=batch_size,
            transform=transform,
            shared_transform=shared_transform,
            filter_short_videos=filter_short_videos,
            decode_one_clip=decode_one_clip,
            repeat=repeat_wds,
            collator=collator,
            num_frames=clip_len,
            sampling_rate=frame_sample_rate,
            duration=duration,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
        )

    elif data.lower() == "video_webdataset":
        from vit_prisma.vjepa_hf.src.datasets.video_webdataset import make_video_webdataset

        dataset, data_loader, dist_sampler = make_video_webdataset(
            data_paths=root_path,
            batch_size=batch_size,
            transform=transform,
            shared_transform=shared_transform,
            ipe=ipe,
            collator=collator,
            num_frames=clip_len,
            duration=duration,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            pad_frames=pad_frames,
        )

    elif data.lower() == "inat21" or data.lower() == "places205":
        # make_imagenet1k is generic
        # We can use the same function with a different [root_path]

        from vit_prisma.vjepa_hf.src.datasets.imagenet1k import make_imagenet1k

        dataset, data_loader, dist_sampler = make_imagenet1k(
            transform=transform,
            batch_size=batch_size,
            collator=collator,
            pin_mem=pin_mem,
            training=training,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            image_folder=image_folder,
            persistent_workers=persistent_workers,
            copy_data=copy_data,
            drop_last=drop_last,
            subset_file=subset_file,
        )
    elif data.lower() == "dm_perception":
        from vit_prisma.vjepa_hf.src.datasets.dm_perception import make_dm_perception_dataset

        data_split = get_data_split_name(training, is_test_split)

        dataset, data_loader, dist_sampler = make_dm_perception_dataset(
            batch_size=batch_size,
            transform=transform,
            data_split=data_split,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            image_folder=image_folder,
            data_subcategory=data_subcategory,
        )
    elif data.lower() == "mvbench":
        from vit_prisma.vjepa_hf.src.datasets.mvbench import make_mvbench_dataset

        if training:
            raise NotImplementedError("MVBench dataset does not have a training set.")
        else:
            dataset, data_loader, dist_sampler = make_mvbench_dataset(
                batch_size=batch_size,
                num_workers=num_workers,
                world_size=world_size,
                rank=rank,
                root_path=root_path,
                image_folder=image_folder,
                num_frames=clip_len,
                transform=transform,
            )

    return (data_loader, dist_sampler)
