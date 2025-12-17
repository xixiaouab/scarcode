import os
import torch
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler
from typing import Optional, List, Dict, Union, Tuple
from .dataset import ScarElasticDataset
from .pre_processing import Preprocessor


def get_train_transforms(spatial_size: Tuple[int, int, int] = (128, 128, 48)):
    from monai.transforms import (
        Compose,
        LoadImaged,
        EnsureChannelFirstd,
        Orientationd,
        Spacingd,
        NormalizeIntensityd,
        RandAffined,
        RandGaussianNoised,
        RandScaleIntensityd,
        Rand3DElasticd,
        EnsureTyped,
        ConcatItemsd
    )

    return Compose([
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"]),
        Orientationd(keys=["image", "mask"], axcodes="RAS"),
        Spacingd(
            keys=["image", "mask"],
            pixdim=(1.25, 1.25, 1.25),
            mode=("bilinear", "nearest")
        ),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        RandAffined(
            keys=["image", "mask", "prior_thickness", "prior_edge"],
            prob=1.0,
            rotate_range=(np.pi / 12, np.pi / 12, np.pi / 12),
            scale_range=(0.1, 0.1, 0.1),
            mode=("bilinear", "nearest", "bilinear", "bilinear"),
            padding_mode="zeros"
        ),
        Rand3DElasticd(
            keys=["image", "mask", "prior_thickness", "prior_edge"],
            sigma_range=(5, 7),
            magnitude_range=(50, 150),
            prob=0.5,
            mode=("bilinear", "nearest", "bilinear", "bilinear"),
            padding_mode="zeros"
        ),
        RandGaussianNoised(keys=["image"], prob=0.5, std=0.01),
        RandScaleIntensityd(keys=["image"], factors=0.2, prob=0.5),
        EnsureTyped(keys=["image", "mask", "prior_thickness", "prior_edge"]),
    ])


def get_val_transforms():
    from monai.transforms import (
        Compose,
        LoadImaged,
        EnsureChannelFirstd,
        Orientationd,
        Spacingd,
        NormalizeIntensityd,
        EnsureTyped
    )

    return Compose([
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"]),
        Orientationd(keys=["image", "mask"], axcodes="RAS"),
        Spacingd(
            keys=["image", "mask"],
            pixdim=(1.25, 1.25, 1.25),
            mode=("bilinear", "nearest")
        ),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "mask", "prior_thickness", "prior_edge"]),
    ])


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    import random
    random.seed(worker_seed)


def build_dataloader(
        data_dir: str,
        json_list: str,
        train: bool = True,
        batch_size: int = 2,
        num_workers: int = 4,
        distributed: bool = False,
        seed: int = 42
) -> DataLoader:
    if train:
        transforms = get_train_transforms()
        shuffle = not distributed
    else:
        transforms = get_val_transforms()
        shuffle = False

    dataset = ScarElasticDataset(
        data_root=data_dir,
        datalist_json=json_list,
        transform=transforms,
        phase="train" if train else "test"
    )

    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=train)

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=True,
        worker_init_fn=_seed_worker,
        generator=generator,
        drop_last=train
    )


class DataFactory:
    def __init__(self, config: Dict):
        self.config = config
        self.preprocessor = Preprocessor(
            target_spacing=(1.25, 1.25, 1.25),
            normalize=True
        )

    def get_loader(self, phase: str) -> DataLoader:
        if phase not in ["train", "val", "test"]:
            raise ValueError(f"Unknown phase: {phase}")

        return build_dataloader(
            data_dir=self.config["data_root"],
            json_list=self.config[f"{phase}_json"],
            train=(phase == "train"),
            batch_size=self.config["batch_size"] if phase == "train" else 1,
            num_workers=self.config["num_workers"],
            distributed=self.config.get("distributed", False),
            seed=self.config.get("seed", 42)
        )


__all__ = [
    "ScarElasticDataset",
    "build_dataloader",
    "get_train_transforms",
    "get_val_transforms",
    "DataFactory"
]