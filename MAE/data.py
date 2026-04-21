from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
import os
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, DistributedSampler

from physics_jepa.data import get_sequence_dataset

from .config import set_the_well_env


def _worker_init_fn(rank: int):
    def init(worker_id: int) -> None:
        worker_seed = (torch.initial_seed() + rank + worker_id) % 2**32
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return init


def _loader_options(cfg: DictConfig) -> dict:
    num_workers = int(cfg.data.get("num_workers", 4))
    options = {
        "num_workers": num_workers,
        "pin_memory": bool(cfg.data.get("pin_memory", True)),
        "persistent_workers": bool(cfg.data.get("persistent_workers", True)) if num_workers > 0 else False,
        "prefetch_factor": int(cfg.data.get("prefetch_factor", 2)) if num_workers > 0 else None,
    }
    return options


def make_sequence_dataloader(
    cfg: DictConfig,
    split: str,
    batch_size: Optional[int] = None,
    shuffle: Optional[bool] = None,
    drop_last: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    """Build a sequence dataloader using the existing physics_jepa dataset code."""
    resolved_data_dir = set_the_well_env(cfg)
    split_name = "val" if split == "valid" else split
    disk_split = "valid" if split_name == "val" else split_name
    is_train = split_name == "train"
    if batch_size is None:
        batch_size = int(cfg.train.batch_size if is_train else cfg.eval.batch_size)
    if shuffle is None:
        shuffle = is_train

    expected_split_dir = Path(resolved_data_dir) / cfg.dataset.name / "data" / disk_split
    hdf5_files = list(expected_split_dir.rglob("*.h5")) + list(expected_split_dir.rglob("*.hdf5"))
    if not hdf5_files:
        configured = cfg.paths.get("the_well_data_dir", None)
        raise FileNotFoundError(
            "No HDF5 files found for MAE dataloader. "
            f"Configured paths.the_well_data_dir={configured!r}; "
            f"resolved THE_WELL_DATA_DIR={os.environ.get('THE_WELL_DATA_DIR')!r}; "
            f"expected split directory={str(expected_split_dir)!r}."
        )

    dataset = get_sequence_dataset(
        dataset_name=cfg.dataset.name,
        num_frames=int(cfg.dataset.num_frames),
        split=split_name,
        resolution=cfg.dataset.get("resolution", None),
        offset=cfg.dataset.get("offset", None),
        subset_config_path=cfg.dataset.get("subset_config_path", None),
        noise_std=float(cfg.dataset.get("noise_std", 0.0)),
    )

    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset=dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=int(cfg.train.get("seed", 42)),
        )

    loader_kwargs = _loader_options(cfg)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        drop_last=drop_last,
        worker_init_fn=_worker_init_fn(rank),
        **loader_kwargs,
    )


def get_train_loader(cfg: DictConfig, rank: int = 0, world_size: int = 1) -> DataLoader:
    return make_sequence_dataloader(
        cfg,
        split="train",
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        drop_last=True,
        rank=rank,
        world_size=world_size,
    )


def get_val_loader(cfg: DictConfig, rank: int = 0, world_size: int = 1) -> DataLoader:
    return make_sequence_dataloader(
        cfg,
        split="val",
        batch_size=int(cfg.train.get("val_batch_size", cfg.train.batch_size)),
        shuffle=False,
        drop_last=False,
        rank=rank,
        world_size=world_size,
    )


def get_eval_loader(cfg: DictConfig, split: str) -> DataLoader:
    return make_sequence_dataloader(
        cfg,
        split=split,
        batch_size=int(cfg.eval.batch_size),
        shuffle=False,
        drop_last=False,
        rank=0,
        world_size=1,
    )
