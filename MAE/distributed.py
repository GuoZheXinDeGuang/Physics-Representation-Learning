from __future__ import annotations

import os

import torch
import torch.distributed as dist


def init_distributed() -> tuple[int, int, torch.device]:
    """Initialize torch.distributed when launched by torchrun."""
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        return rank, world_size, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return 0, 1, device


def is_main_process(rank: int) -> bool:
    return rank == 0


def cleanup_distributed(world_size: int) -> None:
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


def reduce_mean(value: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size < 2 or not dist.is_initialized():
        return value
    value = value.detach().clone()
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    value /= world_size
    return value
