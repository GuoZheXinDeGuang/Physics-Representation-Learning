from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    cfg: DictConfig,
) -> None:
    model_to_save = unwrap_model(model)
    checkpoint = {
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def save_encoder(path: str | Path, model: torch.nn.Module, cfg: DictConfig) -> None:
    model_to_save = unwrap_model(model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder": model_to_save.encoder.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
        path,
    )


def load_training_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model_to_load = unwrap_model(model)
    model_to_load.load_state_dict(checkpoint["model"], strict=True)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint


def load_encoder_or_model(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    map_location: str | torch.device = "cpu",
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model_to_load = unwrap_model(model)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model_to_load.load_state_dict(checkpoint["model"], strict=True)
        return
    if isinstance(checkpoint, dict) and "encoder" in checkpoint:
        model_to_load.encoder.load_state_dict(checkpoint["encoder"], strict=True)
        return
    model_to_load.encoder.load_state_dict(checkpoint, strict=True)
