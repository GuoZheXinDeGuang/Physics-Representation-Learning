from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from .checkpoint import load_training_checkpoint, save_checkpoint, save_encoder, unwrap_model
from .config import load_config, save_resolved_config, set_the_well_env
from .data import get_train_loader, get_val_loader
from .distributed import cleanup_distributed, init_distributed, is_main_process, reduce_mean
from .model import MaskedAutoencoderVideo, count_parameters


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(payload) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(torch.cuda.is_available())


def get_autocast(device: torch.device, enabled: bool, dtype: str):
    if not enabled:
        return nullcontext()
    amp_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def build_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except AttributeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def cosine_lr(
    optimizer: torch.optim.Optimizer,
    update: int,
    warmup_updates: int,
    total_updates: int,
    base_lr: float,
    min_lr: float,
) -> float:
    if update < warmup_updates:
        lr = base_lr * float(update + 1) / float(max(1, warmup_updates))
    else:
        progress = float(update - warmup_updates) / float(max(1, total_updates - warmup_updates))
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


@torch.no_grad()
def validate(model: torch.nn.Module, loader, device: torch.device, cfg, world_size: int) -> float:
    model.eval()
    max_batches = cfg.train.get("val_max_batches", None)
    losses = []
    use_amp = bool(cfg.train.amp) and device.type == "cuda"
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        videos = batch["sequence"].to(device, non_blocking=True).float()
        with get_autocast(device, use_amp, str(cfg.train.get("amp_dtype", "float16"))):
            loss = model(videos)["loss"]
        losses.append(reduce_mean(loss.detach(), world_size).cpu())
    model.train()
    if not losses:
        return float("nan")
    return float(torch.stack(losses).mean().item())


def train(cfg, resume: str | None = None) -> Path:
    set_the_well_env(cfg)
    rank, world_size, device = init_distributed()
    set_seed(int(cfg.train.seed) + rank)

    output_root = Path(cfg.paths.output_dir)
    if resume is not None:
        output_dir = Path(resume).resolve().parent
    elif cfg.train.get("timestamp_output", True):
        output_dir = output_root / dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else:
        run_name = str(cfg.train.get("run_name", "active_matter_videomae_mae"))
        output_dir = output_root / run_name

    if is_main_process(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
        save_resolved_config(cfg, output_dir / "config.yaml")

    train_loader = get_train_loader(cfg, rank=rank, world_size=world_size)
    val_loader = get_val_loader(cfg, rank=rank, world_size=world_size)

    model = MaskedAutoencoderVideo.from_config(cfg).to(device)
    total_params = count_parameters(model)
    if total_params >= int(cfg.model.max_parameters):
        raise ValueError(f"Model has {total_params} parameters, exceeding limit {cfg.model.max_parameters}")

    if is_main_process(rank):
        print(f"THE_WELL_DATA_DIR={set_the_well_env(cfg)}", flush=True)
        print(f"Total MAE parameters: {total_params:,}", flush=True)
        print(f"Output directory: {output_dir}", flush=True)

    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.lr),
        betas=(0.9, 0.95),
        weight_decay=float(cfg.train.weight_decay),
    )
    use_amp = bool(cfg.train.amp) and device.type == "cuda"
    scaler = build_scaler(use_amp)

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    if resume is not None:
        checkpoint = load_training_checkpoint(resume, model, optimizer, scaler, map_location=device)
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        if is_main_process(rank):
            print(f"Resumed from {resume} at epoch {start_epoch}", flush=True)

    batches_per_epoch = len(train_loader)
    grad_accum_steps = max(
        1,
        math.ceil(float(cfg.train.target_global_batch_size) / float(int(cfg.train.batch_size) * world_size)),
    )
    total_steps = int(cfg.train.num_epochs) * batches_per_epoch
    total_updates = math.ceil(total_steps / grad_accum_steps)
    warmup_updates = math.ceil((int(cfg.train.warmup_epochs) * batches_per_epoch) / grad_accum_steps)
    optimizer.zero_grad(set_to_none=True)

    if is_main_process(rank):
        metadata = {
            "parameters": total_params,
            "grad_accum_steps": grad_accum_steps,
            "world_size": world_size,
            "batches_per_epoch": batches_per_epoch,
            "total_updates": total_updates,
            "warmup_updates": warmup_updates,
        }
        (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
        print(json.dumps(metadata, indent=2), flush=True)
        append_jsonl(output_dir / "metrics.jsonl", {"event": "metadata", **metadata})

    for epoch in range(start_epoch, int(cfg.train.num_epochs)):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        progress = tqdm(train_loader, disable=not is_main_process(rank), desc=f"epoch {epoch}")
        running_loss = []
        model.train()

        for batch_idx, batch in enumerate(progress):
            videos = batch["sequence"].to(device, non_blocking=True).float()
            update_idx = global_step // grad_accum_steps
            lr = cosine_lr(
                optimizer,
                update=update_idx,
                warmup_updates=warmup_updates,
                total_updates=total_updates,
                base_lr=float(cfg.train.lr),
                min_lr=float(cfg.train.min_lr),
            )

            with get_autocast(device, use_amp, str(cfg.train.get("amp_dtype", "float16"))):
                loss = model(videos)["loss"]
                scaled_loss = loss / grad_accum_steps

            scaler.scale(scaled_loss).backward()

            should_step = (global_step + 1) % grad_accum_steps == 0 or batch_idx + 1 == batches_per_epoch
            if should_step:
                if float(cfg.train.get("clip_grad_norm", 0.0)) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.clip_grad_norm))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            reduced_loss = reduce_mean(loss.detach(), world_size)
            running_loss.append(reduced_loss.cpu())
            global_step += 1

            if is_main_process(rank):
                progress.set_postfix(loss=f"{reduced_loss.item():.4f}", lr=f"{lr:.2e}")
                log_every = int(cfg.train.get("log_every_steps", 50))
                if log_every > 0 and (global_step == 1 or global_step % log_every == 0):
                    append_jsonl(
                        output_dir / "metrics.jsonl",
                        {
                            "event": "train_step",
                            "epoch": epoch,
                            "batch_idx": batch_idx,
                            "global_step": global_step,
                            "optimizer_update": update_idx,
                            "loss": float(reduced_loss.item()),
                            "lr": float(lr),
                        },
                    )

            max_train_batches = cfg.train.get("max_train_batches", None)
            if max_train_batches is not None and batch_idx + 1 >= int(max_train_batches):
                break

        train_loss = float(torch.stack(running_loss).mean().item()) if running_loss else float("nan")
        val_loss = validate(model, val_loader, device, cfg, world_size)

        if is_main_process(rank):
            print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}", flush=True)
            append_jsonl(
                output_dir / "metrics.jsonl",
                {
                    "event": "epoch",
                    "epoch": epoch,
                    "global_step": global_step,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": min(best_val_loss, val_loss),
                },
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(output_dir / "checkpoint_best.pt", model, optimizer, scaler, epoch, global_step, best_val_loss, cfg)
                save_encoder(output_dir / "encoder_best.pt", model, cfg)
            last_path = output_dir / "checkpoint_last.pt"
            save_checkpoint(last_path, model, optimizer, scaler, epoch, global_step, best_val_loss, cfg)
            save_encoder(output_dir / "encoder_last.pt", model, cfg)

    cleanup_distributed(world_size)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a VideoMAE-style masked autoencoder baseline.")
    parser.add_argument("config", type=str)
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    OmegaConf.set_struct(cfg, False)
    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
