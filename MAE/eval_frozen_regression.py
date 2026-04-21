from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from sklearn.neighbors import KNeighborsRegressor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .checkpoint import load_encoder_or_model
from .config import load_config, save_resolved_config, set_the_well_env
from .data import get_eval_loader
from .model import MaskedAutoencoderVideo


LABEL_STATS = {
    "names": ["alpha", "zeta"],
    "means": torch.tensor([-3.0, 9.0], dtype=torch.float32),
    "stds": torch.tensor([1.41, 5.16], dtype=torch.float32),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_labels(labels: torch.Tensor) -> torch.Tensor:
    means = LABEL_STATS["means"].to(labels.device)
    stds = LABEL_STATS["stds"].to(labels.device)
    return (labels.float() - means) / stds


@torch.no_grad()
def extract_embeddings(cfg, model: MaskedAutoencoderVideo, split: str, device: torch.device):
    loader = get_eval_loader(cfg, split)
    max_batches = cfg.eval.get(f"max_{split}_batches", cfg.eval.get("max_batches", None))
    all_embeddings = []
    all_labels = []
    model.eval()
    for batch_idx, batch in enumerate(tqdm(loader, desc=f"extract {split}")):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        videos = batch["sequence"].to(device, non_blocking=True).float()
        embeddings = model.encode(videos, pool=str(cfg.eval.get("pool", "mean")))
        labels = normalize_labels(batch["physical_params"]).cpu()
        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels)
    if not all_embeddings:
        raise RuntimeError(f"No embeddings extracted for split {split}")
    return torch.cat(all_embeddings, dim=0), torch.cat(all_labels, dim=0)


def fit_linear_regressor(cfg, train_x, train_y, val_x, val_y, device: torch.device):
    model = nn.Linear(train_x.shape[1], train_y.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.eval.linear_lr),
        weight_decay=float(cfg.eval.linear_weight_decay),
    )
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(train_x.float(), train_y.float()),
        batch_size=int(cfg.eval.linear_batch_size),
        shuffle=True,
    )

    best_state = None
    best_val = float("inf")
    history = []
    for epoch in range(int(cfg.eval.linear_epochs)):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            loss = loss_fn(model(xb), yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val_x.float().to(device)), val_y.float().to(device)).item()
        history.append({"epoch": epoch + 1, "train_mse": float(np.mean(losses)), "val_mse": float(val_loss)})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history


def fit_knn_regressor(cfg, train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray):
    ks = [int(k) for k in cfg.eval.knn_k]
    ks = [k for k in ks if 1 <= k <= len(train_x)]
    if not ks:
        ks = [1]

    best_model = None
    best_k = None
    best_val = float("inf")
    trials = []
    for k in ks:
        model = KNeighborsRegressor(n_neighbors=k, weights="distance")
        model.fit(train_x, train_y)
        pred = model.predict(val_x)
        val_mse = float(((pred - val_y) ** 2).mean())
        trials.append({"k": k, "val_mse": val_mse})
        if val_mse < best_val:
            best_val = val_mse
            best_model = model
            best_k = k
    return best_model, best_k, trials


def mse_report(pred: np.ndarray, target: np.ndarray) -> dict:
    per_dim = ((pred - target) ** 2).mean(axis=0)
    report = {"mse": float(((pred - target) ** 2).mean())}
    for idx, name in enumerate(LABEL_STATS["names"]):
        report[f"mse_{name}"] = float(per_dim[idx])
    return report


def default_save_path(cfg, checkpoint_path: str | Path) -> Path:
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.parent.name:
        return checkpoint_path.parent / "eval_results.json"
    return Path(cfg.paths.output_dir) / "eval_results.json"


def run_evaluation(cfg, checkpoint_path: str | Path, save_path: str | Path | None = None) -> dict:
    set_the_well_env(cfg)
    set_seed(int(cfg.eval.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MaskedAutoencoderVideo.from_config(cfg).to(device)
    load_encoder_or_model(checkpoint_path, model, map_location=device)
    model.eval()

    train_x, train_y = extract_embeddings(cfg, model, "train", device)
    val_x, val_y = extract_embeddings(cfg, model, "val", device)
    test_x, test_y = extract_embeddings(cfg, model, "test", device)

    linear_model, linear_history = fit_linear_regressor(cfg, train_x, train_y, val_x, val_y, device)
    with torch.no_grad():
        linear_val_pred = linear_model(val_x.float().to(device)).cpu().numpy()
        linear_test_pred = linear_model(test_x.float().to(device)).cpu().numpy()

    train_x_np = train_x.numpy()
    train_y_np = train_y.numpy()
    val_x_np = val_x.numpy()
    val_y_np = val_y.numpy()
    test_x_np = test_x.numpy()
    test_y_np = test_y.numpy()

    knn_model, best_k, knn_trials = fit_knn_regressor(cfg, train_x_np, train_y_np, val_x_np, val_y_np)
    knn_val_pred = knn_model.predict(val_x_np)
    knn_test_pred = knn_model.predict(test_x_np)

    results = {
        "dataset": cfg.dataset.name,
        "checkpoint": str(checkpoint_path),
        "embedding_shape": list(train_x.shape[1:]),
        "encoder_input": {
            "channels": int(cfg.dataset.num_chans),
            "frames": int(cfg.dataset.num_frames),
            "resolution": int(cfg.dataset.resolution),
        },
        "label_names": LABEL_STATS["names"],
        "label_stats": {
            "means": LABEL_STATS["means"].tolist(),
            "stds": LABEL_STATS["stds"].tolist(),
        },
        "linear": {
            "val": mse_report(linear_val_pred, val_y_np),
            "test": mse_report(linear_test_pred, test_y_np),
            "history": linear_history,
        },
        "knn": {
            "best_k": best_k,
            "val": mse_report(knn_val_pred, val_y_np),
            "test": mse_report(knn_test_pred, test_y_np),
            "trials": knn_trials,
        },
    }

    out_path = Path(save_path) if save_path is not None else default_save_path(cfg, checkpoint_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    save_resolved_config(cfg, out_path.parent / "eval_config.yaml")
    print(json.dumps(results, indent=2), flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen linear/kNN regression evaluation for MAE encoders.")
    parser.add_argument("config", type=str)
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    OmegaConf.set_struct(cfg, False)
    run_evaluation(cfg, args.checkpoint, save_path=args.save_path)


if __name__ == "__main__":
    main()
