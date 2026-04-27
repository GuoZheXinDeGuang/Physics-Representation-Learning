# CSCI-GA 2572 Deep Learning Final Project — Physics Representation Learning

This repo contains three self-supervised representation learning approaches for spatiotemporal physical systems, all evaluated on The Well's `active_matter` dataset (11 physical channels, 16-frame windows at 224×224, regressing `alpha` and `zeta`):

- **`baseline_jepa/`** — Convolutional JEPA baseline ([arXiv:2603.13227](https://arxiv.org/abs/2603.13227))
- **`MAE/`** — VideoMAE-style masked autoencoder
- **`Dynamo/`** — DynaMo in-domain dynamics pretraining (inspired by the paper [arXiv:2409.12192](https://arxiv.org/abs/2409.12192))

---

## Common setup

**Requirements:** Python 3.10+ (3.8 for Dynamo), PyTorch 2.0+ with CUDA.

```bash
git clone https://github.com/GuoZheXinDeGuang/Physics-Representation-Learning
cd Physics-Representation-Learning
```

### Dataset (shared by all three solutions)

```bash
pip install the_well
the-well-download --base-path /your/path/to/the_well --dataset active_matter --split train
the-well-download --base-path /your/path/to/the_well --dataset active_matter --split valid
the-well-download --base-path /your/path/to/the_well --dataset active_matter --split test
```

Both layouts are supported:

```
/your/path/to/the_well/active_matter/data/{train,valid,test}
/your/path/to/the_well/datasets/active_matter/data/{train,valid,test}
```

---

## 1. Baseline JEPA (`baseline_jepa/`)

### Install

```bash
cd baseline_jepa
conda create -n jepa_physics python=3.10
conda activate jepa_physics
pip install -r requirements.txt
```

### DataLoader (sanity check)

```python
from physics_jepa.data import get_train_sequence_dataloader

train_loader = get_train_sequence_dataloader(
    dataset_name="active_matter",
    num_frames=16,
    num_examples=None,
    batch_size=8,
    resolution=224,
    offset=1,
    noise_std=0.0,
)
```

### Train

Set the dataset path inside `configs/train_activematter_small.yaml`. Single GPU:

```bash
CUDA_VISIBLE_DEVICES=1 python -m physics_jepa.train_jepa \
    configs/train_activematter_small.yaml \
    train.num_epochs=6 \
    train.batch_size=3
```

Multi-GPU (e.g. 4 × 3090):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --standalone \
    -m physics_jepa.train_jepa \
    configs/train_activematter_small.yaml \
    train.num_epochs=6 train.batch_size=4
```

### Frozen evaluation

```bash
CUDA_VISIBLE_DEVICES=1 python -m physics_jepa.eval_frozen_regression \
    configs/train_activematter_small.yaml \
    ft.batch_size=16 \
    --trained_model_path checkpoints/active_matter-16frames-cnn-jepa-noise-std-1.0_2026-04-18_20-54-10/ConvEncoder_5.pth
```

Following the reference: 100 epochs, lr `1e-3`, weight decay `1e-4`. Change probe batch size via `ft.batch_size` (or `ft/linear` in the config).

---

## 2. MAE (`MAE/`)

Self-contained VideoMAE-style MAE that imports `physics_jepa`'s sequence dataset and a local VideoMAE encoder. Does not modify `baseline_jepa`.

### Install

Reuses the `jepa_physics` env from §1:

```bash
cd Physics-Representation-Learning   # repo root
conda activate jepa_physics
```

If you skipped §1:

```bash
conda create -n jepa_physics python=3.10
conda activate jepa_physics
pip install -r baseline_jepa/requirements.txt
```

### Configure dataset path

Set `paths.the_well_data_dir` in `MAE/configs/train_active_matter_mae.yaml`. If the path contains `datasets/active_matter`, MAE auto-resolves it — no need to edit `scripts/env_setup.sh`.

### Train

Wrapper script (activates env, exports `PYTHONPATH`, forwards overrides):

```bash
MAE/scripts/run_train_mae.sh
```

Equivalent direct command:

```bash
conda activate jepa_physics
export PYTHONPATH=.
python -m MAE.train MAE/configs/train_active_matter_mae.yaml
```

Common overrides:

```bash
MAE/scripts/run_train_mae.sh train.num_epochs=10 train.batch_size=1 train.target_global_batch_size=16
```

Multi-GPU (4 GPUs):

```bash
conda activate jepa_physics
export PYTHONPATH=.
torchrun --nproc_per_node=4 --standalone \
    -m MAE.train \
    MAE/configs/train_active_matter_mae.yaml \
    train.batch_size=2
```

Outputs go to `MAE/checkpoints/<timestamp>/`, containing `checkpoint_{last,best}.pt`, `encoder_{last,best}.pt`, `config.yaml`, `run_metadata.json`, and `metrics.jsonl` (one line per epoch; default `train.log_every_steps=50`).

Resume:

```bash
PYTHONPATH=. python -m MAE.train \
    MAE/configs/train_active_matter_mae.yaml \
    --resume MAE/checkpoints/<run_name>/checkpoint_last.pt
```

### Frozen evaluation

```bash
MAE/scripts/run_eval_frozen_regression.sh MAE/checkpoints/<run_name>/checkpoint_best.pt
```

Equivalent direct command:

```bash
conda activate jepa_physics
export PYTHONPATH=.
python -m MAE.eval_frozen_regression \
    MAE/configs/train_active_matter_mae.yaml \
    --checkpoint MAE/checkpoints/<run_name>/checkpoint_best.pt
```

Evaluation uses mean-pooled unmasked encoder patch embeddings. Labels `(alpha, zeta)` are z-scored with means `[-3.0, 9.0]` and stds `[1.41, 5.16]`. Results land in `eval_results.json` next to the checkpoint (override with `--save_path`).

### Defaults

```
encoder:        physics_jepa.videomae.vit_small_patch16_224
input:          11 channels, 16 frames, 224x224
patch size:     16
tubelet size:   2
mask ratio:     0.9
decoder dim/depth/heads: 192 / 4 / 3
```

Training script aborts if model > 100M params.

### Smoke test

```bash
conda activate jepa_physics
PYTHONPATH=. python MAE/tests/test_mae_smoke.py
```

---

## 3. DynaMo (`Dynamo/`)

DynaMo dynamics pretraining adapted to `active_matter`. `train_thewell.py` reuses `train.py::Trainer` and only overrides the dataset path so the official train/valid splits from The Well are used (no random re-splitting, no `TrajectorySlicerDataset` wrapping — windows are already produced as `(T, V=1, C=11, H, W)`). We use the ideas of Dynamo to learn robust physics representations. 

### Install

Separate conda env (Python 3.8):

```bash
cd Dynamo
conda env create --file=conda_env.yml
conda activate dynamo-repro
```

Optional logging:

```bash
wandb login
# or disable:
export WANDB_MODE=disabled
```

### Configure dataset path

Edit `Dynamo/configs/env_vars/env_vars.yaml`:

```yaml
dataset_root: /your/path/to/the_well/datasets

datasets:
    active_matter: ${env_vars.dataset_root}/active_matter
```

### Train

```bash
conda activate dynamo-repro
cd Dynamo
accelerate launch train_thewell.py
```

Default `configs/train_thewell.yaml`: `batch_size=16`, `window_size=16`, `num_epochs=100`, `ssl_lr=1e-4`, warmup 5 epochs, ResNet18 physics encoder, inverse-dynamics projector, online linear + kNN probes on `(alpha, zeta)` every 50 epochs.

Hydra overrides:

```bash
accelerate launch train_thewell.py num_epochs=50 batch_size=8
```

Snapshots → `./exp_local/{date}/{time}_thewell_dynamo/encoder.pt`.

### Frozen evaluation

Standalone, mirrors the baseline JEPA's eval (same label stats `alpha=-3/1.41`, `zeta=9/5.16`), so numbers are directly comparable:

```bash
python eval_thewell.py \
    --ckpt exp_local/2026.04.22/114527_thewell_dynamo/encoder.pt \
    --data_dir /your/path/to/the_well/datasets/active_matter
```

Useful flags:

```
--window_size 16
--resolution 224
--stride <int>            # default = window_size (non-overlapping)
--batch_size 16
--linear_epochs 200
--linear_lr 1e-3
--linear_weight_decay 1e-4
--knn_ks 1 3 5 10 20 50
--save_path <path>
```

Runs:

- **Linear probe:** single `nn.Linear` + AdamW + MSE, early-stopped on val.
- **kNN regression:** `sklearn KNeighborsRegressor(weights="distance")`, best `k` from val.
- **Features:** per-window mean-pool of encoder output over `(T, V)`.

Frozen-encoder rules from the project PDF are enforced (`requires_grad=False`, `eval()` mode, single linear layer, test split only for the final number).