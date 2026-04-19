# CSCI-GA 2572 Deep Learning Final Project

Baseline Solution - Implementation on Representation Learning for Spatiotemporal Physical Systems](https://arxiv.org/abs/2603.13227).

## Installation

**Requirements:** Python 3.10+, PyTorch 2.0+ with CUDA.

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-org/physics_jepa_public
cd physics_jepa_public
conda create -n jepa_physics python=3.10
# pip install torch torchvision einops omegaconf wandb tqdm h5py psutil scikit-learn timm the-well
# pip install -e .
pip install -r requirements.txt
```

Pareparing for the dataset
```bash
pip install the_well
the-well-download --base-path /your/path/to/the_well --dataset active_matter --split train
the-well-download --base-path /your/path/to/the_well --dataset active_matter --split valid
the-well-download --base-path /your/path/to/the_well --dataset active_matter --split test
```

DataLoader
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

Before run, please make sure to set the path of your dataset. Here is the command to run:
```bash
CUDA_VISIBLE_DEVICES=1 python -m physics_jepa.train_jepa   configs/train_activematter_small.yaml   train.num_epochs=6   train.batch_size=3
```
Right now I only use one 3090 GPU to run this, if you have more GPU, say you have 4, you can increase the batch size and run like the following:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --standalone \
    -m physics_jepa.train_jepa \
    configs/train_activematter_small.yaml \
    train.num_epochs=6 train.batch_size=4
```

How to evaluate based on baseline JEPA:
```bash
CUDA_VISIBLE_DEVICES=1 python -m physics_jepa.eval_frozen_regression \
  configs/train_activematter_small.yaml \
  ft.batch_size=16 \
  --trained_model_path checkpoints/active_matter-16frames-cnn-jepa-noise-std-1.0_2026-04-18_20-54-10/ConvEncoder_5.pth
```
Here we follow the value of this, we use 100 epochs, learning rate at 1e-3 and weight decay at 1e-4. You cab change the batch size at ft/linear.


### Below are from the original Repo

### Environment setup

Edit `scripts/env_setup.sh` to activate your virtual environment and set the path to [The Well](https://github.com/PolymathicAI/the_well) datasets. This file is sourced automatically by all scripts. The `THE_WELL_DATA_DIR` variable is required by all training and finetuning scripts that use The Well data.

## Training

### 1. JEPA pretraining

Pretrain a convolutional JEPA encoder on a physics dataset using the scripts in `scripts/<dataset>/`:

| Dataset | Script |
|---|---|
| Shear flow | `scripts/shear_flow/run_train_jepa.sh` |
| Rayleigh-Bénard | `scripts/rayleigh_benard/run_train_jepa.sh` |
| Active matter | `scripts/active_matter/run_train_jepa.sh` |

Config fields `out_path` and `cache_path` control where checkpoints and dataset caches are written. Key training hyperparameters (learning rate, number of epochs, noise level, etc.) are set in the `train:` block of the corresponding config. Config fields can be overridden from the command line by passing `key=value` arguments to the script, e.g.:

```bash
scripts/shear_flow/run_train_jepa.sh train.num_epochs=10 train.lr=5e-4
```

### 2. VideoMAE finetuning (baseline)

Fine-tune a pretrained [VideoMAE](https://github.com/MCG-NJU/VideoMAE) backbone for physical parameter estimation. Set the `CHECKPOINT_PATH` environment variable to the pretrained VideoMAE checkpoint and run the appropriate script:

| Dataset | Script |
|---|---|
| Shear flow | `scripts/shear_flow/run_finetune_videomae.sh` |
| Rayleigh-Bénard | `scripts/rayleigh_benard/run_finetune_videomae.sh` |
| Active matter | `scripts/active_matter/run_finetune_videomae.sh` |

### 3. JEPA finetuning (parameter estimation)

Fine-tune a pretrained JEPA encoder for physical parameter estimation. Set `CHECKPOINT_PATH` to a saved encoder checkpoint and run the appropriate script:

| Dataset | Script |
|---|---|
| Shear flow | `scripts/shear_flow/run_finetune_jepa.sh` |
| Rayleigh-Bénard | `scripts/rayleigh_benard/run_finetune_jepa.sh` |
| Active matter | `scripts/active_matter/run_finetune_jepa.sh` |

The same configs used for pretraining are reused here; the `ft:` block controls finetuning hyperparameters. A multi-GPU variant is available at `scripts/shear_flow/run_finetune_jepa_ddp.sh`.

## Baselines

### 4. DISCO finetuning

[DISCO](https://arxiv.org/abs/2401.09246) is a latent-space parameter estimation baseline. It operates on precomputed DISCO latent representations rather than raw data. Pass the path to a directory of DISCO inference outputs as the first argument:

```bash
scripts/run_finetune_disco.sh /path/to/disco_inference_shear_flow
```

The data directory name must match one of the dataset keys in `physics_jepa/baselines/disco.py` (e.g. `disco_inference_shear_flow`, `disco_inference_rayleigh_benard`, `disco_inference_active_matter`).

### 5. MPP finetuning

Fine-tune a pretrained [MPP](https://github.com/PolymathicAI/multiple_physics_pretraining) (Multiple Physics Pretraining) model for physical parameter estimation. Pass the dataset name and path to a pretrained MPP checkpoint:

```bash
scripts/run_mpp_param_estimation.sh shear_flow /path/to/MPP_AViT_Ti
```

`--dataset_name` should match the corresponding dataset directory name in `THE_WELL_DATA_DIR`. The checkpoint save directory can be controlled via the `CHECKPOINT_DIR` environment variable (defaults to `./checkpoints`).
