# VideoMAE-Style MAE Baseline

This folder contains a self-contained masked autoencoder baseline for the final project. It does not modify `physics_jepa`; it imports the existing sequence dataset implementation and the local VideoMAE-style encoder.

## Data

The default The Well base path is:

```bash
/home_shared/grail_enoch/the_well
```

Both of these active matter layouts are supported:

```bash
/home_shared/grail_enoch/the_well/active_matter/data/train
/home_shared/grail_enoch/the_well/active_matter/data/valid
/home_shared/grail_enoch/the_well/active_matter/data/test
```

```bash
/home_shared/grail_enoch/the_well/datasets/active_matter/data/train
/home_shared/grail_enoch/the_well/datasets/active_matter/data/valid
/home_shared/grail_enoch/the_well/datasets/active_matter/data/test
```

The MAE entrypoints set `THE_WELL_DATA_DIR` from `paths.the_well_data_dir` in `MAE/configs/train_active_matter_mae.yaml`. If the configured path contains a `datasets/active_matter` folder, MAE automatically resolves it before calling the existing `physics_jepa` dataloader, so the top-level `scripts/env_setup.sh` does not need to be edited.

## Train

```bash
MAE/scripts/run_train_mae.sh
```

Equivalent direct command:

```bash
conda activate jepa_physics
export PYTHONPATH=.
PYTHONPATH=. python -m MAE.train MAE/configs/train_active_matter_mae.yaml
```

Common overrides:

```bash
MAE/scripts/run_train_mae.sh train.num_epochs=10 train.batch_size=1 train.target_global_batch_size=16
```

For multi-GPU training:

```bash
conda activate jepa_physics
export PYTHONPATH=.
torchrun --nproc_per_node=4 --standalone \
  -m MAE.train \
  MAE/configs/train_active_matter_mae.yaml \
  train.batch_size=2
```

Outputs are saved under a timestamp-only directory such as `MAE/checkpoints/2026-04-19_23-16-03/`:

```text
checkpoint_last.pt
checkpoint_best.pt
encoder_last.pt
encoder_best.pt
config.yaml
run_metadata.json
metrics.jsonl
```

`metrics.jsonl` contains metadata, periodic training-step metrics, and one epoch summary per line. The default logging interval is `train.log_every_steps=50`.

Resume a preempted run with:

```bash
PYTHONPATH=. python -m MAE.train \
  MAE/configs/train_active_matter_mae.yaml \
  --resume MAE/checkpoints/<run_name>/checkpoint_last.pt
```

## Frozen Evaluation

Run frozen linear probing and kNN regression:

```bash
MAE/scripts/run_eval_frozen_regression.sh MAE/checkpoints/<run_name>/checkpoint_best.pt
```

Equivalent direct command:

```bash
conda activate jepa_physics
export PYTHONPATH=.
PYTHONPATH=. python -m MAE.eval_frozen_regression \
  MAE/configs/train_active_matter_mae.yaml \
  --checkpoint MAE/checkpoints/<run_name>/checkpoint_best.pt
```

Evaluation uses mean-pooled unmasked encoder patch embeddings. Labels are treated as continuous regression targets in `alpha, zeta` order and z-score normalized with means `[-3.0, 9.0]` and standard deviations `[1.41, 5.16]`.

Results are written to `eval_results.json` next to the checkpoint unless `--save_path` is provided.

## Defaults

The default model uses:

```text
encoder: physics_jepa.videomae.vit_small_patch16_224
input: 11 channels, 16 frames, 224x224
patch size: 16
tubelet size: 2
mask ratio: 0.9
decoder dim/depth/heads: 192 / 4 / 3
```

The training script reports total parameter count and fails if the model reaches the project limit of 100M parameters.

## Tests

The tests use synthetic HDF5 data and a tiny debug encoder so they can run without the full dataset:

```bash
conda activate jepa_physics
PYTHONPATH=. python MAE/tests/test_mae_smoke.py
```
