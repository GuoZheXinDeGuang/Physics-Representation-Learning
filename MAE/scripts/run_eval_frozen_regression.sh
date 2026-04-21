#!/bin/bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <checkpoint.pt> [config overrides...]"
  exit 1
fi

CHECKPOINT_PATH="$1"
shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate jepa_physics

export THE_WELL_DATA_DIR="${THE_WELL_DATA_DIR:-/home_shared/grail_enoch/the_well}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

python -m MAE.eval_frozen_regression \
  MAE/configs/train_active_matter_mae.yaml \
  --checkpoint "$CHECKPOINT_PATH" \
  "$@"
