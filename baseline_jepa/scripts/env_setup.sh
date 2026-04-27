#!/bin/bash
# Edit this file to match your computing environment, then source it from the
# other scripts (it is sourced automatically by all scripts in this directory).

# Load Python module if your cluster uses environment modules, e.g.:
# module load python/3.11.7

# Activate your virtual environment:
# source /path/to/your/venv/bin/activate
conda activate jepa_physics

# Navigate to the project root:
cd /home/zihan/Physics-Representation-Learning

# Set the path to The Well datasets:
export THE_WELL_DATA_DIR=/home/zl4701/Physics-Representation-Learning/dataset
