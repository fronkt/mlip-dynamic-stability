#!/bin/bash
# C1c converged grid, one worker process: grid_run.sh <env> <model> <k>   (k only names the log)
# Workers of one env share the units through the per-(unit, start) flock in
# results/revision/sscha_converged_grid/locks/, so start as many per env as GPU memory allows.
# Runs the copy scripts/sscha_seed_study_grid.py, not scripts/sscha_seed_study.py: the C1c
# processes were started from the latter and hash it on disk at every unit (provenance()), so it
# must not change under them. Extra arguments after <k> are passed on (e.g. --retry-failed).
cd /root/mlip-dynamic-stability || exit 1
mkdir -p /root/logs
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
env=$1 model=$2 k=$3
shift 3
exec timeout --kill-after=120 48h /root/env-$env/bin/python -u scripts/sscha_seed_study_grid.py \
  --preset grid --model "$model" --device cuda "$@" >> /root/logs/grid_${env}_$k.log 2>&1 < /dev/null
