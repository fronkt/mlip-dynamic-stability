#!/bin/bash
# Extended grid worker (2026-10-03): like grid_run.sh, but runs scripts/sscha_seed_study_grid2.py,
# a second copy of scripts/sscha_seed_study.py carrying the 600/900 K non-bcc set in GRID_SETS.
# The first copy is still running under other processes and is hashed per unit, so it is not
# replaced; the per-(unit, start) locks are shared, so both copies split the work without overlap.
# Usage: grid2_run.sh <env> <model> <k>
cd /root/mlip-dynamic-stability || exit 1
mkdir -p /root/logs
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
env=$1 model=$2 k=$3
shift 3
timeout --kill-after=120 48h /root/env-$env/bin/python -u scripts/sscha_seed_study_grid2.py \
  --preset grid --model "$model" --device cuda "$@" >> /root/logs/grid2_${env}_$k.log 2>&1 < /dev/null
echo "WORKER_EXIT rc=$? $(date -u)" >> /root/logs/grid2_${env}_$k.log
