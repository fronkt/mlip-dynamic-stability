#!/bin/bash
# C1 + C5 SSCHA seed/diagnostic study; one env per invocation (preset skips non-importable units).
cd /root/mlip-dynamic-stability
export OMP_NUM_THREADS=$2 MKL_NUM_THREADS=$2 OPENBLAS_NUM_THREADS=$2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
echo "===== C1 preset env=$1 $(date -u)"
timeout 57600 /root/env-$1/bin/python -u scripts/sscha_seed_study.py --preset revision --device cuda || echo "[error] C1 $1 rc=$?"
echo "C1_DONE $1 $(date -u)"
