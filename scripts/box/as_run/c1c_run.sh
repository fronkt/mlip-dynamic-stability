#!/bin/bash
# C1c converged SSCHA, one (env, start) per process. usage: c1c_run.sh <mace|mattersim> <A|B>
cd /root/mlip-dynamic-stability || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
exec timeout --kill-after=120 14h /root/env-$1/bin/python -u scripts/sscha_seed_study.py \
  --preset converged --start "$2" --device cuda --unit-timeout 10800 >> /root/logs/c1c_$1_$2.log 2>&1
