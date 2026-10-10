#!/bin/bash
cd /root/mlip-dynamic-stability || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
exec timeout --kill-after=120 6h /root/env-mace/bin/python -u scripts/sscha_seed_study.py --converge --system batio3_cubic --model mace_mp0 --T 100 --supercell 2 2 2 --start B --start-b-thz 1.0 --device cuda --unit-timeout 10800 --out-dir results/revision/sscha_converged_b1 >> /root/logs/c1c_bto_B1.log 2>&1
