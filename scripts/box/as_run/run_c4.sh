#!/bin/bash
# C4: displacement-amplitude sweep for the four remaining models, SEQUENTIAL (one ledger writer).
cd /root/mlip-dynamic-stability
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for m in mace_mp0 mattersim sevennet0 orb_v2; do
  case $m in mace_mp0) e=mace;; mattersim) e=mattersim;; sevennet0) e=sevennet;; orb_v2) e=orb;; esac
  echo "===== C4 $m $(date -u)"
  timeout 14400 /root/env-$e/bin/python -u scripts/run_disp_sweep.py --models $m --device cuda || echo "[error] C4 $m"
done
echo "C4_DONE $(date -u)"
