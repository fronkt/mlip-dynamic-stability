#!/bin/bash
# C2 (R2.2) on the box: fc for five models -> freeze configs -> forces for five models and the
# three committee members. Sequential; resumable (force_spread.py skips finished work).
cd /root/mlip-dynamic-stability
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
py() { case $1 in mace_mp0) echo /root/env-mace/bin/python;; chgnet) echo /root/env-chgnet/bin/python;;
       sevennet0) echo /root/env-sevennet/bin/python;; mattersim) echo /root/env-mattersim/bin/python;;
       orb_v2) echo /root/env-orb/bin/python;; esac; }
run() { echo "===== $* $(date -u)"; timeout 5400 "$@" || echo "[error] $*"; }
for m in mace_mp0 chgnet sevennet0 mattersim orb_v2; do
  run $(py $m) -u scripts/force_spread.py --stage fc --model $m --device cuda
done
run $(py mace_mp0) -u scripts/force_spread.py --stage configs
for m in mace_mp0 chgnet sevennet0 mattersim orb_v2; do
  run $(py $m) -u scripts/force_spread.py --stage forces --model $m --device cuda
done
run $(py mace_mp0) -u scripts/force_spread.py --stage forces --model mace_mp0 --ckpt small --device cuda
run $(py mace_mp0) -u scripts/force_spread.py --stage forces --model mace_mp0 --ckpt large --device cuda
run $(py mattersim) -u scripts/force_spread.py --stage forces --model mattersim --ckpt mattersim-v1.0.0-1m --device cuda
run $(py mace_mp0) -u scripts/force_spread.py --stage analyze
echo "C2_DONE $(date -u)"
