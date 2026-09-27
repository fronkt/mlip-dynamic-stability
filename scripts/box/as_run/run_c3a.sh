#!/bin/bash
# C3a on the box: regenerate + verify the screen's paths in every model env, evaluate every model
# on every structure, write pw.x inputs, then run the R1.1 paths (a_) through the resumable queue.
# geom exits 4 when a path misses its cached map but still writes and flags it -> not fatal here.
cd /root/mlip-dynamic-stability
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
py() { case $1 in mace_mp0) echo /root/env-mace/bin/python;; chgnet) echo /root/env-chgnet/bin/python;;
       sevennet0) echo /root/env-sevennet/bin/python;; mattersim) echo /root/env-mattersim/bin/python;;
       orb_v2) echo /root/env-orb/bin/python;; esac; }
run() { echo "===== $* $(date -u)"; timeout 14400 "$@"; rc=$?; [ $rc -ne 0 ] && echo "[rc=$rc] $*"; return 0; }
# MatterSim first: it defines the reference coordinate for models with no instability.
for m in mattersim mace_mp0 chgnet sevennet0 orb_v2; do
  run $(py $m) -u scripts/dft_reference.py geom --model $m --device cuda
done
for m in mattersim mace_mp0 chgnet sevennet0 orb_v2; do
  run $(py $m) -u scripts/dft_reference.py mlip-eval --model $m --device cuda
done
run $(py mace_mp0) -u scripts/dft_reference.py qe-inputs --sssp-dir /root/sssp --sssp-json /root/sssp/SSSP_1.3.0_PBE_efficiency.json \
    --kspacing-insulator 0.25 --kspacing-metal 0.15
ls results/revision/dft/qe | wc -l
echo "C3A_INPUTS_DONE $(date -u)"
export ESPRESSO_PSEUDO=/root/sssp/upf
bash scripts/box/qe_queue.sh --qe-prefix /root/qe --pseudo-dir /root/sssp/upf -r 8 -n 4 --filter '^a_' \
    --log /root/logs/qe_a.log < /dev/null
echo "C3A_QE_DONE $(date -u)"
