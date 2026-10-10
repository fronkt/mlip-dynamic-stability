#!/bin/bash
# C3b on the box: PBE on SSCHA-sampled configurations (R1.2). Inputs + MLIP evaluation now;
# the pw.x jobs start after the C3a queue ('^a_') has finished, so the two never share cores.
cd /root/mlip-dynamic-stability
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
py() { case $1 in mace_mp0) echo /root/env-mace/bin/python;; chgnet) echo /root/env-chgnet/bin/python;;
       sevennet0) echo /root/env-sevennet/bin/python;; mattersim) echo /root/env-mattersim/bin/python;;
       orb_v2) echo /root/env-orb/bin/python;; esac; }
run() { echo "===== $* $(date -u)"; timeout 14400 "$@"; rc=$?; [ $rc -ne 0 ] && echo "[rc=$rc] $*"; return 0; }
run $(py mace_mp0) -u scripts/dft_reference.py c3b-inputs --sssp-dir /root/sssp --sssp-json /root/sssp/SSSP_1.3.0_PBE_efficiency.json \
    --kspacing-insulator 0.25 --kspacing-metal 0.15
for m in mattersim mace_mp0 chgnet sevennet0 orb_v2; do
  run $(py $m) -u scripts/dft_reference.py mlip-eval --model $m --device cuda
done
ls results/revision/dft/qe | grep -c '^b_'
echo "C3B_INPUTS_DONE $(date -u)"
until grep -q C3A_QE_DONE /root/logs/c3a.log; do sleep 60; done
export ESPRESSO_PSEUDO=/root/sssp/upf
bash scripts/box/qe_queue.sh --qe-prefix /root/qe --pseudo-dir /root/sssp/upf -r 8 -n 5 --filter '^b_' \
    --log /root/logs/qe_b.log < /dev/null
echo "C3B_QE_DONE $(date -u)"
