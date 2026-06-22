#!/bin/bash
# Sequential finite-T (softmode) grid over all 5 foundation MLIPs, each in its own env.
# Resumable: already-recorded (system,model,T) units are skipped.
cd /root/mlip-dynamic-stability
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
declare -A PY=(
  [mace_mp0]=/venv/main/bin/python
  [chgnet]=/root/env-chgnet/bin/python
  [orb_v2]=/root/env-orb/bin/python
  [sevennet0]=/root/env-sevennet/bin/python
  [mattersim]=/root/env-mattersim/bin/python
)
for M in mace_mp0 sevennet0 mattersim chgnet orb_v2; do
  echo "===== MODEL $M $(date) ====="
  ${PY[$M]} -u scripts/run_grid.py --model "$M" --layer finite_t --temps 100 300 600 900
done
echo "ALL_FT_DONE $(date)"
