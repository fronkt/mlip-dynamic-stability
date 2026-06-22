#!/bin/bash
# Extend the multi-mode SSCHA dynamic-stabilization curves to bcc-Ti and bcc-Hf across all 5
# MLIPs (same 2x2x2 cell + T-ladder as bcc-Zr). Resumable: recorded (system,model,T) skip.
cd /root/mlip-dynamic-stability
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
declare -A PY=(
  [mace_mp0]=/venv/main/bin/python
  [sevennet0]=/root/env-sevennet/bin/python
  [mattersim]=/root/env-mattersim/bin/python
  [chgnet]=/root/env-chgnet/bin/python
  [orb_v2]=/root/env-orb/bin/python
)
for S in ti_bcc hf_bcc; do
  for M in mattersim sevennet0 mace_mp0 chgnet orb_v2; do
    for T in 50 100 200 300 600; do
      echo "===== $M $S T=$T $(date) ====="
      ${PY[$M]} -u -m mlip_dynstab.cli --system "$S" --model "$M" --method sscha --T "$T"
    done
  done
done
echo "SSCHA_TIHF_DONE $(date)"
