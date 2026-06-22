#!/bin/bash
# Multi-mode SSCHA dynamic-stabilization curve for bcc-Zr across all 5 MLIPs.
# Maps the dynamic-stabilization temperature T_dyn (free-energy Hessian turns positive) per
# model. bcc->hcp is martensitic, so T_dyn (dynamic) differs from the thermodynamic Tc; the
# cross-model spread in T_dyn is the finding. Resumable (ledger skips recorded units).
cd /root/mlip-dynamic-stability
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
declare -A PY=(
  [mace_mp0]=/venv/main/bin/python
  [sevennet0]=/root/env-sevennet/bin/python
  [mattersim]=/root/env-mattersim/bin/python
  [chgnet]=/root/env-chgnet/bin/python
  [orb_v2]=/root/env-orb/bin/python
)
for M in mattersim sevennet0 mace_mp0 chgnet orb_v2; do
  for T in 50 100 200 300 600; do
    echo "===== $M zr_bcc T=$T $(date) ====="
    ${PY[$M]} -u -m mlip_dynstab.cli --system zr_bcc --model "$M" --method sscha --T "$T"
  done
done
echo "SSCHA_ZR_DONE $(date)"
