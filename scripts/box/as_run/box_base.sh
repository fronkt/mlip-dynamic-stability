#!/bin/bash
# Base layer for the RSC-revision box (idempotent). The full setup_rsc_box.sh re-runs these safely.
set -x
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq gfortran libblas-dev liblapack-dev build-essential tmux rsync git curl bzip2 pkg-config > /root/apt.log 2>&1
echo "APT_DONE rc=$?"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=/root/.local/bin:$PATH
uv --version
cd /root
if [ ! -x /root/bin/micromamba ]; then
  mkdir -p /root/bin
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /root bin/micromamba
fi
export MAMBA_ROOT_PREFIX=/root/mamba
if [ ! -x /root/qe/bin/pw.x ]; then
  /root/bin/micromamba create -y -p /root/qe -c conda-forge qe > /root/qe_install.log 2>&1
fi
ls -la /root/qe/bin/pw.x && echo "QE_DONE"
echo "BASE_DONE $(date -u)"
