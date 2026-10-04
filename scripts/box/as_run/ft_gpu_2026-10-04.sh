#!/bin/bash
# Fine-tuning trial (E5, tasks/preregistration-finetune-2026-10-03.md) on a fresh vast GPU box,
# 2026-10-04. Pytorch 2.6/cu124 image, NOT Blackwell (the pinned torch has no sm_120 kernels).
#
#   tmux new -d -s ft 'bash /root/ft_gpu_2026-10-04.sh run; echo'
#   tmux new -d -s watchdog 'bash /root/ft_gpu_2026-10-04.sh watchdog; echo'
#
# run:      bootstrap (apt, uv, clone, env-mace + env-chgnet from the August locks, smoke) ->
#           wait for /root/ft_ready (the laptop touches it after pushing the last ft_ PBE outputs;
#           gives up after 4 h) -> git pull, require 246/246 ft_ JOB DONE -> the committed runbook
#           results/revision/finetune/train/box_sequence.sh verbatim (its STEP 1 QE queue is a no-op
#           when every ft_ job is done) -> RUN_DONE -> 45 min grace for the pull (/root/pulled ends
#           it early) -> stop this instance. Any exit of the runner stops the instance (trap).
# watchdog: stops the instance if the runner's tmux session is gone or the box is 9 h old.
#
# Stop = PUT state=stopped with this container's own CONTAINER_API_KEY (verified on box 1,
# 2026-10-04). Stop keeps the disk; destroy only after the pull.

set -u
LOG=/root/logs/ft_runner.log
mkdir -p /root/logs

self_stop() {
  local id key
  id=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^CONTAINER_ID=//p')
  key=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^CONTAINER_API_KEY=//p')
  echo "SELF_STOP ($1) $(date -u) instance=$id" >> "$LOG"
  curl -s -w " http=%{http_code}\n" -X PUT -H "Authorization: Bearer $key" \
    -H "Content-Type: application/json" -d '{"state":"stopped"}' \
    "https://console.vast.ai/api/v0/instances/$id/" >> "$LOG" 2>&1
}

case "${1:-}" in
  run)
    trap 'self_stop "runner exited"' EXIT
    exec >> "$LOG" 2>&1
    echo "RUN start $(date -u)"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq gfortran libblas-dev liblapack-dev build-essential tmux rsync git curl \
      bzip2 pkg-config > /root/logs/apt.log 2>&1
    echo "APT_DONE rc=$?"
    command -v uv > /dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH=/root/.local/bin:$PATH
    uv python install 3.11
    cd /root
    [ -d mlip-dynamic-stability ] || git clone -b rsc-figure-fixes https://github.com/fronkt/mlip-dynamic-stability.git
    cd /root/mlip-dynamic-stability && git log --oneline -1
    for m in mace chgnet; do
      bash scripts/box/as_run/env_build.sh "$m" 3.11 > /root/logs/env_$m.log 2>&1 &
    done
    wait
    grep -h "ENVCHECK\|SSCHA_OK\|ENV_DONE" /root/logs/env_*.log
    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
    /root/env-mace/bin/python scripts/box/as_run/model_smoke.py mace_mp0 2>&1 | grep -E "SMOKE|Error" | tail -2
    /root/env-chgnet/bin/python scripts/box/as_run/model_smoke.py chgnet 2>&1 | grep -E "SMOKE|Error" | tail -2
    echo "BOOTSTRAP_DONE $(date -u)"

    for i in $(seq 1 240); do [ -e /root/ft_ready ] && break; sleep 60; done
    [ -e /root/ft_ready ] || { echo "NO_FT_DATA after 4 h $(date -u)"; exit 1; }
    git pull --ff-only && git log --oneline -1
    n=$(grep -l 'JOB DONE' results/revision/dft/qe/ft_*/pw.out 2>/dev/null | wc -l)
    echo "ft_ JOB DONE: $n"
    [ "$n" -eq 246 ] || { echo "ABORT: need 246 ft_ outputs, have $n"; exit 1; }

    bash results/revision/finetune/train/box_sequence.sh > /root/logs/ft_sequence.log 2>&1
    echo "RUN_DONE rc=$? $(date -u)"
    for i in $(seq 1 45); do [ -e /root/pulled ] && break; sleep 60; done
    ;;
  watchdog)
    t0=$(date +%s)
    while true; do
      sleep 300
      if ! tmux has-session -t ft 2> /dev/null; then self_stop "runner session gone"; exit 0; fi
      if [ $(( $(date +%s) - t0 )) -ge 32400 ]; then self_stop "9 h cap"; exit 0; fi
    done
    ;;
  *) echo "usage: $0 run|watchdog" >&2; exit 2 ;;
esac
