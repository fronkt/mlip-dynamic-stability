#!/bin/bash
# Box 2 (vast 54068206, CPU/QE) resume after the 2026-10-04 ~19:20 UTC credit stop, which killed
# every queue (tmux gone after the auto-restart on top-up).
#
#   tmux new -d -s resume 'bash /root/resume_box2_2026-10-04.sh run; echo'
#   tmux new -d -s watchdog 'bash /root/resume_box2_2026-10-04.sh watchdog; echo'
#
# run:      the 8 unfinished KNbO3 ft_ jobs -> the 270 remaining ax_ jobs -> the E2 checks
#           (cv_/xs_/pl_ vc-relax), then STOPS this instance (disk kept, storage-only billing).
#           CsSnBr3 ft_ is excluded on purpose: those 82 finished on box 1 and live there only.
# watchdog: independent of the runner. Stops the instance if no pw.x has run for 30 minutes,
#           so a dead runner or an empty queue can never idle on credit again.
#
# The stop is a PUT with this container's own CONTAINER_API_KEY (vast sets it for every
# instance); verified on box 1 2026-10-04 ({"success": true}, http 200).

set -u
cd /root/mlip-dynamic-stability || exit 1
LOG=/root/logs/resume.log

self_stop() {
  local id key
  id=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^CONTAINER_ID=//p')
  key=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^CONTAINER_API_KEY=//p')
  echo "SELF_STOP ($1) $(date -u) instance=$id" >> "$LOG"
  curl -s -w " http=%{http_code}\n" -X PUT -H "Authorization: Bearer $key" \
    -H "Content-Type: application/json" -d '{"state":"stopped"}' \
    "https://console.vast.ai/api/v0/instances/$id/" >> "$LOG" 2>&1
}

Q() { bash scripts/box/qe_queue.sh --qe-prefix /root/qe --pseudo-dir /root/sssp "$@"; }

case "${1:-}" in
  run)
    echo "RUN start $(date -u)" >> "$LOG"
    # Two passes per stage: the queue never retries a job that failed inside its own run.
    for pass in 1 2; do Q --filter "^ft_knbo3_" -r 8 -n 8 -k 4 --log /root/logs/qe_ft.log; done
    echo "QE_FT_EXIT $(date -u)" >> /root/logs/qe_ft.log
    for pass in 1 2; do Q --filter "^ax_" -r 8 -n 15 --log /root/logs/qe_ax2.log; done
    echo "QE_AX2_EXIT $(date -u)" >> /root/logs/qe_ax2.log
    for pass in 1 2; do
      Q --root results/revision/dft_checks/qe --filter "^(cv_|xs_|pl_.*_vcrelax)" -r 8 -n 7 -k 2 \
        --log /root/logs/qe_checks.log
    done
    echo "QE_CHECKS_EXIT $(date -u)" >> /root/logs/qe_checks.log
    echo "RUN done $(date -u)" >> "$LOG"
    self_stop "queue finished"
    ;;
  watchdog)
    echo "WATCHDOG start $(date -u)" >> "$LOG"
    idle=0
    while true; do
      if pgrep -x pw.x > /dev/null; then idle=0; else idle=$(( idle + 1 )); fi
      if [ "$idle" -ge 6 ]; then self_stop "no pw.x for 30 min"; exit 0; fi
      sleep 300
    done
    ;;
  *) echo "usage: $0 run|watchdog" >&2; exit 2 ;;
esac
