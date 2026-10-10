#!/bin/bash
# Keeps each model at its worker target across both grid copies (2026-10-03). Every 60 s, for each
# model, counts live workers of either copy and starts grid2 workers (detached, setsid, so that
# `timeout` is not a session leader) until the target is met. Stops once a grid2 worker of every
# model has exited with nothing left to do.
cd /root/mlip-dynamic-stability || exit 1
declare -A ENV=([mace_mp0]=mace [chgnet]=chgnet [sevennet0]=sevennet [mattersim]=mattersim [orb_v2]=orb)
declare -A TARGET=([mace_mp0]=3 [chgnet]=3 [sevennet0]=3 [mattersim]=3 [orb_v2]=2)
declare -A NEXT=([mace_mp0]=1 [chgnet]=1 [sevennet0]=1 [mattersim]=1 [orb_v2]=1)
while true; do
  done_models=0
  for m in "${!ENV[@]}"; do
    # anchored at the start of the command line: the `timeout` parent carries the same arguments
    # after its own, so an unanchored pattern would count every worker twice
    live=$(pgrep -fc "^/root/env-[a-z]+/bin/python -u scripts/sscha_seed_study_grid2?\.py --preset grid --model $m ")
    # a grid2 worker that found nothing to run means this model's queue is empty (or what is left
    # failed or is held by a live worker); restarting more would only spin
    if grep -qsE "\[grid\] pid [0-9]+ finished .*: 0 run here" /root/logs/grid2_${ENV[$m]}_*.log; then
      done_models=$((done_models + 1)); continue
    fi
    while [ "$live" -lt "${TARGET[$m]}" ]; do
      k=${NEXT[$m]}; NEXT[$m]=$((k + 1))
      setsid bash scripts/box/grid2_run.sh "${ENV[$m]}" "$m" "$k" > /dev/null 2>&1 < /dev/null &
      echo "$(date -u +%H:%M:%S) started grid2 $m k=$k (live was $live)"
      live=$((live + 1)); sleep 5
    done
  done
  [ "$done_models" -eq "${#ENV[@]}" ] && { echo "TOPUP_DONE $(date -u)"; exit 0; }
  sleep 60
done
