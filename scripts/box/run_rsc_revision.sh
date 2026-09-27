#!/usr/bin/env bash
# Orchestrate the RSC-revision compute on one rented box (tasks/todo.md Phase 1, C1-C6).
#
# Launch ONCE, inside tmux, after setup_rsc_box.sh has printed READY:
#   tmux new-session -d -s rsc 'bash /root/mlip-dynamic-stability/scripts/box/run_rsc_revision.sh run'
# From any shell, any time:
#   bash scripts/box/run_rsc_revision.sh status      # stage markers, QE done/total + ETA, GPU, CPU
#   bash scripts/box/run_rsc_revision.sh check       # CLI/env pre-flight only (run does it first)
#   bash scripts/box/run_rsc_revision.sh pack        # tarball of everything to bring home
#   bash scripts/box/run_rsc_revision.sh pull-list   # the commands that bring it home
#
# Why stage markers: boxes die and ssh drops. A stage that exits 0 writes
# /root/logs/stages/<stage>.done, so relaunching `run` resumes at the first unfinished stage; the
# Python studies skip finished units themselves, so a stage killed halfway loses only its current
# unit. To force a stage again, delete its marker.
#
# Core split. vast containers show every host core to nproc but enforce a cgroup quota, and
# oversubscribing that quota once made QE 12x slower. QE gets (quota - 8) cores as N jobs x R ranks;
# the GPU chain (MLIP inference plus SSCHA's OpenMP Fortran) gets 8 threads (6 on a small box).
#
# Order. (1) PBE preparation: dft_reference geom (MatterSim first: the harmonically stable models'
# reference paths use its deciding pattern), mlip-eval, qe-inputs. (2) GPU chain: C1+C5 SSCHA
# seeds (MACE env, then MatterSim env); C3b inputs and their MLIP evaluations once every C1 unit's
# seeds are final; C2 force spread; the C4 coverage check; C1b include_v4 last, because it is
# CPU-bound. (3) Beside the GPU chain, a QE driver runs qe_queue.sh passes in their own tmux
# window, one at a time: a_ (R1.1 paths), then b_ (R1.2, once the C3b inputs exist), then ax_
# (extra modes), then a retry pass. QE is the long pole; `status` shows its progress.
#
# Nothing here writes results/ledger.*. C4 (the displacement sweep) was run and committed before
# this box existed (a27e1e3, 57 units x 5 models), so on the box it is only a dry-run check that
# the rebuilt envs hash to the deposited units; RUN_C4=1 restores the real sweep, which then is
# the ledger's only writer (method harmonic_dispsweep, one model at a time).

set -uo pipefail

SELF=$(readlink -f "${BASH_SOURCE[0]}")
REPO=${REPO:-$(cd "$(dirname "$SELF")/../.." && pwd)}
[[ -f /root/rsc_env.sh ]] && source /root/rsc_env.sh
ENV_ROOT=${ENV_ROOT:-/root}   # where setup_rsc_box.sh put env-<name>; overridden only by the mock test
LOGS=${LOGS:-/root/logs}
STAGES=$LOGS/stages
QE_PREFIX=${QE_PREFIX:-/root/qe}
SSSP_DIR=${SSSP_DIR:-/root/sssp}
QE_TIMEOUT=${QE_TIMEOUT:-72h}
QE_DIR=$REPO/results/revision/dft/qe
V4_PY=${V4_PY:-$ENV_ROOT/env-mace/bin/python}
V4_TIMEOUT_S=${V4_TIMEOUT_S:-10800}
# Threaded OpenBLAS for the include_v4 child only (setup_rsc_box.sh installs it and pins the
# system BLAS back to the reference one the seeds use). V4_BLAS=reference turns it off.
OPENBLAS_V4_DIR=${OPENBLAS_V4_DIR:-/usr/lib/x86_64-linux-gnu/openblas-pthread}
V4_BLAS=${V4_BLAS:-openblas}
PULL_DIR=${PULL_DIR:-/root/pull}
QE_POLL_S=${QE_POLL_S:-120}

MODELS=(mace_mp0 chgnet orb_v2 sevennet0 mattersim)
# MatterSim first: dft_reference.py geom builds the other models' bcc-Zr / SrTiO3 reference paths
# from MatterSim's deciding pattern, which only the MatterSim env can produce.
GEOM_ORDER=(mattersim mace_mp0 chgnet orb_v2 sevennet0)
declare -A ENVNAME=([mace_mp0]=mace [chgnet]=chgnet [orb_v2]=orb [sevennet0]=sevennet [mattersim]=mattersim)
# C2 committees: force_spread.py maps the production checkpoint (MACE medium, MatterSim 5M) onto
# the production file, so only the other members are listed.
FS_COMMITTEE=(mace_mp0:small mace_mp0:large mattersim:mattersim-v1.0.0-1m)
# C4: all five models' sweeps are in the committed ledger (57/57 each; CHGNet 1d93e25, the other
# four a27e1e3), so the default is a coverage check over all five.
SWEEP_MODELS=(mace_mp0 chgnet orb_v2 sevennet0 mattersim)

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1   # mace-torch 0.3.16 under torch >= 2.6; as in August

py() { echo "$ENV_ROOT/env-${ENVNAME[$1]}/bin/python"; }

cpu_quota() {
  local q p
  if [[ -r /sys/fs/cgroup/cpu.max ]]; then
    read -r q p < /sys/fs/cgroup/cpu.max
    [[ $q != max ]] && { echo $(( q / p )); return; }
  elif [[ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]]; then
    q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us); p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
    (( q > 0 )) && { echo $(( q / p )); return; }
  fi
  env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc   # GNU nproc obeys both variables
}

QUOTA=$(cpu_quota)
if (( QUOTA >= 24 )); then GPU_THREADS=${GPU_THREADS:-8}; else GPU_THREADS=${GPU_THREADS:-6}; fi
QE_BUDGET=$(( QUOTA > 12 ? QUOTA - 8 : QUOTA / 2 ))
(( QE_BUDGET >= 1 )) || QE_BUDGET=1
if [[ -z ${QE_RANKS:-} ]]; then
  # The rank count that uses most of the budget, preferring 8 (a 40-atom perovskite SCF scales
  # well to about there) and then fewer, larger jobs.
  best=0
  for r in 8 6 10 12; do
    (( r <= QE_BUDGET )) || continue
    (( (QE_BUDGET / r) * r > best )) && { best=$(( (QE_BUDGET / r) * r )); QE_RANKS=$r; }
  done
  : "${QE_RANKS:=$QE_BUDGET}"
fi
QE_JOBS=${QE_JOBS:-$(( QE_BUDGET / QE_RANKS ))}
(( QE_JOBS >= 1 )) || QE_JOBS=1
export OMP_NUM_THREADS=$GPU_THREADS MKL_NUM_THREADS=$GPU_THREADS OPENBLAS_NUM_THREADS=$GPU_THREADS

mkdir -p "$LOGS" "$STAGES"

say() {
  local line; line="[$(date -u +%FT%TZ)] $*"
  echo "$line"; echo "$line" >> "$LOGS/orchestrator.log"
}

# stage NAME TIMEOUT CMD... : skip if marked done; else run CMD from the repo root under timeout,
# append to $LOGS/NAME.log, mark done on exit 0, print an [error] line otherwise. A caller may set
# OK_RC to the exit codes that mean "finished, with a recorded flag" (dft_reference.py geom exits
# 4 when a path fails its cache check but its structures are written); those count as done.
stage() {
  local name=$1 limit=$2; shift 2
  if [[ -f $STAGES/$name.done ]]; then say "skip  $name (done $(<"$STAGES/$name.done"))"; return 0; fi
  say "start $name [timeout $limit]"
  echo "=== $(date -u +%FT%TZ) $*" >> "$LOGS/$name.log"
  : > "$STAGES/$name.running"
  local t0=$SECONDS rc c ok=0
  ( cd "$REPO" && exec timeout --kill-after=120 "$limit" "$@" ) >> "$LOGS/$name.log" 2>&1 < /dev/null
  rc=$?
  rm -f "$STAGES/$name.running"
  for c in 0 ${OK_RC:-}; do (( rc == c )) && ok=1; done
  if (( ok )); then
    date -u +%FT%TZ > "$STAGES/$name.done"
    if (( rc == 0 )); then say "done  $name ($(( SECONDS - t0 ))s)"
    else say "done  $name ($(( SECONDS - t0 ))s) [flag] exit $rc means finished with a recorded flag -- read $LOGS/$name.log"; fi
    return 0
  else
    local why=""
    (( rc == 124 || rc == 137 )) && why=" (timeout)"
    say "[error] $name rc=$rc$why after $(( SECONDS - t0 ))s -- see $LOGS/$name.log"
    echo "[error] rc=$rc$why $(date -u +%FT%TZ)" >> "$LOGS/$name.log"
  fi
  return $rc
}

# ------------------------------------------------------------------------------ QE queue ----
qe_alive() {
  # The pid must still be a qe-run shell: after a reboot or a kill the file can outlive its
  # process, and a recycled pid would otherwise hold every later pass back forever.
  local p
  p=$(cat "$LOGS/qe_queue.pid" 2>/dev/null) && [[ -n $p ]] && kill -0 "$p" 2>/dev/null \
    && tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q -- 'qe-run'
}

qe_busy() {  # a pass is running, or the driver may still start one
  qe_alive || { [[ -n ${DRIVER_PID:-} ]] && kill -0 "$DRIVER_PID" 2>/dev/null; }
}

qe_queue_args() {
  local a=(--qe-prefix "$QE_PREFIX" --pseudo-dir "$SSSP_DIR" --jobs "$QE_JOBS" --ranks "$QE_RANKS")
  [[ -n ${QE_FILTER:-} ]] && a+=(--filter "$QE_FILTER")
  echo "${a[@]}"
}

qe_run() {
  # Runs inside its own tmux window. Environment is passed on the command line because a
  # tmux window does not inherit the orchestrator's.
  echo $$ > "$LOGS/qe_queue.pid"
  echo "${QE_TOKEN:-}" > "$LOGS/qe_queue.token"   # tells qe_launch this pass started
  export PATH="$QE_PREFIX/bin:$PATH" LD_LIBRARY_PATH="$QE_PREFIX/lib:${LD_LIBRARY_PATH:-}"
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  export ESPRESSO_PSEUDO=$SSSP_DIR     # the pw.in files leave pseudo_dir to the environment
  if [[ -x $QE_PREFIX/bin/ompi_info ]]; then
    # Several independent mpiruns share the box; left to itself OpenMPI binds each one's ranks to
    # the same first cores.
    export OMPI_MCA_hwloc_base_binding_policy=none OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
  fi
  local n; n=$(find "$QE_DIR" -mindepth 2 -maxdepth 2 -name pw.in 2>/dev/null | wc -l)
  say "qe-run start: pass '${QE_FILTER:-all}', $n inputs on disk, $QE_JOBS jobs x $QE_RANKS ranks (quota $QUOTA), timeout $QE_TIMEOUT"
  local t0=$SECONDS rc
  # shellcheck disable=SC2046
  ( cd "$REPO" && exec timeout --kill-after=300 "$QE_TIMEOUT" bash scripts/box/qe_queue.sh $(qe_queue_args) ) \
    >> "$LOGS/qe_queue.log" 2>&1 < /dev/null
  rc=$?
  if (( rc == 0 )); then say "qe-run finished ($(( SECONDS - t0 ))s)"
  else say "[error] qe-run rc=$rc after $(( SECONDS - t0 ))s -- see $LOGS/qe_queue.log"; fi
  rm -f "$LOGS/qe_queue.pid"
}

qe_launch() {  # qe_launch [FILTER]: one queue pass over the jobs whose name matches FILTER
  local filter=${1:-}
  if qe_alive; then say "QE queue already running (pid $(<"$LOGS/qe_queue.pid")); pass '${filter:-all}' not started"; return 0; fi
  local token; token="$(date +%s)-$$-$RANDOM"
  local cmd="env QE_TOKEN=$token QE_FILTER='$filter' QE_JOBS=$QE_JOBS QE_RANKS=$QE_RANKS QE_TIMEOUT=$QE_TIMEOUT QE_PREFIX=$QE_PREFIX SSSP_DIR=$SSSP_DIR REPO=$REPO LOGS=$LOGS bash $SELF qe-run"
  if [[ -n ${TMUX:-} ]]; then tmux new-window -d -n qe "$cmd"
  else tmux new-session -d -s rsc-qe "$cmd"; fi \
    || { say "[error] could not open a tmux window for the QE queue"; return 1; }
  # A pass with nothing left to do can finish before qe_alive would see it, so wait for the
  # token the pass writes when it starts rather than for a live pid.
  local _
  for _ in $(seq 1 60); do [[ $(cat "$LOGS/qe_queue.token" 2>/dev/null) == "$token" ]] && break; sleep 1; done
  if [[ $(cat "$LOGS/qe_queue.token" 2>/dev/null) == "$token" ]]; then
    say "QE queue pass launched ($QE_JOBS x $QE_RANKS ranks; log $LOGS/qe_queue.log)"
  else
    say "[error] QE queue did not start within 60 s; see $LOGS/qe_queue.log"; return 1
  fi
}

qe_passes_when_idle() {
  # Each pass waits for the previous one to exit. Exits quietly if the orchestrator that started
  # it has died, so an orphan cannot race the driver of a relaunched orchestrator.
  # Budget lever, read when each pass is due: `touch $LOGS/qe.no_extras` drops the ax_ extra modes
  # and narrows the retry pass to the a_/b_ jobs.
  local f
  for f in "$@"; do
    while qe_alive; do kill -0 "$ORCH_PID" 2>/dev/null || exit 0; sleep "$QE_POLL_S"; done
    kill -0 "$ORCH_PID" 2>/dev/null || exit 0
    if [[ -f $LOGS/qe.no_extras ]]; then
      [[ $f == '^ax_' ]] && { say "QE pass '^ax_' skipped ($LOGS/qe.no_extras)"; continue; }
      [[ -z $f ]] && f='^(a|b)_'
    fi
    qe_launch "$f"
  done
}

qe_driver() {
  # qe_queue.sh runs smallest-first whatever the prefix, so unstaged, the 40-atom SrTiO3 paths
  # that answer R1.1 would come last. Staging (the queue author's suggestion): a_ = C3a deciding
  # and reference paths (R1.1), then b_ = C3b SSCHA configurations (R1.2) once their inputs exist,
  # then ax_ = optional extra modes, then one unfiltered pass that retries anything that failed.
  qe_passes_when_idle '^a_'
  while [[ ! -f $LOGS/c3b.verdict ]]; do kill -0 "$ORCH_PID" 2>/dev/null || exit 0; sleep "$QE_POLL_S"; done
  if [[ $(<"$LOGS/c3b.verdict") == ok ]]; then qe_passes_when_idle '^b_' '^ax_' ''
  else qe_passes_when_idle '^ax_' ''; fi
}

# ----------------------------------------------------------------------------- stages ----
dft_prep() {
  local m ok=1
  # Modelled cost only; the C3a/C3b job list from the caches, logged for the budget.
  stage dft_plan 30m "$(py mace_mp0)" -u scripts/dft_reference.py plan
  for m in "${GEOM_ORDER[@]}"; do
    # Exit 4: a regenerated path missed its cached map (seen locally on BaTiO3 for MACE and
    # CHGNet). Its structures are still written and flagged in geom/checks/, and analyze
    # reports the flag, so it is a C3a finding, not a reason to hold back every PBE job. Exit 3
    # (deferred: MatterSim's pattern missing) and anything else still stop C3a.
    OK_RC=4 stage "dft_geom_$m" 3h "$(py "$m")" -u scripts/dft_reference.py geom --model "$m" --device cuda || ok=0
    if [[ $m == mattersim ]] && (( ! ok )); then
      say "[error] MatterSim geom failed: the other models' reference paths need its pattern; C3a held back"
      return 1
    fi
  done
  (( ok )) || { say "[error] dft geom incomplete; mlip-eval and qe-inputs held back (relaunch after fixing)"; return 1; }
  for m in "${MODELS[@]}"; do
    stage "dft_mlipeval_$m" 2h "$(py "$m")" -u scripts/dft_reference.py mlip-eval --model "$m" --device cuda
  done
  stage dft_qe_inputs 1h "$(py mace_mp0)" -u scripts/dft_reference.py qe-inputs --sssp-dir "$SSSP_DIR"
}

seeds_gate() {
  # seeds_gate WHAT: read the C1/C1b unit JSONs that sscha_seed_study.py writes, and decide from
  # the data (not from stage exit codes) whether the seeds WHAT needs are final.
  #   WHAT=c3b   every C3b unit is final (each preset seed ok or failed) or not started, and no
  #              saved configuration failed its extxyz unit check
  #   WHAT=<tag> that unit is final and its seed 0 is ok (the include_v4 target)
  # Exit 0 ready, 10 a unit has seeds still to run, 11 an extxyz unit check failed, 12 nothing
  # usable yet. "failed" counts as final because the study skips it without --retry-failed.
  local pyb; pyb=$(py mace_mp0)
  "$pyb" - "$REPO" "$1" <<'PY'
import importlib.util
import json
import sys
from pathlib import Path

repo, want = Path(sys.argv[1]), sys.argv[2]
spec = importlib.util.spec_from_file_location("sscha_seed_study", repo / "scripts" / "sscha_seed_study.py")
ssd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ssd)
states = []
for u in ssd.PRESETS["revision"]:
    tag = ssd.unit_tag(u["system"], u["model"], u["T"], u["supercell"])
    if (want == "c3b" and not u.get("c3b")) or (want != "c3b" and tag != want):
        continue
    p = ssd.STUDY_DIR / (tag + ".json")
    seeds = (json.loads(p.read_text(encoding="utf-8")).get("seeds") or {}) if p.exists() else {}
    st = {int(k): (v or {}).get("status") for k, v in seeds.items()}
    # The extxyz unit check is about the saved C3b configurations only; include_v4 reloads the
    # Hessian ensemble itself, so it is not held back by it.
    if want == "c3b" and any((((v or {}).get("configs") or {}).get("unit_check") or {})
                             .get("passed") is False for v in seeds.values()):
        state = "unit-check-failed"
    elif not seeds:
        state = "not-started"
    elif all(st.get(int(s)) in ("ok", "failed") for s in u["seeds"]):
        state = "final" if (want == "c3b" or st.get(0) == "ok") else "seed0-not-ok"
    else:
        state = "seeds-to-run"
    states.append(state)
    print(f"  [seeds] {tag:40s} {state:18s} {st}")
if "unit-check-failed" in states:
    sys.exit(11)
if "seeds-to-run" in states:
    sys.exit(10)
sys.exit(0 if "final" in states and (want == "c3b" or set(states) == {"final"}) else 12)
PY
}

seeds_and_c3b() {
  local m rc
  # C1 + C5. The preset runs whichever of its units this env can import: three C1 units and the
  # two MACE C5 units in env-mace, the MatterSim C1 unit and two C5 units in env-mattersim.
  stage sscha_seeds_mace 16h "$(py mace_mp0)" -u scripts/sscha_seed_study.py --preset revision --device cuda
  stage sscha_seeds_mattersim 10h "$(py mattersim)" -u scripts/sscha_seed_study.py --preset revision --device cuda
  # C3b. c3b-inputs keeps the first selection it writes for a unit (--force would orphan PBE
  # jobs), so a unit with seeds still to run (a timed-out or killed seed stage) must not reach it:
  # its later seeds could never enter C3b. A unit not started at all is safe to leave out, as
  # c3b-inputs adds it on a later launch. So c3b-inputs and the MLIP evaluations run on every
  # launch once the gate passes; both are idempotent and skip what is already current.
  seeds_gate c3b >> "$LOGS/orchestrator.log" 2>&1
  rc=$?
  case $rc in
    0)  ;;
    10) say "[error] C3b held back: a C1 unit still has seeds to run (see [seeds] lines in $LOGS/orchestrator.log). Relaunch to finish them; C3b follows on that launch"; return 1 ;;
    11) say "[error] C3b refused: an extxyz unit check failed (sscha_seed_study.py: do not use those configs); inspect before any C3b job exists"; return 1 ;;
    *)  say "[error] C3b: no C1 unit has final seeds yet (gate rc=$rc)"; return 1 ;;
  esac
  rm -f "$STAGES/dft_c3b_inputs.done"
  if ! stage dft_c3b_inputs 1h "$(py mace_mp0)" -u scripts/dft_reference.py c3b-inputs --sssp-dir "$SSSP_DIR"; then
    say "[error] C3b inputs failed; C3b PBE jobs will not run this pass"
    return 1
  fi
  for m in "${MODELS[@]}"; do
    rm -f "$STAGES/dft_mlipeval_c3b_$m.done"
    stage "dft_mlipeval_c3b_$m" 2h "$(py "$m")" -u scripts/dft_reference.py mlip-eval --model "$m" --device cuda
  done
  return 0
}

force_spread() {
  local m spec ck fc_ok=1
  for m in "${MODELS[@]}"; do
    stage "fs_fc_$m" 3h "$(py "$m")" -u scripts/force_spread.py --stage fc --model "$m" --device cuda || fc_ok=0
  done
  # The configurations are drawn from the five-model mean FCs and frozen by hash; forces on a
  # four-model set would answer a different question, so a missing FC set holds C2 back.
  if (( ! fc_ok )) || ! stage fs_configs 2h "$(py mace_mp0)" -u scripts/force_spread.py --stage configs; then
    say "[error] C2 held back: configs need all five FC sets and must precede every forces stage"
    return 1
  fi
  for m in "${MODELS[@]}"; do
    stage "fs_forces_$m" 3h "$(py "$m")" -u scripts/force_spread.py --stage forces --model "$m" --device cuda
  done
  for spec in "${FS_COMMITTEE[@]}"; do
    m=${spec%%:*}; ck=${spec#*:}
    stage "fs_forces_${m}_$ck" 3h "$(py "$m")" -u scripts/force_spread.py --stage forces --model "$m" --ckpt "$ck" --device cuda
  done
}

disp_sweep() {
  local m name have
  if [[ ${RUN_C4:-0} != 1 ]]; then
    # Coverage check only. The box envs must hash every sweep unit to a deposited row (the unit
    # hash includes model_version). A shortfall means the rebuilt env reports a different
    # version, not that work is missing, and recomputing would add a second copy of the sweep
    # under another version. It is an error to read, never something to fill in automatically.
    for m in "${SWEEP_MODELS[@]}"; do
      name=c4_coverage_$m
      stage "$name" 30m "$(py "$m")" -u scripts/run_disp_sweep.py --models "$m" --device cpu --dry-run || continue
      have=$(grep '^total already done:' "$LOGS/$name.log" | tail -n 1 | awk '{print $4}')
      if [[ -z $have || ${have%/*} != "${have#*/}" ]]; then
        rm -f "$STAGES/$name.done"
        say "[error] $name: ${have:-no count} sweep units of the committed ledger hash from this env. Version drift? Nothing recomputed (RUN_C4=1 would write new ledger rows); compare model_version in $LOGS/$name.log"
      fi
    done
    return 0
  fi
  for m in "${SWEEP_MODELS[@]}"; do
    name=disp_sweep_$m
    [[ -f $STAGES/$name.done ]] && { say "skip  $name (done $(<"$STAGES/$name.done"))"; continue; }
    stage "$name" 8h "$(py "$m")" -u scripts/run_disp_sweep.py --models "$m" --device cuda || continue
    # run_disp_sweep.py exits 0 when single units fail (it never aborts a sweep for one unit), so
    # its last summary line is the verdict; a failed unit leaves the stage unmarked for a retry.
    if ! grep '^done:' "$LOGS/$name.log" | tail -n 1 | grep -q ' 0 failed,'; then
      rm -f "$STAGES/$name.done"
      say "[error] $name: some units failed (see [FAIL] lines in $LOGS/$name.log); left unmarked so a relaunch retries them"
    fi
  done
}

v4_attempt() {
  # C1b. python-sscha's include_v4 Hessian is SCHAModules.get_v4 (Fortran + OpenMP over
  # n_mode^4; julia accelerates only the Fourier-space gradient, not this) followed by
  # get_odd_straight_with_v4, which inverts and multiplies (3N)^2 x (3N)^2 matrices with LAPACK
  # dgetrf/dgetri and three dgemm calls (get_odd_straight_with_v4.f90:118-151). For 2x2x2 BaTiO3
  # that is 14400 x 14400, ~2e13 flop: hours on the single-threaded reference BLAS the seeds were
  # built against, about the 3 h cap, and a couple of minutes on threaded OpenBLAS. So the child
  # alone gets OpenBLAS through LD_LIBRARY_PATH. Only the library changes, not the algorithm, and
  # the child's own include_v4=False round trip on the reloaded ensemble is compared with the
  # parent's reference-BLAS value (v4_false_roundtrip_ok), which checks the swap.
  # sscha_seed_study.py's own default would take the whole quota even with QE running, so the
  # thread count is always passed explicitly.
  local t=${V4_THREADS:-} tag=batio3_cubic_mace_mp0_100K_sc222 blas=reference lib=()
  # A seed still to run would be computed here, under this stage's threads and BLAS, and then
  # differ in provenance from its siblings; only a final unit with seed 0 ok goes ahead.
  if ! seeds_gate "$tag" >> "$LOGS/orchestrator.log" 2>&1; then
    say "[error] C1b not attempted: $tag is not final with seed 0 ok (see [seeds] lines in $LOGS/orchestrator.log)"
    return 1
  fi
  if [[ -z $t ]]; then
    # qe_busy, not qe_alive: between two passes nothing runs for up to QE_POLL_S, and a v4 child
    # sized to the whole quota would then share the cores with the next pass for hours.
    if qe_busy; then t=$GPU_THREADS; else t=$(( QUOTA > 4 ? QUOTA - 2 : QUOTA )); fi
  fi
  if [[ $V4_BLAS == openblas && -f $OPENBLAS_V4_DIR/libblas.so.3 && -f $OPENBLAS_V4_DIR/liblapack.so.3 ]]; then
    blas="openblas ($OPENBLAS_V4_DIR)"
    lib=(LD_LIBRARY_PATH="$OPENBLAS_V4_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}")
  fi
  say "C1b include_v4: $t threads, BLAS $blas, cap ${V4_TIMEOUT_S}s (QE $(qe_busy && echo busy || echo idle))"
  # The stage cap sits above the child's own cap, so the child's timeout (recorded in the JSON)
  # always fires first, whatever V4_TIMEOUT_S is raised to.
  stage sscha_v4 "$(( ${V4_TIMEOUT_S%.*} + 7200 ))s" env "${lib[@]}" \
    OMP_NUM_THREADS="$t" MKL_NUM_THREADS="$t" OPENBLAS_NUM_THREADS="$t" \
    "$V4_PY" -u scripts/sscha_seed_study.py --preset revision --system batio3_cubic --model mace_mp0 \
    --device cuda --include-v4 --v4-timeout "$V4_TIMEOUT_S" --v4-threads "$t"
}

# ------------------------------------------------------------------------------ check ----
need_help() {  # need_help LABEL PYTHON ARGS... -- WORD... : every WORD must appear in the help text
  local label=$1 pyb=$2; shift 2
  local args=() words=() seen=0 w help miss=()
  for w in "$@"; do
    if (( ! seen )) && [[ $w == -- ]]; then seen=1; continue; fi
    if (( seen )); then words+=("$w"); else args+=("$w"); fi
  done
  help=$(cd "$REPO" && timeout 300 "$pyb" "${args[@]}" --help 2>&1) \
    || { say "[check] $label: --help failed under $pyb"; echo "$help" | tail -n 5; return 1; }
  for w in "${words[@]}"; do grep -qF -- "$w" <<< "$help" || miss+=("$w"); done
  (( ${#miss[@]} == 0 )) || { say "[check] $label: help lacks ${miss[*]}"; return 1; }
}

check() {
  local bad=0 m f mp
  [[ -f $ENV_ROOT/setup.ready ]] || { say "[check] $ENV_ROOT/setup.ready missing: setup_rsc_box.sh has not finished READY"; bad=1; }
  for f in tmux timeout flock nvidia-smi; do command -v "$f" >/dev/null || { say "[check] $f not installed"; bad=1; }; done
  nvidia-smi -L >/dev/null 2>&1 || { say "[check] nvidia-smi sees no GPU"; bad=1; }
  for m in "${MODELS[@]}"; do [[ -x $(py "$m") ]] || { say "[check] missing $(py "$m")"; bad=1; }; done
  [[ -x $QE_PREFIX/bin/pw.x ]] || { say "[check] no pw.x under $QE_PREFIX"; bad=1; }
  [[ -d $SSSP_DIR ]] || { say "[check] no SSSP dir $SSSP_DIR"; bad=1; }
  if [[ $V4_BLAS == openblas && ! -f $OPENBLAS_V4_DIR/liblapack.so.3 ]]; then
    say "[check] note: no OpenBLAS under $OPENBLAS_V4_DIR, so C1b falls back to the reference BLAS and will probably hit its cap (setup_rsc_box.sh installs libopenblas0-pthread)"
  fi
  for f in scripts/sscha_seed_study.py scripts/force_spread.py scripts/dft_reference.py \
           scripts/run_disp_sweep.py scripts/box/qe_queue.sh; do
    [[ -f $REPO/$f ]] || { say "[check] $f is not in the checkout (committed and pushed?)"; bad=1; }
  done
  if grep -q $'\r' "$REPO"/scripts/box/*.sh 2>/dev/null; then
    say "[check] CRLF line endings under scripts/box/ (bash dies silently on them in tmux): strip the carriage returns"; bad=1
  fi
  if [[ -f $REPO/scripts/box/qe_queue.sh ]]; then
    bash -n "$REPO/scripts/box/qe_queue.sh" || { say "[check] qe_queue.sh does not parse"; bad=1; }
    for f in --qe-prefix --pseudo-dir --jobs --ranks --filter; do
      grep -q -- "$f" "$REPO/scripts/box/qe_queue.sh" || { say "[check] qe_queue.sh has no $f"; bad=1; }
    done
  fi
  mp=$(py mace_mp0)
  if [[ -x $mp ]]; then
    need_help sscha_seed_study "$mp" scripts/sscha_seed_study.py -- --preset --system --model --device --include-v4 --v4-timeout --v4-threads || bad=1
    need_help force_spread "$mp" scripts/force_spread.py -- --stage --model --ckpt --device configs forces || bad=1
    need_help dft_reference "$mp" scripts/dft_reference.py -- plan geom mlip-eval qe-inputs c3b-inputs || bad=1
    need_help "dft_reference geom" "$mp" scripts/dft_reference.py geom -- --model --device || bad=1
    need_help "dft_reference qe-inputs" "$mp" scripts/dft_reference.py qe-inputs -- --sssp-dir || bad=1
    need_help "dft_reference c3b-inputs" "$mp" scripts/dft_reference.py c3b-inputs -- --sssp-dir || bad=1
    need_help run_disp_sweep "$mp" scripts/run_disp_sweep.py -- --models --device --dry-run || bad=1
    # the preset must see its units from each env it runs in
    ( cd "$REPO" && "$mp" scripts/sscha_seed_study.py --preset revision --list ) >> "$LOGS/check.log" 2>&1 \
      || { say "[check] sscha_seed_study --preset revision --list failed (see $LOGS/check.log)"; bad=1; }
  fi
  if (( bad == 0 )); then say "check: OK (quota $QUOTA -> QE $QE_JOBS x $QE_RANKS ranks, GPU chain $GPU_THREADS threads)"
  else say "check: FAILED"; fi
  return $bad
}

# ----------------------------------------------------------------------------- status ----
qe_progress() {
  local pyb; pyb=$(py mace_mp0); [[ -x $pyb ]] || pyb=python3
  "$pyb" - "$QE_DIR" "$QE_JOBS" <<'PY'
import glob, os, re, sys
qe_dir, jobs = sys.argv[1], int(sys.argv[2])
unit = {"d": 86400, "h": 3600, "m": 60, "s": 1}
# qe_queue.sh semantics: done = pw.out with JOB DONE (it also prints that when the SCF did not
# converge); a failed run's output is moved to pw.out.failed and the job is retried next pass.
ins = sorted(glob.glob(os.path.join(qe_dir, "*", "pw.in")))
done, nconv, failed, walls, by_prefix = 0, 0, 0, [], {}
for p in ins:
    d = os.path.dirname(p)
    pre = os.path.basename(d).split("_", 1)[0]
    rec = by_prefix.setdefault(pre, [0, 0, 0.0])
    rec[1] += 1
    out = os.path.join(d, "pw.out")
    txt = open(out, errors="replace").read() if os.path.exists(out) else ""
    if "JOB DONE" in txt:
        done += 1
        rec[0] += 1
        nconv += "convergence NOT achieved" in txt
        for line in txt.splitlines():
            if line.strip().startswith("PWSCF") and "WALL" in line:
                seg = line.split("CPU")[-1].split("WALL")[0]
                w = sum(float(v) * unit[u] for v, u in re.findall("([0-9.]+)([dhms])", seg))
                walls.append(w)
                rec[2] += w
    elif os.path.exists(out + ".failed"):
        failed += 1
pending = len(ins) - done
parts = "  ".join(f"{k}_ {v[0]}/{v[1]} ({v[2] / 3600:.1f} job-h)" for k, v in sorted(by_prefix.items()))
print(f"  QE {done}/{len(ins)} done ({nconv} SCF not converged), {pending} to run "
      f"({failed} failed once)   [{parts}]")
if walls:
    mean = sum(walls) / len(walls)
    eta_h = pending * mean / max(jobs, 1) / 3600
    print(f"  mean wall {mean:.0f} s/job over {len(walls)} jobs -> naive ETA ~{eta_h:.1f} h at {jobs} "
          f"concurrent (jobs run smallest first, so this UNDERestimates; "
          f"dft_reference.py analyze gives the measured c0 and remaining core-h)")
PY
}

status() {
  local f
  echo "== stages ($STAGES)"
  for f in "$STAGES"/*.done; do [[ -e $f ]] && printf '  done     %-44s %s\n' "$(basename "$f" .done)" "$(<"$f")"; done
  for f in "$STAGES"/*.running; do [[ -e $f ]] && printf '  RUNNING  %s\n' "$(basename "$f" .running)"; done
  local op; op=$(cat "$LOGS/orchestrator.pid" 2>/dev/null)
  if [[ -n $op ]] && kill -0 "$op" 2>/dev/null; then echo "  orchestrator pid $op alive"; else echo "  orchestrator not running"; fi
  echo "== errors since the last orchestrator start"
  awk 'index($0, "orchestrator start") { n = 0 } index($0, "[error]") { e[++n] = $0 }
       END { for (i = (n > 10 ? n - 9 : 1); i <= n; i++) print "  " e[i] }' "$LOGS/orchestrator.log" 2>/dev/null
  echo "== QE queue ($(qe_alive && echo "running, pid $(<"$LOGS/qe_queue.pid")" || echo idle); pw.x processes $(pgrep -c -x pw.x))"
  qe_progress
  echo "== GPU"
  nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader | sed 's/^/  /'
  echo "== CPU  quota $QUOTA (QE $QE_JOBS x $QE_RANKS, GPU chain $GPU_THREADS threads)  load $(cut -d' ' -f1-3 /proc/loadavg)"
  echo "== disk  $(df -h "$REPO" | tail -n 1)"
}

# ------------------------------------------------------------------------------- pull ----
ledger_changed() { ! md5sum --status -c "$LOGS/ledger.start.md5" 2>/dev/null; }

pack() {
  # One tarball, so the pull is one scp (vast's ssh gateway throttles many small connections) and
  # the local side needs no rsync. QE scratch (wavefunctions, charge density) is left behind.
  local stamp out extra=() lb
  stamp=$(date -u +%Y%m%dT%H%M%SZ); out=$PULL_DIR/rsc-pull-$stamp.tar.gz; lb=$(basename "$LOGS")
  mkdir -p "$PULL_DIR"
  ledger_changed && extra=(results/ledger.parquet results/ledger.jsonl)
  # pw.in sets outdir './scratch' inside each job directory
  tar -czf "$out" --exclude='scratch' --exclude='*.save' --exclude='*.wfc*' --exclude='*.mix*' \
      --exclude='*.igk*' --exclude='*.hub*' --exclude='*.bfgs' \
      --transform "s,^$lb,results/revision/box_logs," \
      --transform 's,^results/ledger[.],results/ledger.box.,' \
      -C "$REPO" results/revision "${extra[@]}" -C "$(dirname "$LOGS")" "$lb" \
    || { say "[error] pack failed"; return 1; }
  ( cd "$PULL_DIR" && md5sum "$(basename "$out")" > "$(basename "$out").md5" )
  say "packed $out ($(du -h "$out" | cut -f1)); md5 $(cut -d' ' -f1 "$out.md5")$( ((${#extra[@]})) && echo '; includes results/ledger.box.* (C4 rows)')"
}

pull_list() {
  local host=${PULL_HOST:-root@HOST} port=${PULL_PORT:-PORT}
  echo "== pull list ($1) -- from the local repo root (git-bash), one ssh at a time"
  echo "  ssh -p $port $host 'bash $REPO/scripts/box/run_rsc_revision.sh pack'"
  echo "  scp -P $port '$host:$PULL_DIR/rsc-pull-*.tar.gz*' .      # newest pair; check with md5sum -c"
  echo "  md5sum -c rsc-pull-<stamp>.tar.gz.md5 && tar -xzf rsc-pull-<stamp>.tar.gz --force-local"
  echo "  # the tarball holds results/revision/** (logs under results/revision/box_logs/)$(ledger_changed && echo ' and results/ledger.box.{parquet,jsonl}')"
  echo "  # where rsync exists (WSL), the same content:"
  echo "  rsync -avP --partial -e 'ssh -p $port' $host:$REPO/results/revision/ results/revision/"
  echo "  rsync -avP --partial -e 'ssh -p $port' $host:$LOGS/ results/revision/box_logs/"
  if ledger_changed; then
    echo "  rsync -avP --partial -e 'ssh -p $port' $host:$REPO/results/ledger.parquet results/ledger.box.parquet"
    echo "  rsync -avP --partial -e 'ssh -p $port' $host:$REPO/results/ledger.jsonl results/ledger.box.jsonl"
    echo "  # NEVER copy the box ledger over the local one: merge per tasks/compute-runbook-rsc-revision.md"
  else
    echo "  # ledger unchanged on the box: nothing to pull for it"
  fi
}

# -------------------------------------------------------------------------------- run ----
run() {
  exec 9> "$LOGS/orchestrator.lock"
  flock -n 9 || { echo "another 'run' holds $LOGS/orchestrator.lock; use 'status'"; exit 1; }
  echo $$ > "$LOGS/orchestrator.pid"
  rm -f "$STAGES"/*.running      # left by a killed run; the lock proves none is live
  say "orchestrator start: $(git -C "$REPO" rev-parse --short HEAD) on $(hostname); quota $QUOTA -> QE $QE_JOBS x $QE_RANKS ranks, GPU chain $GPU_THREADS threads"
  [[ -f $LOGS/ledger.start.md5 ]] || md5sum "$REPO/results/ledger.parquet" "$REPO/results/ledger.jsonl" > "$LOGS/ledger.start.md5"
  if [[ ${SKIP_CHECK:-0} != 1 ]]; then
    check || { say "[error] pre-flight check failed; fix and relaunch (SKIP_CHECK=1 overrides)"; exit 1; }
  fi
  mkdir -p "$REPO/results/revision"
  rm -f "$LOGS/c3b.verdict"
  ORCH_PID=$$

  dft_prep || say "[error] (1) PBE preparation incomplete; continuing with the GPU chain"
  # The QE driver runs beside the GPU chain for the rest of the box's life. It does not hold the
  # orchestrator lock (fd 9), so only a live orchestrator blocks a relaunch.
  ( exec 9>&-; qe_driver ) &
  DRIVER_PID=$!
  # The verdict says whether C3b inputs exist, which is all the QE driver needs; a failed seed
  # alone does not stop them (see seeds_and_c3b).
  if seeds_and_c3b; then echo ok > "$LOGS/c3b.verdict"
  else echo failed > "$LOGS/c3b.verdict"; say "[error] C3b inputs not written this launch; continuing"; fi
  force_spread || true
  disp_sweep
  v4_attempt || true
  say "GPU chain finished. Pull the GPU results now; QE continues."
  pull_list "GPU chain done"

  wait "$DRIVER_PID"
  while qe_alive; do sleep "$QE_POLL_S"; done
  local nerr
  nerr=$(awk 'index($0, "orchestrator start") { n = 0 } index($0, "[error]") { n++ } END { print n + 0 }' "$LOGS/orchestrator.log")
  if (( nerr )); then say "ALL DONE, with $nerr [error] lines in this launch: read 'bash $SELF status' before trusting the pull"
  else say "ALL DONE"; fi
  status
  pull_list "all done"
  say "Next: pack, pull, verify the md5, then DESTROY the box (vastai destroy instance <id> -y)."
}

case ${1:-run} in
  run)        run ;;
  status)     status ;;
  check)      check ;;
  qe-run)     qe_run ;;
  pack)       pack ;;
  pull-list)  pull_list "on request" ;;
  *)          echo "usage: $0 [run|status|check|pack|pull-list]"; exit 2 ;;
esac
