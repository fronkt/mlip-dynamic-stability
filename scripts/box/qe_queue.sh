#!/usr/bin/env bash
# Resumable pw.x queue for the C3a/C3b PBE single points (inputs from scripts/dft_reference.py).
#
#   bash scripts/box/qe_queue.sh --qe-prefix /root/micromamba/envs/qe --pseudo-dir /root/sssp \
#        [--root results/revision/dft/qe] [-r RANKS] [-n JOBS] [-k POOLS|auto] \
#        [--filter REGEX] [--log FILE] [--dry-run]
#
# Runs every <root>/<job>/pw.in that has no finished run of THAT input (a pw.out with
# 'JOB DONE' next to a pw.in.ran identical to pw.in), smallest first (atoms, then k-mesh
# size), N jobs at a time with R MPI ranks each, and deletes a job's ./scratch after it
# finishes. Re-running it after the box dies picks up where it stopped.
# Suggested staging: --filter '^a_' (R1.1 paths) then '^b_' (R1.2) then '^ax_' (extra modes).
# Sizing: k-point pools scale best, so -k defaults to 'auto': per job, the largest divisor of
# R that does not exceed the job's irreducible k count (job.json, spglib). pw.x aborts when a
# pool is left without k-points, which a fixed -k does to a symmetric Q=0 or undisplaced cell.
# A numeric -k is capped the same way. -n defaults to cores / R. --root defaults to the repo's
# results/revision/dft/qe wherever the queue is started from.
#
# Why it is written the way it is:
#   * Cores come from the cgroup CPU quota (/sys/fs/cgroup/cpu.max), not nproc. A vast.ai
#     container sees every host core in nproc but is throttled to its quota, so sizing to
#     nproc oversubscribes and every job slows down together. nproc is only the fallback when
#     no quota is set.
#   * mpirun reads stdin. Backgrounded, it drains the job list out of a `while read` loop and
#     the queue silently stops early. The list is read from fd 3 and mpirun gets < /dev/null.
#   * PATH/LD_LIBRARY_PATH are exported from --qe-prefix directly: `micromamba run` did not
#     wire PATH for pw.x last time.
#   * --bind-to none: concurrent mpiruns each bind from core 0 by default and pile onto the
#     same cores.
#   * OMP_NUM_THREADS=1: pure MPI; threaded BLAS inside each rank would oversubscribe.
#   * 'JOB DONE' is printed even when the SCF did not converge; such jobs are logged as
#     scf-not-converged (analyze excludes them) and are not re-run automatically.
#   * pw.in is copied to pw.in.ran as a job starts. A pw.out whose pw.in has changed since
#     (dft_reference.py qe-inputs --force) is not 'done': its 'JOB DONE' belongs to another
#     input, so the job runs again, and analyze excludes an output whose copy differs.
#   * Concurrency is counted from the running jobs themselves (jobs -rp), not a counter that
#     assumes each `wait -n` returns exactly one job.
#   * pw.in files carry no pseudo_dir; ESPRESSO_PSEUDO is exported from --pseudo-dir.
#
# Install (once): micromamba create -y -p /root/micromamba/envs/qe -c conda-forge qe openmpi
# (QE 7.x); SSSP 1.3 PBE efficiency: the json and the UPFs in one directory.

set -u -o pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT="$SCRIPT_DIR/../../results/revision/dft/qe"
QE_PREFIX=""
PSEUDO_DIR="${ESPRESSO_PSEUDO:-}"
NJOBS=""
RANKS=8
POOLS=auto
FILTER=""
LOG=""
DRY=0

usage() { sed -n '2,17p' "$0"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --root)          ROOT="$2"; shift 2 ;;
    --qe-prefix)     QE_PREFIX="$2"; shift 2 ;;
    --pseudo-dir)    PSEUDO_DIR="$2"; shift 2 ;;
    -n|--jobs)       NJOBS="$2"; shift 2 ;;
    -r|--ranks)      RANKS="$2"; shift 2 ;;
    -k|--pools)      POOLS="$2"; shift 2 ;;
    --filter)        FILTER="$2"; shift 2 ;;
    --log)           LOG="$2"; shift 2 ;;
    --dry-run)       DRY=1; shift ;;
    -h|--help)       usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[ -d "$ROOT" ] || { echo "no job root $ROOT" >&2; exit 2; }
ROOT=$(cd "$ROOT" && pwd)
[ -n "$LOG" ] || LOG="$ROOT/queue.log"
for v in RANKS NJOBS; do
  eval "x=\${$v}"
  [ -z "$x" ] || [[ "$x" =~ ^[1-9][0-9]*$ ]] || { echo "$v must be a positive integer, got '$x'" >&2; exit 2; }
done
[ "$POOLS" = auto ] || [[ "$POOLS" =~ ^[1-9][0-9]*$ ]] || { echo "-k must be a positive integer or auto" >&2; exit 2; }

# ---- cores: cgroup v2 quota, then v1, then nproc
cgroup_cores() {
  local q p
  if [ -r /sys/fs/cgroup/cpu.max ]; then
    read -r q p < /sys/fs/cgroup/cpu.max
    if [ "$q" != "max" ]; then echo $(( q / p )); return; fi
  elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then
    q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)
    p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
    if [ "$q" -gt 0 ]; then echo $(( q / p )); return; fi
  fi
  echo ""
}
CORES=$(cgroup_cores)
if [ -n "$CORES" ]; then
  CORE_SRC="cgroup quota"
else
  CORES=$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN)
  CORE_SRC="nproc (no cgroup quota set)"
fi
[ "$CORES" -ge 1 ] 2>/dev/null || CORES=1

if [ "$RANKS" -gt "$CORES" ]; then
  echo "note: $RANKS ranks > $CORES cores; using $CORES" >&2
  RANKS=$CORES
fi
MAXJ=$(( CORES / RANKS )); [ "$MAXJ" -ge 1 ] || MAXJ=1
if [ -z "$NJOBS" ]; then
  NJOBS=$MAXJ
elif [ "$NJOBS" -gt "$MAXJ" ]; then
  echo "note: $NJOBS jobs x $RANKS ranks > $CORES cores; using $MAXJ jobs" >&2
  NJOBS=$MAXJ
fi
# Pools for one job: pw.x needs the pool count to divide the ranks, and every pool to get at
# least one k-point. nk_irr_spglib (job.json) is the lower bound on pw.x's own count; a job
# without it gets one pool, which is always legal.
job_pools() {
  local d=$1 want nk p
  nk=$(sed -n 's/.*"nk_irr_spglib": *\([0-9][0-9]*\).*/\1/p' "$d/job.json" 2>/dev/null | head -n 1)
  [ -n "$nk" ] || nk=1
  if [ "$POOLS" = auto ]; then want=$RANKS; else want=$POOLS; fi
  [ "$want" -le "$nk" ] || want=$nk
  p=$want
  while [ "$p" -gt 1 ] && [ $(( RANKS % p )) -ne 0 ]; do p=$(( p - 1 )); done
  echo "$p"
}

# ---- environment
if [ -n "$QE_PREFIX" ]; then
  export PATH="$QE_PREFIX/bin:$PATH"
  export LD_LIBRARY_PATH="$QE_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
[ -n "$PSEUDO_DIR" ] && export ESPRESSO_PSEUDO="$PSEUDO_DIR"
# Open MPI's shared-memory single-copy (CMA) needs ptrace, which docker usually denies. Kept
# identical to setup_rsc_box.sh's QE self-test, the configuration that is known to run.
export OMPI_MCA_btl_vader_single_copy_mechanism=none

MPI_FLAGS=()
if command -v mpirun >/dev/null 2>&1; then
  if mpirun --version < /dev/null 2>&1 | grep -qi "open mpi"; then
    MPI_FLAGS=(--allow-run-as-root --bind-to none)
  else
    MPI_FLAGS=(-bind-to none)          # MPICH / hydra spelling
  fi
fi

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"; }

# ---- job list: "nat<TAB>nk<TAB>dir", smallest first
JOBLIST=$(mktemp)
trap 'rm -f "$JOBLIST"' EXIT
for f in "$ROOT"/*/pw.in; do
  [ -e "$f" ] || continue
  d=${f%/pw.in}
  j=${d##*/}
  if [ -n "$FILTER" ] && ! [[ "$j" =~ $FILTER ]]; then continue; fi
  nat=$(awk -F'=' 'tolower($1) ~ /^[ \t]*nat[ \t]*$/ {gsub(/[ \t,]/, "", $2); print $2; exit}' "$f")
  nk=$(awk 'k {print $1 * $2 * $3; exit} /^K_POINTS/ {k = 1}' "$f")
  printf '%s\t%s\t%s\n' "${nat:-0}" "${nk:-0}" "$d"
done | sort -t $'\t' -k1,1n -k2,2n > "$JOBLIST"

# done = pw.x finished THIS input: 'JOB DONE' in pw.out and pw.in.ran identical to pw.in
is_done() { [ -f "$1/pw.out" ] && grep -q 'JOB DONE' "$1/pw.out" && cmp -s "$1/pw.in" "$1/pw.in.ran"; }

n_all=0; n_done=0; n_nc=0; PENDING=()
while IFS=$'\t' read -r nat nk d <&3; do
  n_all=$(( n_all + 1 ))
  if is_done "$d"; then
    n_done=$(( n_done + 1 ))
    grep -q 'convergence NOT achieved' "$d/pw.out" && n_nc=$(( n_nc + 1 ))
  else
    PENDING+=("$nat"$'\t'"$nk"$'\t'"$d")
  fi
done 3< "$JOBLIST"

# ---- pseudopotentials named by the pending inputs must exist
MISSING=()
if [ ${#PENDING[@]} -gt 0 ]; then
  while IFS= read -r upf; do
    [ -n "$upf" ] || continue
    [ -f "${PSEUDO_DIR%/}/$upf" ] || MISSING+=("$upf")
  done < <(for row in "${PENDING[@]}"; do
             d=${row##*$'\t'}
             awk '/^ATOMIC_SPECIES/ {s = 1; next} s && NF == 3 {print $3; next} s {exit}' "$d/pw.in"
           done | sort -u)
fi

echo "root:   $ROOT"
echo "cores:  $CORES ($CORE_SRC); $NJOBS concurrent jobs x $RANKS ranks, $POOLS pools"
echo "jobs:   $n_all matched, $n_done done ($n_nc scf-not-converged), ${#PENDING[@]} to run"
echo "pw.x:   $(command -v pw.x || echo MISSING)   mpirun: $(command -v mpirun || echo MISSING)"
echo "pseudo: ${PSEUDO_DIR:-UNSET}${MISSING[*]:+   MISSING: ${MISSING[*]}}"

if [ "$DRY" -eq 1 ]; then
  for row in "${PENDING[@]}"; do
    IFS=$'\t' read -r nat nk d <<< "$row"
    printf '  would run %-60s nat=%-4s kpts=%-6s pools=%s\n' "${d##*/}" "$nat" "$nk" "$(job_pools "$d")"
  done
  exit 0
fi
[ ${#PENDING[@]} -gt 0 ] || { echo "nothing to run"; exit 0; }
command -v pw.x >/dev/null 2>&1 || { echo "pw.x not found (give --qe-prefix)" >&2; exit 2; }
command -v mpirun >/dev/null 2>&1 || { echo "mpirun not found (give --qe-prefix)" >&2; exit 2; }
[ -n "$PSEUDO_DIR" ] || { echo "give --pseudo-dir (or export ESPRESSO_PSEUDO)" >&2; exit 2; }
[ ${#MISSING[@]} -eq 0 ] || { echo "missing UPF files in $PSEUDO_DIR: ${MISSING[*]}" >&2; exit 2; }

run_job() {
  local nat=$1 nk=$2 d=$3 j t0 t1 rc status npool
  j=${d##*/}
  npool=$(job_pools "$d")
  t0=$(date +%s)
  log "START $j nat=$nat kpts=$nk np=$RANKS nk_pools=$npool"
  cp -f "$d/pw.in" "$d/pw.in.ran"
  ( cd "$d" && mpirun "${MPI_FLAGS[@]}" -np "$RANKS" pw.x -nk "$npool" -in pw.in \
      > pw.out 2> pw.err < /dev/null )
  rc=$?
  t1=$(date +%s)
  if is_done "$d"; then
    if grep -q 'convergence NOT achieved' "$d/pw.out"; then
      status="scf-not-converged"
    else
      status="ok"
    fi
    rm -rf "$d/scratch"
  else
    status="failed(rc=$rc)"
    mv -f "$d/pw.out" "$d/pw.out.failed" 2>/dev/null
  fi
  log "END   $j status=$status wall=$(( t1 - t0 ))s"
}

on_signal() {
  # The whole process group: the run_job subshells and the mpiruns under them. Killing only
  # `jobs -p` leaves mpirun and its ranks orphaned, still holding the cores.
  trap '' INT TERM
  log "INTERRUPTED: stopping running jobs"
  kill -TERM 0 2>/dev/null
  wait
  exit 130
}
trap on_signal INT TERM

log "QUEUE start: ${#PENDING[@]} jobs, $NJOBS x $RANKS ranks on $CORES cores ($CORE_SRC)"
for row in "${PENDING[@]}"; do
  IFS=$'\t' read -r nat nk d <<< "$row"
  while [ "$(jobs -rp | wc -l)" -ge "$NJOBS" ]; do
    wait -n 2>/dev/null || sleep 1
  done
  run_job "$nat" "$nk" "$d" &
done
wait

n_ok=0; n_nc=0; n_fail=0
for row in "${PENDING[@]}"; do
  d=${row##*$'\t'}
  if is_done "$d"; then
    if grep -q 'convergence NOT achieved' "$d/pw.out"; then n_nc=$(( n_nc + 1 )); else n_ok=$(( n_ok + 1 )); fi
  else
    n_fail=$(( n_fail + 1 ))
  fi
done
log "QUEUE end: $n_ok ok, $n_nc scf-not-converged, $n_fail failed (re-run to retry failed)"
