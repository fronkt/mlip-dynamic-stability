#!/usr/bin/env bash
# Regenerate every derived result, ESI table and figure of the paper from the deposited data, on a
# CPU, then re-run the claim checks. One command, from anywhere:
#
#     bash scripts/reproduce.sh [--resolve] [--no-figures] [--check-only] [--no-manifest]
#
# Needs the environment of envs/analysis-requirements.txt (Python 3.12; no GPU, no MLIP package, no
# network beyond pip). Set PYTHON=/path/to/python to use another interpreter.
#
# What it runs, in order (each step stops the script if it fails):
#   0  results/MANIFEST.sha256 check, BEFORE anything is rewritten: the deposit is the one the
#      paper was built from.
#   1  derived statistics from results/ledger.parquet: stats_hardening, estimator_noise,
#      curvature_identity_check, scha_vs_exact, power_h2 (also writes results/holm_adjusted.json).
#   2  only with --resolve: re-solve the cached E(Q) maps (screen_sensitivity ~10 min, then
#      h2_by_convention ~6 min, on 6 workers of an 8-core laptop). Needs envs/analysis-requirements.txt's pymatgen, and
#      screen_sensitivity also a MACE-MP-0 environment (see below); otherwise skipped with a note.
#   3  revision analyses from the deposited raw outputs: force_spread --stage analyze,
#      dft_reference analyze (PBE vs MLIP), grid_compare (only if its summary.csv is deposited).
#   4  build_esi_tables: rewrites the generated block of paper/supplementary.md (Tables S4-S22).
#   5  figures: make_figures (Figs. 1-6) and the TOC graphic (skipped by --no-figures).
#   6  verify_claims (every headline number against the ledger) and build_esi_tables --check.
#   7  lists the tracked files under results/ and paper/ that now differ from git. Most are
#      expected to be byte-identical; the exceptions are re-analysis outputs whose provenance block
#      records the time, host and package versions of the run.
#
# --check-only runs steps 0 and 6 and rewrites nothing; this is what CI does (.github/workflows).
# --no-manifest skips step 0 (use it after you have changed something in results/ on purpose).
#
# NOT run here, because it generates new data and needs a GPU box: the harmonic phonons, the screen's
# E(Q) maps, SSCHA and the PBE calculations. See README.md "Reproducing the paper".
#
# To re-run screen_sensitivity.py's one MACE-MP-0 check, run this script with PYTHON pointing at the
# interpreter of the MACE environment (envs/lock-mace-*.txt, CPU torch is enough); the model weights
# are fetched on first use. Without MACE the step is skipped: running it with --no-mlip-check would
# change the extensivity entry that Table S14 quotes.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PYTHON="${PYTHON:-python}"
export MPLBACKEND=Agg
export PYTHONDONTWRITEBYTECODE=1

RESOLVE=0
FIGURES=1
CHECK_ONLY=0
MANIFEST=1
for arg in "$@"; do
  case "$arg" in
    --resolve) RESOLVE=1 ;;
    --no-figures) FIGURES=0 ;;
    --check-only) CHECK_ONLY=1 ;;
    --no-manifest) MANIFEST=0 ;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed -e '$d' -e 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $arg (see --help)" >&2; exit 2 ;;
  esac
done

N=0
START=$(date +%s)
trap 'echo "reproduce.sh: FAILED at step $N" >&2' ERR

step() {   # step "label" command...
  local label="$1"; shift
  N=$((N + 1))
  printf '\n[%d] %s\n    $ %s\n' "$N" "$label" "$*"
  local t0
  t0=$(date +%s)
  "$@"
  printf '    done in %ds\n' $(( $(date +%s) - t0 ))
}

have() { "$PYTHON" -c "import $1" >/dev/null 2>&1; }

need() {   # need module "why"
  have "$1" || { echo "missing Python module '$1' ($2); pip install -r envs/analysis-requirements.txt" >&2; exit 1; }
}

need numpy "all steps"; need pandas "all steps"; need pyarrow "reading the ledger"
need yaml "configs/curated_systems.yaml"; need scipy "pandas Spearman in mlip_dynstab.analysis"

# ---- 0. the deposit is what the paper was built from
if [ "$MANIFEST" -eq 1 ]; then
  step "deposit integrity: results/MANIFEST.sha256" "$PYTHON" scripts/make_manifest.py --check
fi

if [ "$CHECK_ONLY" -eq 1 ]; then
  step "claims against the ledger" "$PYTHON" scripts/verify_claims.py
  step "ESI generated tables are up to date" "$PYTHON" scripts/build_esi_tables.py --check
  printf '\nreproduce.sh --check-only: all checks passed (%ds)\n' $(( $(date +%s) - START ))
  exit 0
fi

# ---- 1. derived statistics (seconds each)
step "headline counts, intervals, clustered tests -> results/stats_hardening.json" "$PYTHON" scripts/stats_hardening.py
step "estimator noise -> results/estimator_noise.json" "$PYTHON" scripts/estimator_noise.py
step "screen curvature identity -> results/curvature_identity_check.json" "$PYTHON" scripts/curvature_identity_check.py
step "SCHA screen vs exact 1D solution -> results/scha_vs_exact.json" "$PYTHON" scripts/scha_vs_exact.py
step "H2 power projection and Holm families -> results/power_h2.json, results/holm_adjusted.json" "$PYTHON" scripts/power_h2.py

# ---- 2. optional: re-solve the cached E(Q) maps (slow)
if [ "$RESOLVE" -eq 1 ]; then
  need pymatgen "scripts/screen_sensitivity.py and h2_by_convention.py build the reference cells with it"
  if have mace; then
    step "screen sensitivity (Table S14) -> results/screen_sensitivity.json (~10 min)" "$PYTHON" scripts/screen_sensitivity.py
  else
    echo "note: MACE-MP-0 is not importable in this environment, so screen_sensitivity.py is skipped (see --help)." >&2
  fi
  step "H2 under each frozen-cell convention -> results/h2_by_convention.json (~6 min)" "$PYTHON" scripts/h2_by_convention.py
fi

# ---- 3. revision analyses from the deposited raw outputs
step "force-level ensemble spread (Table S18) -> results/revision/force_spread/summary.json" \
  "$PYTHON" scripts/force_spread.py --stage analyze
need ase "dft_reference.py analyze reads the structure files with ase.io"
step "PBE vs MLIP (Tables S19-S20) -> results/revision/dft/*.csv, summary.json" "$PYTHON" scripts/dft_reference.py analyze
if [ -f results/revision/sscha_converged_grid/summary.csv ]; then
  step "converged-recipe SSCHA grid beside the production claims (Table S22)" "$PYTHON" scripts/grid_compare.py
else
  echo "note: results/revision/sscha_converged_grid/summary.csv is not deposited; Table S22 stays as is." >&2
fi

# ---- 4. ESI tables
step "ESI generated tables -> paper/supplementary.md" "$PYTHON" scripts/build_esi_tables.py

# ---- 5. figures
if [ "$FIGURES" -eq 1 ]; then
  need matplotlib "figures"; need PIL "figures (pillow)"
  step "Figs. 1-6 -> results/figures/" "$PYTHON" scripts/make_figures.py
  step "TOC graphic -> paper/toc_entry/" "$PYTHON" paper/toc_entry/make_toc_graphic.py
fi

# ---- 6. verification
step "claims against the ledger" "$PYTHON" scripts/verify_claims.py
step "ESI generated tables are up to date" "$PYTHON" scripts/build_esi_tables.py --check

# ---- 7. what moved
printf '\nreproduce.sh: all steps passed in %ds.\n' $(( $(date +%s) - START ))
if git rev-parse --git-dir >/dev/null 2>&1; then
  echo "Tracked files under results/ and paper/ that now differ from git (expected: provenance blocks only):"
  git status --short --untracked-files=no -- results paper | sed 's/^/  /' || true
  echo "  (end of list; inspect any entry with: git diff -- <path>)"
fi
