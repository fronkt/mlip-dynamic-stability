# Per-model environments

The foundation MLIPs pin conflicting torch / e3nn / pymatgen versions, so each gets its own
venv. The harness (`mlip_dynstab`) is backend-agnostic and is `pip install -e .` into every
env; only the one MLIP package differs. On the RTX 5090, install **PyTorch from the cu128
index** (per the user's Vast.ai workflow note) *before* the MLIP package.

```bash
# common, per env:
python -m venv env-<model> && source env-<model>/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r envs/base-requirements.txt
pip install -e .            # installs mlip_dynstab
```

Then one of:

```bash
# MACE-MP-0
pip install mace-torch

# CHGNet
pip install chgnet

# ORB
pip install orb-models

# SevenNet
pip install sevenn

# MatterSim
pip install mattersim
```

Finite-T SSCHA cross-check (P3b) — a dedicated env:
```bash
pip install ase phonopy
pip install cellconstructor python-sscha   # may need a Fortran/LAPACK toolchain
```

## Vast.ai

Provision one RTX 5090 box; cap worker counts on many-core hosts (workflow note). Clone the
repo, create the env for the model you're running, and launch the grid (see
`scripts/run_grid.py`). All work is per-unit checkpointed in `results/ledger.parquet`, so an
interrupted box can be replaced and the grid simply re-run — recorded units are skipped.

## Smoke test (one structure, one model)
```bash
python -m mlip_dynstab.cli --system si_diamond --model <model> --method harmonic
python -m mlip_dynstab.cli --system srtio3_cubic --model <model> --method tdep --T 300
```
