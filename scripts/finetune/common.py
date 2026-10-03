"""Shared constants, I/O and provenance helpers for the fine-tuning trial.

Design reference: tasks/preregistration-finetune-2026-10-03.md (do not edit; deviations are
recorded there by the owner, never here).  Nothing in this package writes results/ledger.*.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
FT_DIR = SCRIPTS / "finetune"
DFT_ROOT = REPO / "results" / "revision" / "dft"
QE_ROOT = DFT_ROOT / "qe"
OUT_DEFAULT = REPO / "results" / "revision" / "finetune"

# ---- design (pre-registration, "Design") ---------------------------------------------------
TEST_SYSTEMS = ("batio3_cubic", "knbo3_cubic", "cssnbr3_cubic")
BASE_MODELS = ("mace_mp0", "chgnet")
N_PER_T = {100.0: 13, 300.0: 13, 600.0: 14}          # 40 configurations per (system, model)
N_CONFIGS = sum(N_PER_T.values())
SUPERCELL = (2, 2, 2)
REPLICATE_SEEDS = (0, 1, 2)
LEAK_RMS_A = 0.05                                      # leakage guard radius (Angstrom, RMS)
CONFIG_SEED = 20261003                                 # draws (with crc32(system), crc32(model), T)
SPLIT_SEED = 20261003                                  # 90/10 train/validation split
VAL_FRACTION = 0.10

# ---- screen pipeline (as in dft_reference.py / cli.run_unit) --------------------------------
LADDER_T = (100.0, 300.0, 600.0, 900.0)               # P2 ladder (pre-registration)
SCREEN_T = (50.0, 100.0, 300.0, 600.0, 900.0)         # dft_reference.T_SCREEN, for the C3a re-solve
MAX_MODES = 24
FC_DISP_ANG = 0.01
RELAX_FMAX = 1e-3
CONTROLS = ("si_diamond", "mgo_rocksalt", "nacl_rocksalt", "cu_fcc", "c_diamond", "ceo2_cubic")


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---- JSON / hashing ------------------------------------------------------------------------

def _default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return o.as_posix()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def _finite(o):
    if isinstance(o, dict):
        return {str(k): _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    if isinstance(o, np.ndarray):
        return _finite(o.tolist())
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


def jdump(path, obj, indent=1) -> None:
    """Atomic JSON write; NaN/inf become null so strict readers accept the file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(_finite(obj), fh, indent=indent, default=_default, allow_nan=False)
    os.replace(tmp, path)


def jload(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_arrays(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    return h.hexdigest()


def rel(path, root=REPO) -> str:
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def write_frames(path, frames) -> None:
    from ase.io import write
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.extxyz")
    write(tmp, frames, format="extxyz")
    os.replace(tmp, path)


def read_frames(path):
    from ase.io import read
    return read(path, index=":", format="extxyz")


def rng_for(system: str, model: str, T: float):
    """The draw stream of one (system, base model, T): fixed, documented, independent."""
    return np.random.default_rng([CONFIG_SEED, zlib.crc32(system.encode()),
                                  zlib.crc32(model.encode()), int(round(T))])


# ---- provenance ----------------------------------------------------------------------------

def _git() -> dict:
    try:
        head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=20).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain",
                                "--untracked-files=no"], capture_output=True, text=True,
                               timeout=20).stdout.strip()
        return {"git_head": head or None, "git_dirty": bool(dirty)}
    except Exception:
        return {"git_head": None, "git_dirty": None}


def versions(extra=()) -> dict:
    from importlib.metadata import PackageNotFoundError, version
    out = {}
    for p in ("numpy", "scipy", "ase", "phonopy", "spglib", "torch", "mace-torch", "chgnet",
              "pandas", *extra):
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = None
    return out


def provenance(stage: str, args=None, **extra) -> dict:
    s = {}
    if args is not None:
        s = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
             if k != "func"}
    return {"script": "scripts/finetune_trial.py", "stage": stage, **_git(),
            "host": socket.gethostname(), "utc": now_utc(), "python": platform.python_version(),
            "platform": platform.platform(), "versions": versions(), "settings": s, **extra}


# ---- the three perovskite prototypes: ideal sites and normalised displacement fields ---------
#
# A configuration is compared with another through its displacement field from the IDEAL
# cubic perovskite (Pm-3m) sites, in units of the lattice constant a.  a is read off the cell:
# a = (|det cell| / (n_atoms / 5))^(1/3).  Cartesian axes are the cubic axes in every geometry
# this study handles (cells may be permuted or sign-flipped sets of lattice vectors, the
# positions are not), so positions / a are directly comparable between cells, between the
# modulated supercells of the C3a paths and 2x2x2 training cells, and between models whose
# relaxed lattice constants differ.

def lattice_constant(atoms) -> float:
    vol = abs(float(np.linalg.det(np.asarray(atoms.cell.array, float))))
    return (vol / (len(atoms) / 5.0)) ** (1.0 / 3.0)


def ideal_sites(system: str):
    """(numbers, normalised positions in [0, 2)^3) of the ideal 2x2x2 supercell, canonical order."""
    from ase.build import make_supercell
    from mlip_dynstab.systems import build_atoms, get_spec
    prim = build_atoms(get_spec(system))
    a = lattice_constant(prim)
    sc = make_supercell(prim, np.diag(SUPERCELL))
    x = np.mod(sc.get_positions() / a, 2.0)
    x[x > 2.0 - 1e-6] -= 2.0
    return np.asarray(sc.get_atomic_numbers()), x


def field_from_sites(numbers, positions, cell, sites, tol=0.30):
    """Normalised displacement of each atom from its nearest same-species ideal site (minimum
    image in the 2a periodic box), returned in the canonical site order.  ``None`` if the
    assignment is not a bijection or an atom is farther than ``tol`` (in units of a) from every
    site (not a perturbation of the prototype)."""
    s_num, s_x = sites
    a = lattice_constant(_AtomsLike(numbers, cell))
    x = np.asarray(positions, float) / a
    out = np.full((len(s_num), 3), np.nan)
    taken = np.zeros(len(s_num), bool)
    for i, z in enumerate(numbers):
        cand = np.where(s_num == z)[0]
        d = x[i] - s_x[cand]
        d -= 2.0 * np.round(d / 2.0)
        k = int(np.argmin(np.linalg.norm(d, axis=1)))
        if np.linalg.norm(d[k]) > tol or taken[cand[k]]:
            return None
        taken[cand[k]] = True
        out[cand[k]] = d[k]
    return out


class _AtomsLike:
    """Just enough of ase.Atoms for lattice_constant()."""

    def __init__(self, numbers, cell):
        self.numbers = numbers
        self.cell = type("C", (), {"array": np.asarray(cell, float)})()

    def __len__(self):
        return len(self.numbers)


def rms_distance(f_a, f_b, a_ang: float) -> float:
    """RMS (over atoms, of the 3-D length) of the difference of two normalised fields after
    removing the mean displacement of the difference (a rigid translation), in Angstrom for
    the lattice constant ``a_ang``."""
    d = (np.asarray(f_a, float) - np.asarray(f_b, float)) * a_ang
    d = d - d.mean(axis=-2, keepdims=True)
    r = np.sqrt((d ** 2).sum(axis=-1).mean(axis=-1))
    return float(r) if np.ndim(r) == 0 else r
