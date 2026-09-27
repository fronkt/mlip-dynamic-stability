"""Force-level ensemble uncertainty as a trust metric (RSC Advances R2.2; plan item C2).

Referee 2: "there is no mention of what the ensemble uncertainty of the force predictions
are. This metric can be used as a reliable measure of where the MLIP predictions should not
be trusted."

Section 3.4 already tests two disagreement signals, and both are built from the models'
finite-T CALLS: the discrete split of the five stability votes (AUC 0.76; 0.628 with a
cluster-bootstrap CI spanning 0.5 once ORB-v2 is removed) and the spread of the screen's
frequency (AUC 0.36). Neither is the quantity the referee named. This script measures that
quantity: how far apart are the models' FORCES on the same thermally displaced atoms, and
does that spread mark the (system, T) units whose finite-T call is wrong?

PRE-REGISTRATION -- written 2026-09-26, before any configuration was generated or any force
evaluated; amended 2026-09-27 in review, still before any real configuration existed ((g)
made species-pair-resolved, the missing-data rule, the softness confound; the primary is
unchanged). Do not edit this block once results exist. A later change of design goes in a
separately labelled post-hoc section, and the primary below is still the one reported.

  Units. Exactly the units of analysis.h3_guardrail_summary(df, method="softmode") on
  analysis.load_canonical(): the consensus (system, T) units of h3_ensemble_guardrail with
  bcc and borderline systems removed -- 15 systems x 4 ladder temperatures (100, 300, 600,
  900 K) = 60 units. Label: consensus call wrong (stable-vote fraction >= 0.5 is called
  stable, compared with gt_stable). Without ORB-v2 the consensus is rebuilt from the four
  remaining votes by the same rule (a 2-2 tie is called stable). The per-(system, model, T)
  softmode units those consensus units are built from (300; 240 without ORB-v2) carry the
  per-model label pred_stable != gt_stable.

  Geometry. build_atoms(get_spec(system)), the unrelaxed registry cell with every atom on a
  high-symmetry site, in the 2x2x2 supercell the harmonic layer uses (cli.run_unit default;
  the 6x6x6 bcc override is softmode-only and bcc is outside the unit set). One geometry for
  every model, so the spread measures disagreement about forces and nothing else. At that
  geometry an equivariant model's force is zero by site symmetry, so the spread is entirely
  the models' disagreement about the response to displacement.

  Configurations. Force constants from each of the five production models at that geometry
  (phonopy, cli.PRODUCTION_DISP_ANG = 0.01 A, symmetrised as in harmonic.compute_harmonic;
  the residual force of the undisplaced cell is subtracted, which is a no-op for an
  equivariant model and removes ORB-v2's symmetry-breaking offset), averaged over the five
  models. Supercell dynamical matrix at Gamma; the three translations removed exactly by
  diagonalising in the orthogonal complement of the mass-weighted translation vectors;
  imaginary omega replaced by |omega| with a floor of 0.5 THz. N = 16 draws per (system, T)
  from the quantum harmonic distribution: mode variance
  <q_k^2> = hbar / (2 omega_k) coth(hbar omega_k / 2 k_B T) in mass-weighted coordinates,
  mapped back to Cartesian by u = M^(-1/2) E q. Units: amu, THz, Angstrom, eV. RNG seed
  (0, crc32(system), T). Frozen once written; every force file records the hash of the
  configurations it was evaluated on, and the analysis refuses a mismatch.

  Models. The five production models (mace_mp0, chgnet, orb_v2, sevennet0, mattersim) and
  two within-architecture committees: MACE-MP-0 small / medium / large (medium is the
  production model) and MatterSim v1.0.0 1M / 5M (5M is production). ORB-v2 is float32 and
  non-conservative; its dtype is recorded and every result is also given without it.

  PRIMARY METRIC. For each of the 60 consensus units, the force spread S is the median over
  the 16 configurations of sqrt(mean over atoms i and Cartesian components a of
  sigma_ia^2), where sigma_ia is the population standard deviation (ddof = 0) across the
  five production models of the force F_ia in eV/A. The primary statistic is the rank AUC
  of S as a predictor of 'consensus call wrong' (stats.auc, ties credited 0.5), with a
  two-sided system-clustered permutation p (stats.cluster_permutation_auc, 10000
  permutations, seed 0) and a system-clustered bootstrap 95% CI
  (stats.cluster_bootstrap_auc, 10000 resamples, seed 0).

  Secondary, each reported whatever the primary shows and none promoted after the fact:
  (a) S over the four models without ORB-v2, against the four-model consensus label;
  (b) S over the MACE-MP-0 committee and (c) over the MatterSim committee, against the
  five-model label; (d) the normalised spread, S computed per configuration divided by the
  RMS of the across-model mean force before the median, for five and for four models;
  (e) per-(system, model, T): the median over configurations of the RMS deviation of that
  model's forces from the mean of the OTHER models (leave-one-out), as a predictor of that
  model's call being wrong -- pooled over models and per model, with and without ORB-v2;
  (f) the MACE committee spread for MACE-MP-0's own call and the MatterSim committee spread
  for MatterSim's own call; (g) the primary restricted to units in which none of the 16
  configurations brings any pair of atoms closer than 0.75 of the shortest distance between
  the same two species in the reference cell (a harmonic draw on a floored soft spectrum can
  overlap atoms, and a spread measured there is extrapolation by every model, not
  disagreement about thermal forces; the ratio is resolved by species pair because the
  overlaps are A-site cations driven into anions, which a ratio against the shortest, B-X,
  bond misses). All clustered by system exactly as the primary; where (g) leaves unequal
  blocks the clustered permutation is undefined and only the clustered bootstrap CI is
  reported. The naive unit-level permutation p is printed beside each only as a labelled
  companion; it is not evidence, because the units share systems.

  Missing data. A configuration on which any model in a score's set raises or returns a
  non-finite force is dropped from that score's median for every model in the set (never
  imputed; the count is reported per unit), and a unit left with fewer than half its
  configurations is unscored, which makes the score incomplete rather than re-defined.

  Decision rule. The force spread is reported as flagging untrustworthy finite-T calls only
  if the clustered-bootstrap 95% CI of the primary AUC lies entirely above 0.5 AND the
  four-model (a) AUC point estimate is also above 0.5. A CI entirely below 0.5 is reported
  as anti-informative (the spread is larger where the consensus is right), not re-signed.
  Anything else is reported as "not shown".

  Known confound, stated in advance. Thermal amplitude grows with T, so S grows with T, and
  consensus errors are not uniform over the ladder. Amplitude also grows with the softness
  of the mean spectrum, and the soft systems are the displacive ones where the calls are
  hard, so S partly measures softness. The per-temperature AUCs (descriptive only; 15 units
  each) and the normalised variant (d) are the checks. Neither replaces the primary. A
  second, found in the CPU smoke test before any real configuration existed: the soft
  halide spectra put many modes on the 0.5 THz floor, and the harmonic draws then bring
  the A-site cation into the anions (MACE+CHGNet mean FCs, all 15 systems, 16 draws: the
  pair-resolved contact ratio is 0.74 / 0.56 / 0.47 / 0.15 for CsSnBr3 at 100 / 300 / 600 /
  900 K, the last a Cs-Br pair at 0.63 A; CsPbI3 is at 0.69 already at 100 K; PbTiO3 has
  a Pb-O pair at 1.10 A at 600 K). The halide consensus errors sit at exactly those
  temperatures, so (g) is the check that a positive primary is not this artifact. Raising
  the floor is not a remedy: at 1.0 or 1.5 THz it moves 75-81 of the 117 halide modes,
  most of the real spectrum rather than the unstable part, and the 900 K ratios stay at
  0.46-0.75.

END PRE-REGISTRATION

Stages (resumable: each skips work whose output already exists). Run on the box:

  # in each production model's env, once per model (5x):
  python scripts/force_spread.py --stage fc --model mace_mp0 --device cuda
  # any env, CPU only; needs all five FC sets (refuses a partial set unless told):
  python scripts/force_spread.py --stage configs --n-configs 16
  # in each model env; the committee members take --ckpt:
  python scripts/force_spread.py --stage forces --model mace_mp0 --device cuda
  python scripts/force_spread.py --stage forces --model mace_mp0 --ckpt small --device cuda
  python scripts/force_spread.py --stage forces --model mace_mp0 --ckpt large --device cuda
  python scripts/force_spread.py --stage forces --model mattersim --ckpt mattersim-v1.0.0-1m --device cuda
  # local, no GPU:
  python scripts/force_spread.py --stage analyze

``--smoke`` shrinks everything to 2 systems x 2 temperatures x 2 configurations and writes
to results/revision/force_spread/smoke/ unless --out says otherwise. ``--stage selftest``
runs only the sampler's unit checks.

Outputs under results/revision/force_spread/: fc/, configs/, forces/forces_<model>[_<ckpt>].npz,
summary.json, units_consensus.csv, units_per_model.csv. Never touches results/ledger.*.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))          # match the convention in the sibling scripts

import numpy as np                     # noqa: E402

SCRIPT = "scripts/force_spread.py"
OUT_DEFAULT = REPO / "results" / "revision" / "force_spread"

PRODUCTION = ("mace_mp0", "chgnet", "orb_v2", "sevennet0", "mattersim")
EX_ORB = tuple(m for m in PRODUCTION if m != "orb_v2")
# The checkpoint each production model was run with (calculators.py defaults). Passing one of
# these as --ckpt is the production model, so it maps to the production file, not a new one.
PRODUCTION_CKPT = {"mace_mp0": "medium", "mattersim": "mattersim-v1.0.0-5m",
                   "sevennet0": "7net-0"}
MACE_COMMITTEE = ("mace_mp0_small", "mace_mp0", "mace_mp0_large")
MATTERSIM_COMMITTEE = ("mattersim_mattersim-v1.0.0-1m", "mattersim")
# Every --ckpt the analysis can use. Anything else would load (or crash in the calculator
# constructor, for the models that take no checkpoint) and then write a file whose tag no
# committee names, so the box time is spent and the result silently ignored.
ALLOWED_CKPT = {"mace_mp0": ("small", "medium", "large"),
                "mattersim": ("mattersim-v1.0.0-1m", "mattersim-v1.0.0-5m"),
                "sevennet0": ("7net-0",)}
# ORB-v2 predicts forces with a direct head; the others differentiate an energy.
CONSERVATIVE = {"mace_mp0": True, "chgnet": True, "orb_v2": False, "sevennet0": True,
                "mattersim": True}
MODEL_PKG = {"mace_mp0": "mace-torch", "chgnet": "chgnet", "orb_v2": "orb-models",
             "sevennet0": "sevenn", "mattersim": "mattersim"}

SUPERCELL = (2, 2, 2)
N_CONFIGS = 16
FREQ_FLOOR_THZ = 0.5
MIN_DIST_RATIO = 0.75     # variant (g): pair distance / reference distance of the same species pair
N_PERM = 10000
SEED = 0
SMOKE_SYSTEMS = ("cssnbr3_cubic", "mgo_rocksalt")   # one with split calls, one control
SMOKE_TEMPS = (100.0, 300.0)
SMOKE_N_CONFIGS = 2

# CODATA 2018. Two unit paths are kept deliberately separate (J*s for the variance, eV*s for
# the Bose factor) because mixing them is the trap recorded in tasks/lessons.md (2026-06-21).
HBAR_JS = 1.054571817e-34
HBAR_EVS = 6.582119569e-16
KB_EVK = 8.617333262e-5
AMU_KG = 1.66053906660e-27
EV_J = 1.602176634e-19
ANG_M = 1e-10
# omega (rad/s) of a dynamical-matrix eigenvalue of 1 eV / (A^2 amu); /2pi/1e12 = 15.6333 THz,
# which is phonopy's VaspToTHz.
OMEGA_PER_SQRT_EIG = np.sqrt(EV_J / (ANG_M ** 2 * AMU_KG))
THZ_PER_SQRT_EIG = OMEGA_PER_SQRT_EIG / (2 * np.pi) / 1e12


# ------------------------------------------------------------- provenance ----

def _git() -> dict:
    try:
        head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=20).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                               capture_output=True, text=True, timeout=20).stdout.strip()
        return {"git_head": head or None, "git_dirty": bool(dirty)}
    except Exception:
        return {"git_head": None, "git_dirty": None}


def _ver(pkg: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(pkg)
    except Exception:
        return None


def _provenance(stage: str, args, **extra) -> dict:
    pkgs = ["numpy", "pandas", "ase", "phonopy", "torch"]
    if getattr(args, "model", None) in MODEL_PKG:
        pkgs.append(MODEL_PKG[args.model])
    out = {"script": SCRIPT, "stage": stage, **_git(), "host": socket.gethostname(),
           "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "python": platform.python_version(),
           "packages": {p: _ver(p) for p in pkgs},
           "settings": {k: (str(v) if isinstance(v, Path) else v)
                        for k, v in sorted(vars(args).items())}}
    if stage in ("fc", "forces") and getattr(args, "device", "").startswith("cuda"):
        try:
            import torch
            out["gpu"] = torch.cuda.get_device_name(0)
        except Exception:
            out["gpu"] = None
    out.update(extra)
    return out


def _save_npz(path: Path, **arrays) -> None:
    """Write atomically: a box that dies mid-write must not leave a truncated file that the
    resume logic would then trust."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(tmp, path)


def _meta(npz) -> dict:
    return json.loads(str(npz["meta"]))


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(_finite(obj), indent=2, default=_jsonable, allow_nan=False),
                   encoding="utf-8")
    os.replace(tmp, path)


def _finite(o):
    """NaN/inf -> null. json.dumps writes a bare NaN, which strict JSON readers reject, and
    it never passes floats through ``default``."""
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


# --------------------------------------------------------------- unit sets ----

def unit_sets(df):
    """The consensus and per-model units, built exactly as h3_guardrail_summary builds them.

    Rebuilt here rather than read from its return value because that function returns only
    the aggregate numbers; the unit count is cross-checked against it so the two cannot drift.
    """
    from mlip_dynstab import analysis as A

    def consensus(d):
        g = A.h3_ensemble_guardrail(d, method="softmode")
        g = g[~g["system"].str.contains("bcc")]
        g = g[~g["system"].isin(A.borderline_systems())]
        g = g.sort_values(["system", "T"]).reset_index(drop=True)
        g["consensus_wrong"] = ~g["consensus_correct"].astype(bool)
        return g

    c5 = consensus(df)
    ref = A.h3_guardrail_summary(df, method="softmode")
    if ref.get("n_units") != len(c5):
        raise RuntimeError(f"unit set drifted from h3_guardrail_summary: {len(c5)} vs {ref}")
    c4 = consensus(df[df["model"] != "orb_v2"])
    sm = df[(df["method"] == "softmode") & df["system"].isin(set(c5["system"]))]
    pm = sm[["system", "model", "temperature_K", "pred_stable", "gt_stable",
             "model_version"]].rename(columns={"temperature_K": "T"}).copy()
    pm["wrong"] = pm["pred_stable"].astype(bool) != pm["gt_stable"].astype(bool)
    pm = pm.sort_values(["system", "model", "T"]).reset_index(drop=True)
    return c5, c4, pm, ref


def _scope(args):
    from mlip_dynstab import analysis as A
    df = A.load_canonical()
    c5, c4, pm, ref = unit_sets(df)
    systems = sorted(c5["system"].unique())
    temps = sorted(float(t) for t in c5["T"].unique())
    n_configs = args.n_configs
    if args.smoke:
        missing = [s for s in SMOKE_SYSTEMS if s not in systems]
        if missing:
            raise SystemExit(f"smoke systems not in the unit set: {missing}")
        systems, temps, n_configs = list(SMOKE_SYSTEMS), list(SMOKE_TEMPS), SMOKE_N_CONFIGS
    return df, c5, c4, pm, ref, systems, temps, n_configs


def _tag(model: str, ckpt: str | None) -> tuple[str, str | None]:
    if ckpt is not None and PRODUCTION_CKPT.get(model) == ckpt:
        ckpt = None
    return (model if ckpt is None else f"{model}_{ckpt}"), ckpt


def _torch_dtype(calc) -> str:
    """Parameter dtype of the torch module behind an ASE calculator, best effort.

    The calculators nest their module differently (MACE .models, MatterSim .potential.model,
    the rest .model), so this walks those names two levels deep instead of special-casing."""
    dt = getattr(calc, "dtype", None)
    try:
        import torch
    except ImportError:
        return str(dt) if dt is not None else "unknown"
    seen: set[int] = set()

    def walk(obj, depth):
        if obj is None or depth > 3 or id(obj) in seen:
            return None
        seen.add(id(obj))
        if isinstance(obj, torch.nn.Module):
            for p in obj.parameters():
                return str(p.dtype)
            return None
        if isinstance(obj, (list, tuple)):
            for o in obj:
                r = walk(o, depth + 1)
                if r:
                    return r
            return None
        for name in ("models", "model", "potential", "_model"):
            r = walk(getattr(obj, name, None), depth + 1)
            if r:
                return r
        return None

    return walk(calc, 0) or (str(dt) if dt is not None else "unknown")


# ---------------------------------------------------------------- sampler ----

def normal_modes(fc: np.ndarray, masses: np.ndarray, drop_translations: bool = True):
    """Eigen-decomposition of the supercell dynamical matrix at Gamma.

    fc is phonopy's full (N, N, 3, 3) array in eV/A^2, masses in amu. Returns eigenvalues in
    eV/(A^2 amu) and mass-weighted eigenvectors as columns.

    Translations are removed by construction rather than by picking the three smallest
    |omega|: an averaged FC set only satisfies the acoustic sum rule approximately, and on a
    spectrum with soft optical modes the smallest-|omega| rule can discard a real mode (the
    sort-order trap fixed in sscha v2). Diagonalising in the orthogonal complement of the
    mass-weighted translation vectors drops exactly those three directions and nothing else.
    """
    n = len(masses)
    d = fc.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)
    s = 1.0 / np.sqrt(np.repeat(masses, 3))
    d = d * s[:, None] * s[None, :]
    d = 0.5 * (d + d.T)
    if not drop_translations:
        return np.linalg.eigh(d)
    t = np.zeros((3 * n, 3))
    for a in range(3):
        t[a::3, a] = np.sqrt(masses)
    q, _ = np.linalg.qr(t, mode="complete")
    b = q[:, 3:]
    lam, w = np.linalg.eigh(b.T @ d @ b)
    return lam, b @ w


def signed_thz(lam: np.ndarray) -> np.ndarray:
    return np.sign(lam) * np.sqrt(np.abs(lam)) * THZ_PER_SQRT_EIG


def mode_variance(nu_thz: np.ndarray, temperature_K: float) -> np.ndarray:
    """<q^2> of each mass-weighted normal coordinate, in amu A^2 (nu must be > 0)."""
    omega = 2 * np.pi * np.asarray(nu_thz, float) * 1e12
    zero_point = HBAR_JS / (2 * omega) / (AMU_KG * ANG_M ** 2)
    if temperature_K <= 0:
        return zero_point
    x = HBAR_EVS * omega / (2 * KB_EVK * temperature_K)
    return zero_point / np.tanh(x)


def sample_displacements(evecs: np.ndarray, nu_used_thz: np.ndarray, masses: np.ndarray,
                         temperature_K: float, n: int, rng) -> np.ndarray:
    """n Cartesian displacement sets (n, N, 3) in A from the quantum harmonic distribution."""
    sd = np.sqrt(mode_variance(nu_used_thz, temperature_K))
    q = rng.standard_normal((n, len(sd))) * sd
    u_mw = q @ evecs.T
    return (u_mw / np.sqrt(np.repeat(masses, 3))).reshape(n, len(masses), 3)


class Contacts:
    """Closest approach in a displaced supercell, relative to the reference cell.

    Two ratios, because they answer different questions. The species-agnostic one divides the
    closest pair of any kind by the reference cell's shortest bond; in a perovskite that bond
    is B-X, so it misses the overlaps a floored spectrum actually produces, which are the
    A-site rattler driven into the anions (CsSnBr3, MACE+CHGNet FCs, 300 K: a Cs-Br pair at
    2.31 A, 0.56 of the 4.10 A reference Cs-Br distance, reads 0.80 on the agnostic ratio).
    The pair-resolved one divides each pair by the shortest reference distance between the
    same two species, so a Cs-Br contact is judged against Cs-Br. Variant (g) uses it.
    """

    def __init__(self, numbers, cell, ref_positions):
        import ase
        self._ase, self.numbers, self.cell = ase, np.asarray(numbers), np.asarray(cell)
        n = len(self.numbers)
        self.iu = np.triu_indices(n, 1)
        zi, zj = self.numbers[self.iu[0]], self.numbers[self.iu[1]]
        lo, hi = np.minimum(zi, zj), np.maximum(zi, zj)
        self.key = lo * 1000 + hi
        d = self._dist(ref_positions)
        self.d0 = float(d.min())
        ref = {k: float(d[self.key == k].min()) for k in np.unique(self.key)}
        self.ref_pair = np.array([ref[k] for k in self.key])
        from ase.data import chemical_symbols as cs
        self.ref_by_pair = {f"{cs[k // 1000]}-{cs[k % 1000]}": v for k, v in ref.items()}
        self._name = {k: f"{cs[k // 1000]}-{cs[k % 1000]}" for k in ref}

    def _dist(self, pos):
        a = self._ase.Atoms(self.numbers, positions=pos, cell=self.cell, pbc=True)
        return a.get_all_distances(mic=True)[self.iu]

    def __call__(self, pos):
        """(closest distance in A, pair-resolved ratio, the species pair that sets it)."""
        d = self._dist(pos)
        r = d / self.ref_pair
        i = int(np.argmin(r))
        return float(d.min()), float(r[i]), self._name[int(self.key[i])]


def selftest_sampler() -> dict:
    """Unit checks on the sampler, run before any configuration is written.

    An Einstein solid (independent isotropic springs, omega_i^2 = k_i / m_i exactly) is the
    test case because every limit has a closed form that does not go through the code's own
    unit chain: the classical <u^2> = kT / k in A^2 with kT in eV and k in eV/A^2, and the
    zero-point <u^2> = hbar^2 / (2 m hbar omega), with hbar^2/(amu A^2) evaluated in eV.
    """
    m = np.array([1.008, 50.94, 207.2])
    k = np.array([2.0, 5.0, 11.0])
    fc = np.zeros((3, 3, 3, 3))
    for i in range(3):
        fc[i, i] = k[i] * np.eye(3)
    lam, e = normal_modes(fc, m, drop_translations=False)
    out = {}
    assert np.allclose(np.sort(lam), np.sort(np.repeat(k / m, 3)), rtol=1e-12)
    nu = signed_thz(lam)

    # (1) classical limit per mode: <q^2> omega^2 = kT, i.e. <u^2> = kT / (m omega^2)
    t_hi = 1e6
    r = mode_variance(nu, t_hi) * lam / (KB_EVK * t_hi)
    out["classical_limit_max_rel_err"] = float(np.max(np.abs(r - 1)))
    assert out["classical_limit_max_rel_err"] < 1e-5, out

    # (2) zero-point limit through an independent path: omega from SI k and m, hbar^2 in eV
    assert abs(THZ_PER_SQRT_EIG - 15.633302) < 1e-5, THZ_PER_SQRT_EIG   # phonopy VaspToTHz
    hb2 = HBAR_JS ** 2 / (AMU_KG * ANG_M ** 2) / EV_J            # eV
    hw = HBAR_EVS * np.sqrt(k * EV_J / ANG_M ** 2 / (m * AMU_KG))  # eV
    u2_zp = hb2 / (2 * m * hw)
    var_q0 = mode_variance(nu, 0.0)
    # map each mode to its atom (Einstein eigenvectors live on one atom each)
    owner = np.argmax(np.abs(e).reshape(3, 3, -1).sum(1), axis=0)
    u2_code = np.array([np.mean(var_q0[owner == i]) / m[i] for i in range(3)])
    out["zero_point_max_rel_err"] = float(np.max(np.abs(u2_code / u2_zp - 1)))
    assert out["zero_point_max_rel_err"] < 1e-7, out

    # (3) sampled variance against the closed forms, quantum (300 K) and classical (1e5 K)
    rng = np.random.default_rng(12345)
    n = 200_000
    for t, expect, key in [
        (300.0, np.array([np.mean(mode_variance(nu[owner == i], 300.0)) / m[i]
                          for i in range(3)]), "sampled_300K_max_rel_err"),
        (1e5, KB_EVK * 1e5 / k, "sampled_classical_max_rel_err"),
    ]:
        u = sample_displacements(e, nu, m, t, n, rng)
        emp = (u ** 2).mean(axis=0).mean(axis=1)
        out[key] = float(np.max(np.abs(emp / expect - 1)))
        assert out[key] < 0.02, out
    out["passed"] = True
    return out


# ------------------------------------------------------------------ stages ----

def _ase_forces(calc, symbols_or_numbers, positions, cell, **kw):
    import ase
    a = ase.Atoms(symbols_or_numbers, positions=positions, cell=cell, pbc=True, **kw)
    a.calc = calc
    f = a.get_forces()
    e = a.get_potential_energy()
    return np.asarray(f, dtype=np.float64), float(e), str(np.asarray(f).dtype)


def stage_fc(args, out: Path) -> None:
    from phonopy import Phonopy
    from mlip_dynstab.calculators import get_calculator
    from mlip_dynstab.cli import PRODUCTION_DISP_ANG
    from mlip_dynstab.harmonic import _ase_to_phonopy_atoms
    from mlip_dynstab.systems import build_atoms, get_spec

    if args.model not in PRODUCTION:
        raise SystemExit(f"fc stage is for the five production models, not {args.model}")
    if _tag(args.model, args.ckpt)[1] is not None:
        raise SystemExit("fc stage uses the production checkpoint only (no --ckpt)")
    *_, systems, _, _ = _scope(args)
    handle = None
    for system in systems:
        path = out / "fc" / f"fc_{system}__{args.model}.npz"
        if path.exists():
            print(f"[skip] {path.name}")
            continue
        if handle is None:
            handle = get_calculator(args.model, device=args.device)
        t0 = time.time()
        atoms = build_atoms(get_spec(system))
        ph = Phonopy(_ase_to_phonopy_atoms(atoms), supercell_matrix=np.diag(SUPERCELL),
                     primitive_matrix="auto")
        ph.generate_displacements(distance=PRODUCTION_DISP_ANG)
        sc = ph.supercell
        f0, _, fdtype = _ase_forces(handle.calc, sc.symbols, sc.positions, sc.cell)
        forces = [_ase_forces(handle.calc, s.symbols, s.positions, s.cell)[0] - f0
                  for s in ph.supercells_with_displacements]
        ph.forces = np.array(forces)
        ph.produce_force_constants()
        ph.symmetrize_force_constants()
        fc = np.asarray(ph.force_constants, dtype=np.float64)
        nat = len(sc.numbers)
        if fc.shape != (nat, nat, 3, 3):
            raise RuntimeError(f"expected full FCs {(nat, nat, 3, 3)}, got {fc.shape}")
        meta = _provenance("fc", args, system=system, model=args.model,
                           model_version=handle.version, model_dtype=_torch_dtype(handle.calc),
                           forces_dtype_returned=fdtype, conservative=CONSERVATIVE[args.model],
                           supercell=list(SUPERCELL), disp_ang=PRODUCTION_DISP_ANG,
                           n_displacements=len(forces), n_atoms=nat,
                           residual_force_rms_ev_ang=float(np.sqrt(np.mean(f0 ** 2))),
                           wall_s=round(time.time() - t0, 2))
        _save_npz(path, fc=fc, positions=np.asarray(sc.positions, float),
                  cell=np.asarray(sc.cell, float), numbers=np.asarray(sc.numbers, int),
                  masses=np.asarray(sc.masses, float), meta=np.array(json.dumps(meta)))
        print(f"[fc] {system}/{args.model}: {len(forces)} displacements, "
              f"residual {meta['residual_force_rms_ev_ang']:.2e} eV/A, {meta['wall_s']} s")


def _config_hash(ref_pos, cell, disp) -> str:
    h = hashlib.sha256()
    for a in (ref_pos, cell, disp):
        h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    return h.hexdigest()[:16]


def stage_configs(args, out: Path) -> None:
    *_, systems, temps, n_configs = _scope(args)
    selftest = selftest_sampler()
    print(f"[selftest] sampler passed: {selftest}")
    for system in systems:
        path = out / "configs" / f"configs_{system}.npz"
        fc_paths = {m: out / "fc" / f"fc_{system}__{m}.npz" for m in PRODUCTION}
        fc_paths = {m: p for m, p in fc_paths.items() if p.exists()}
        if path.exists():
            with np.load(path) as z:
                meta = _meta(z)
            if (meta["n_configs"] != n_configs
                    or [float(t) for t in meta["temps_K"]] != [float(t) for t in temps]):
                raise SystemExit(
                    f"{path} exists with n_configs={meta['n_configs']}, temps={meta['temps_K']}; "
                    f"requested {n_configs}, {temps}. Configurations are frozen: to rebuild, "
                    "delete it AND every forces file that used it.")
            if sorted(fc_paths) != sorted(meta["fc_models"]):
                print(f"[warn] {system}: frozen configs used FCs from {meta['fc_models']}, "
                      f"now available {sorted(fc_paths)}; not regenerating")
            print(f"[skip] {path.name}")
            continue
        if not fc_paths:
            print(f"[wait] {system}: no FC files yet")
            continue
        if len(fc_paths) < len(PRODUCTION) and not (args.smoke or args.allow_partial_fc):
            raise SystemExit(f"{system}: FCs from {sorted(fc_paths)} only; the design needs all "
                             f"five. Pass --allow-partial-fc to freeze configurations anyway.")
        fcs, geo, fc_versions = [], None, {}
        for m, p in sorted(fc_paths.items()):
            with np.load(p) as z:
                g = {k: z[k] for k in ("positions", "cell", "numbers", "masses")}
                fcs.append(z["fc"])
                fc_versions[m] = _meta(z)["model_version"]
            if geo is None:
                geo = g
            elif not (np.array_equal(g["numbers"], geo["numbers"])
                      and np.allclose(g["positions"], geo["positions"], atol=1e-8)
                      and np.allclose(g["cell"], geo["cell"], atol=1e-8)
                      and np.allclose(g["masses"], geo["masses"], atol=1e-3)):
                raise RuntimeError(f"{system}: FC supercells differ between models ({m})")
        masses = geo["masses"]
        lam, evecs = normal_modes(np.mean(fcs, axis=0), masses)
        nu = signed_thz(lam)
        nu_used = np.maximum(np.abs(nu), FREQ_FLOOR_THZ)

        contacts = Contacts(geo["numbers"], geo["cell"], geo["positions"])
        d0 = contacts.d0
        disp = np.empty((len(temps), n_configs, len(masses), 3))
        pair_ratio = np.empty((len(temps), n_configs))
        per_t = {}
        for ti, t in enumerate(temps):
            rng = np.random.default_rng([args.seed, zlib.crc32(system.encode()), int(round(t))])
            u = sample_displacements(evecs, nu_used, masses, t, n_configs, rng)
            com = np.abs(np.einsum("i,cia->ca", masses, u)).max()
            if com > 1e-8 * masses.sum():
                raise RuntimeError(f"{system} {t} K: net mass-weighted displacement {com}")
            disp[ti] = u
            c = [contacts(geo["positions"] + x) for x in u]
            dmin = min(x[0] for x in c)
            pair_ratio[ti] = [x[1] for x in c]
            worst = min(c, key=lambda x: x[1])
            per_t[f"{t:g}"] = {"rms_disp_ang": float(np.sqrt(np.mean(u ** 2))),
                               "max_disp_ang": float(np.linalg.norm(u, axis=2).max()),
                               "min_interatomic_dist_ang": dmin,
                               "min_dist_ratio": dmin / d0,
                               "min_pair_ratio": float(pair_ratio[ti].min()),
                               "min_pair_ratio_pair": worst[2],
                               "frac_configs_below_pair_ratio": float(
                                   np.mean(pair_ratio[ti] < MIN_DIST_RATIO))}
        meta = _provenance(
            "configs", args, system=system, fc_models=sorted(fc_paths),
            fc_model_versions=fc_versions,
            fc_is_full_production_set=len(fc_paths) == len(PRODUCTION),
            n_configs=n_configs, temps_K=[float(t) for t in temps],
            supercell=list(SUPERCELL), n_atoms=int(len(masses)), freq_floor_thz=FREQ_FLOOR_THZ,
            rng_seed=[args.seed, "crc32(system)", "round(T)"],
            n_modes=int(len(nu)), n_imaginary=int(np.sum(nu < 0)),
            n_floored=int(np.sum(np.abs(nu) < FREQ_FLOOR_THZ)),
            min_freq_thz=float(nu.min()), max_freq_thz=float(nu.max()),
            ref_nearest_neighbour_ang=d0, ref_pair_min_ang=contacts.ref_by_pair,
            min_dist_ratio_threshold=MIN_DIST_RATIO, per_temperature=per_t,
            sampler_selftest=selftest,
            config_hash=_config_hash(geo["positions"], geo["cell"], disp))
        _save_npz(path, ref_positions=geo["positions"], cell=geo["cell"],
                  numbers=geo["numbers"], masses=masses, temps_K=np.asarray(temps, float),
                  disp=disp, freqs_thz=nu, freqs_used_thz=nu_used, pair_ratio=pair_ratio,
                  meta=np.array(json.dumps(meta)))
        print(f"[configs] {system}: FCs {sorted(fc_paths)}, min freq {nu.min():.2f} THz, "
              f"{meta['n_imaginary']} imaginary, hash {meta['config_hash']}")


def stage_forces(args, out: Path) -> None:
    from mlip_dynstab.calculators import get_calculator

    tag, ckpt = _tag(args.model, args.ckpt)
    *_, systems, _, _ = _scope(args)
    path = out / "forces" / f"forces_{tag}.npz"
    arrays, meta = {}, None
    if path.exists():
        with np.load(path) as z:
            arrays = {k: z[k] for k in z.files if k != "meta"}
            meta = _meta(z)
    handle = None
    for system in systems:
        cpath = out / "configs" / f"configs_{system}.npz"
        if not cpath.exists():
            print(f"[wait] {system}: no configurations yet")
            continue
        with np.load(cpath) as z:
            cfg = {k: z[k] for k in z.files}
        chash = _meta(cfg)["config_hash"]
        done = (meta or {}).get("systems", {}).get(system)
        if done is not None:
            if done["config_hash"] != chash:
                raise SystemExit(f"{path.name}: {system} was evaluated on configurations "
                                 f"{done['config_hash']}, current are {chash}. Delete the "
                                 "forces file to recompute.")
            print(f"[skip] {tag}/{system}")
            continue
        if handle is None:
            kw = {"model": ckpt} if ckpt else {}
            handle = get_calculator(args.model, device=args.device, **kw)
            meta = meta or _provenance("forces", args, tag=tag, model=args.model, ckpt=ckpt,
                                       model_version=handle.version,
                                       model_dtype=_torch_dtype(handle.calc),
                                       conservative=CONSERVATIVE.get(args.model),
                                       systems={})
            if meta["model_version"] != handle.version:
                raise SystemExit(f"{path.name} was started with {meta['model_version']}, "
                                 f"this env loads {handle.version}")
            # a resumed file keeps its first run's provenance; record each later session too
            meta.setdefault("runs", []).append(
                {k: v for k, v in _provenance("forces", args).items()
                 if k in ("git_head", "git_dirty", "host", "utc", "packages", "gpu")})
        t0 = time.time()
        numbers, cell, ref = cfg["numbers"], cfg["cell"], cfg["ref_positions"]
        f_ref, e_ref, fdtype = _ase_forces(handle.calc, numbers, ref, cell)
        nt, nc = cfg["disp"].shape[:2]
        F = np.empty(cfg["disp"].shape)
        E = np.empty((nt, nc))
        errors = []
        for ti in range(nt):
            for c in range(nc):
                # The floored halide draws contain near-overlapping atoms. One geometry a
                # calculator refuses must not abort the model's run, because a rerun would
                # stop on the same frozen geometry every time; it is stored as NaN and the
                # analysis applies the pre-registered missing-data rule.
                try:
                    F[ti, c], E[ti, c], _ = _ase_forces(handle.calc, numbers,
                                                        ref + cfg["disp"][ti, c], cell)
                except Exception as e:                        # noqa: BLE001
                    F[ti, c], E[ti, c] = np.nan, np.nan
                    errors.append(f"T[{ti}] cfg {c}: {type(e).__name__}: {e}"[:300])
        n_bad = int((~np.isfinite(F).all(axis=(2, 3))).sum())
        if n_bad > nt * nc // 2:
            # a systematic failure (OOM, a poisoned CUDA context, a broken env) is not a
            # property of the geometries; do not save it as data
            raise RuntimeError(f"{tag}/{system}: {n_bad}/{nt * nc} configurations failed; "
                               f"first errors: {errors[:3]}")
        wall = time.time() - t0
        arrays.update({f"{system}__F": F, f"{system}__E": E, f"{system}__F_ref": f_ref,
                       f"{system}__E_ref": np.array(e_ref)})
        meta["systems"][system] = {
            "config_hash": chash, "n_evals": int(nt * nc + 1), "wall_s": round(wall, 2),
            "s_per_eval": round(wall / (nt * nc + 1), 4), "forces_dtype_returned": fdtype,
            "residual_force_rms_ev_ang": float(np.sqrt(np.mean(f_ref ** 2))),
            "n_nonfinite_configs": n_bad, "errors": errors[:20],
            "max_abs_force_ev_ang": float(np.nanmax(np.abs(F))) if n_bad < nt * nc else None,
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        _save_npz(path, **arrays, meta=np.array(json.dumps(meta, default=_jsonable)))
        print(f"[forces] {tag}/{system}: {nt * nc + 1} evals in {wall:.1f} s "
              f"(residual {meta['systems'][system]['residual_force_rms_ev_ang']:.2e} eV/A, "
              f"{n_bad} non-finite, max |F| {meta['systems'][system]['max_abs_force_ev_ang']:.3g})")


# ----------------------------------------------------------------- scoring ----

def _median_ok(per_cfg: np.ndarray, F: np.ndarray) -> tuple[float, int]:
    """Median over the configurations every model in F evaluated to finite forces.

    The pre-registered missing-data rule: a configuration any model in the score's set failed
    on is dropped for all of them (never imputed), and a unit left with fewer than half its
    configurations is unscored rather than scored on a remnant.
    """
    ok = np.isfinite(F).all(axis=(0, 2, 3)) & np.isfinite(per_cfg)
    if 2 * ok.sum() < ok.size:
        return float("nan"), int(ok.sum())
    return float(np.median(per_cfg[ok])), int(ok.sum())


def _spread(forces, tags, system, ti, normalised=False):
    avail = [t for t in tags if system in forces.get(t, {})]
    if ti is None or len(avail) < 2:
        return float("nan"), len(avail), 0
    F = np.stack([forces[t][system][ti] for t in avail])            # (M, n_cfg, N, 3)
    with np.errstate(invalid="ignore"):
        s = np.sqrt(np.mean(F.std(axis=0) ** 2, axis=(1, 2)))        # per configuration
        if normalised:
            s = s / np.sqrt(np.mean(F.mean(axis=0) ** 2, axis=(1, 2)))
    v, n_cfg = _median_ok(s, F)
    return v, len(avail), n_cfg


def _loo(forces, tags, model, system, ti):
    others = [t for t in tags if t != model and system in forces.get(t, {})]
    if ti is None or system not in forces.get(model, {}) or not others:
        return float("nan"), 0, 0
    F = np.stack([forces[t][system][ti] for t in (model, *others)])
    with np.errstate(invalid="ignore"):
        d = np.sqrt(np.mean((F[0] - F[1:].mean(axis=0)) ** 2, axis=(1, 2)))
    v, n_cfg = _median_ok(d, F)
    return v, len(others) + 1, n_cfg


def _auc_block(score, label, cluster, n_perm, seed) -> dict:
    from mlip_dynstab import stats as S
    score = np.asarray(score, float)
    label = np.asarray(label, bool)
    cluster = np.asarray(cluster)
    res = {"n_units": int(score.size), "n_wrong": int(label.sum()),
           "n_clusters": int(len(set(cluster.tolist())))}
    if score.size < 2 or label.all() or not label.any():
        res.update(auc=None, status="single-class labels: AUC undefined")
        return res
    try:
        res.update(S.cluster_permutation_auc(score, label, cluster, n_perm=n_perm, seed=seed))
    except ValueError as e:                     # unequal blocks after dropping units
        res.update(auc=round(S.auc(score, label), 3), p_perm_clustered=None,
                   perm_note=str(e))
    res.update(S.cluster_bootstrap_auc(score, label, cluster, n_boot=n_perm, seed=seed))
    rng = np.random.default_rng(seed)
    naive = np.array([S.auc(score, rng.permutation(label)) for _ in range(n_perm)])
    res["p_perm_naive_unit_level__companion_only"] = round(
        float((np.sum(np.abs(naive - 0.5) >= abs(S.auc(score, label) - 0.5)) + 1)
              / (naive.size + 1)), 4)
    res["status"] = "ok"
    return res


def _result(table, score_col, label_col, required, forces, expected_n, n_perm, seed,
            n_col=None) -> dict:
    missing = [t for t in required if t not in forces]
    d = table[np.isfinite(table[score_col].astype(float))]
    full = (d[n_col] == len(required)).all() if (n_col and len(d)) else bool(len(d))
    res = {"score": score_col, "label": label_col, "models_required": list(required),
           "models_missing": missing, "n_units_expected": int(expected_n),
           "complete": bool(not missing and len(d) == expected_n and full)}
    if d.empty:
        res["status"] = "no scored units"
        return res
    res.update(_auc_block(d[score_col], d[label_col], d["system"], n_perm, seed))
    return res


def _verdict(primary: dict, ex_orb: dict, design_ok: bool) -> str:
    if not (primary.get("complete") and design_ok):
        return "INCOMPLETE: not the pre-registered primary (models, units or configs missing)"
    lo, hi = primary.get("ci_lo"), primary.get("ci_hi")
    if lo is None or hi is None or primary.get("auc") is None:
        return "undefined"
    if lo > 0.5 and (ex_orb.get("auc") or 0) > 0.5:
        return "supported: the force spread flags wrong consensus calls (pre-registered rule met)"
    if hi < 0.5:
        return "anti-informative: the spread is larger where the consensus is right"
    return "not shown: the clustered CI spans 0.5, or the ex-ORB-v2 estimate is <= 0.5"


def stage_analyze(args, out: Path) -> None:
    import pandas as pd
    from mlip_dynstab import stats as S

    df, c5, c4, pm, ref, systems, temps, n_cfg_req = _scope(args)
    keep = lambda d: d[d["system"].isin(systems) & d["T"].isin(temps)]  # noqa: E731
    c5, c4, pm = keep(c5), keep(c4), keep(pm)

    configs = {}
    for s in systems:
        p = out / "configs" / f"configs_{s}.npz"
        if p.exists():
            with np.load(p) as z:
                configs[s] = {"temps": [float(t) for t in z["temps_K"]], "meta": _meta(z)}
    forces, finfo = {}, {}
    ledger_version = pm.groupby("model")["model_version"].first().to_dict()
    for p in sorted((out / "forces").glob("forces_*.npz")):
        with np.load(p) as z:
            meta = _meta(z)
            tag = meta["tag"]
            forces[tag] = {}
            for s, info in meta["systems"].items():
                if s not in configs:
                    continue
                if info["config_hash"] != configs[s]["meta"]["config_hash"]:
                    raise SystemExit(f"{p.name}: {s} evaluated on stale configurations")
                forces[tag][s] = z[f"{s}__F"]
        lv = ledger_version.get(tag)
        finfo[tag] = {"file": p.name, "model_version": meta["model_version"],
                      "ledger_model_version": lv,
                      "matches_ledger": (meta["model_version"] == lv) if lv else None,
                      "model_dtype": meta.get("model_dtype"),
                      "conservative": meta.get("conservative"),
                      "systems": sorted(forces[tag]), "host": meta.get("host"),
                      "git_head": meta.get("git_head"),
                      "wall_s_total": round(sum(v["wall_s"] for v in meta["systems"].values()), 1),
                      "s_per_eval_median": float(np.median(
                          [v["s_per_eval"] for v in meta["systems"].values()] or [np.nan])),
                      "max_residual_force_rms_ev_ang": max(
                          [v["residual_force_rms_ev_ang"] for v in meta["systems"].values()]
                          or [float("nan")]),
                      "n_nonfinite_configs": {s: v.get("n_nonfinite_configs")
                                              for s, v in meta["systems"].items()
                                              if v.get("n_nonfinite_configs")},
                      "errors": {s: v["errors"] for s, v in meta["systems"].items()
                                 if v.get("errors")}}

    def ti_of(system, T):
        c = configs.get(system)
        return c["temps"].index(float(T)) if c and float(T) in c["temps"] else None

    # consensus-level table
    c4i = c4.set_index(["system", "T"])
    variants = {
        "S_5model": (PRODUCTION, False), "S_ex_orb_4model": (EX_ORB, False),
        "S_mace_committee": (MACE_COMMITTEE, False),
        "S_mattersim_committee": (MATTERSIM_COMMITTEE, False),
        "Snorm_5model": (PRODUCTION, True), "Snorm_ex_orb_4model": (EX_ORB, True),
    }
    rows = []
    for _, u in c5.iterrows():
        ti = ti_of(u["system"], u["T"])
        r4 = c4i.loc[(u["system"], u["T"])]
        rec = {"system": u["system"], "T": float(u["T"]), "gt_stable": bool(u["gt_stable"]),
               "consensus_stable": bool(u["consensus_stable"]),
               "consensus_wrong": bool(u["consensus_wrong"]),
               "consensus_wrong_ex_orb": bool(r4["consensus_wrong"]),
               "vote_disagreement": float(u["disagreement"]),
               "vote_disagreement_ex_orb": float(r4["disagreement"]),
               "freq_std_thz": float(u["freq_std_thz"]),
               "freq_std_thz_ex_orb": float(r4["freq_std_thz"])}
        geo_t =(configs[u["system"]]["meta"]["per_temperature"].get(f"{float(u['T']):g}", {})
                 if ti is not None else {})
        for k in ("rms_disp_ang", "min_interatomic_dist_ang", "min_dist_ratio",
                  "min_pair_ratio", "frac_configs_below_pair_ratio"):
            rec[k] = geo_t.get(k, float("nan"))
        rec["min_pair_ratio_pair"] = geo_t.get("min_pair_ratio_pair")
        for name, (tags, norm) in variants.items():
            rec[name], rec[f"{name}__n_models"], rec[f"{name}__n_cfg"] = _spread(
                forces, tags, u["system"], ti, norm)
        rows.append(rec)
    units = pd.DataFrame(rows).sort_values(["system", "T"]).reset_index(drop=True)

    # per-model table
    prow = []
    for _, u in pm.iterrows():
        ti = ti_of(u["system"], u["T"])
        rec = {"system": u["system"], "model": u["model"], "T": float(u["T"]),
               "pred_stable": bool(u["pred_stable"]), "gt_stable": bool(u["gt_stable"]),
               "wrong": bool(u["wrong"])}
        none = (float("nan"), 0, 0)
        rec["loo_5model"], rec["loo_5model__n_models"], rec["loo_5model__n_cfg"] = _loo(
            forces, PRODUCTION, u["model"], u["system"], ti)
        rec["loo_ex_orb"], rec["loo_ex_orb__n_models"], rec["loo_ex_orb__n_cfg"] = (
            _loo(forces, EX_ORB, u["model"], u["system"], ti) if u["model"] != "orb_v2"
            else none)
        comm = {"mace_mp0": MACE_COMMITTEE, "mattersim": MATTERSIM_COMMITTEE}.get(u["model"])
        rec["committee"], rec["committee__n_models"], rec["committee__n_cfg"] = (
            _spread(forces, comm, u["system"], ti) if comm else none)
        prow.append(rec)
    per = pd.DataFrame(prow).sort_values(["system", "model", "T"]).reset_index(drop=True)

    n_perm, seed = args.n_perm, args.seed
    n_u = len(units)
    R = lambda *a, **k: _result(*a, n_perm=n_perm, seed=seed, **k)  # noqa: E731
    cons = {
        "primary_S_5model": R(units, "S_5model", "consensus_wrong", PRODUCTION, forces, n_u,
                              n_col="S_5model__n_models"),
        "a_S_ex_orb_4model": R(units, "S_ex_orb_4model", "consensus_wrong_ex_orb", EX_ORB,
                               forces, n_u, n_col="S_ex_orb_4model__n_models"),
        "b_S_mace_committee": R(units, "S_mace_committee", "consensus_wrong", MACE_COMMITTEE,
                                forces, n_u, n_col="S_mace_committee__n_models"),
        "c_S_mattersim_committee": R(units, "S_mattersim_committee", "consensus_wrong",
                                     MATTERSIM_COMMITTEE, forces, n_u,
                                     n_col="S_mattersim_committee__n_models"),
        "d_Snorm_5model": R(units, "Snorm_5model", "consensus_wrong", PRODUCTION, forces, n_u,
                            n_col="Snorm_5model__n_models"),
        "d_Snorm_ex_orb_4model": R(units, "Snorm_ex_orb_4model", "consensus_wrong_ex_orb",
                                   EX_ORB, forces, n_u, n_col="Snorm_ex_orb_4model__n_models"),
    }
    physical = units[units["min_pair_ratio"].astype(float) >= MIN_DIST_RATIO]
    cons["g_S_5model_no_overlap"] = R(physical, "S_5model", "consensus_wrong", PRODUCTION,
                                      forces, len(physical), n_col="S_5model__n_models")
    cons["g_S_5model_no_overlap"].update(
        criterion="min over every configuration of the unit of the pair-resolved contact "
                  f"ratio >= {MIN_DIST_RATIO}",
        min_pair_ratio_threshold=MIN_DIST_RATIO,
        units_dropped=[f"{s}@{t:g}" for s, t in
                       units.loc[~units.index.isin(physical.index), ["system", "T"]].values])
    permodel = {
        "e_loo_pooled_5model": R(per, "loo_5model", "wrong", PRODUCTION, forces, len(per),
                                 n_col="loo_5model__n_models"),
        "e_loo_pooled_ex_orb_4model": R(per[per["model"] != "orb_v2"], "loo_ex_orb", "wrong",
                                        EX_ORB, forces, int((per["model"] != "orb_v2").sum()),
                                        n_col="loo_ex_orb__n_models"),
    }
    for m in PRODUCTION:
        sub = per[per["model"] == m]
        permodel[f"e_loo_{m}_5model"] = R(sub, "loo_5model", "wrong", PRODUCTION, forces,
                                          len(sub), n_col="loo_5model__n_models")
        if m != "orb_v2":
            permodel[f"e_loo_{m}_ex_orb"] = R(sub, "loo_ex_orb", "wrong", EX_ORB, forces,
                                              len(sub), n_col="loo_ex_orb__n_models")
    for m, comm in (("mace_mp0", MACE_COMMITTEE), ("mattersim", MATTERSIM_COMMITTEE)):
        sub = per[per["model"] == m]
        permodel[f"f_committee_{m}_own_call"] = R(sub, "committee", "wrong", comm, forces,
                                                  len(sub), n_col="committee__n_models")

    per_t = {}
    for T, g in units.groupby("T"):
        per_t[f"{T:g}"] = R(g, "S_5model", "consensus_wrong", PRODUCTION, forces, len(g),
                            n_col="S_5model__n_models")
        per_t[f"{T:g}"]["median_S_5model"] = (float(g["S_5model"].median())
                                              if g["S_5model"].notna().any() else None)

    reference = {  # the call-level signals of section 3.4, on exactly these units
        "vote_disagreement_5model": _auc_block(units["vote_disagreement"],
                                               units["consensus_wrong"], units["system"],
                                               n_perm, seed),
        "freq_std_5model": _auc_block(units["freq_std_thz"], units["consensus_wrong"],
                                      units["system"], n_perm, seed),
        "vote_disagreement_ex_orb": _auc_block(units["vote_disagreement_ex_orb"],
                                               units["consensus_wrong_ex_orb"],
                                               units["system"], n_perm, seed),
        "freq_std_ex_orb": _auc_block(units["freq_std_thz_ex_orb"],
                                      units["consensus_wrong_ex_orb"], units["system"],
                                      n_perm, seed),
    }
    if not args.smoke and reference["vote_disagreement_5model"].get("auc") is not None:
        if abs(reference["vote_disagreement_5model"]["auc"] - ref["auc_vote_disagreement"]) > 1e-3:
            raise RuntimeError("reference vote AUC does not reproduce h3_guardrail_summary")

    ok = units["S_5model"].notna()
    descriptive = {
        "n_units_scored_S_5model": int(ok.sum()),
        "median_S_5model_by_consensus": {
            "wrong": (float(units.loc[ok & units["consensus_wrong"], "S_5model"].median())
                      if (ok & units["consensus_wrong"]).any() else None),
            "right": (float(units.loc[ok & ~units["consensus_wrong"], "S_5model"].median())
                      if (ok & ~units["consensus_wrong"]).any() else None)},
        "spearman_S_5model_vs_vote_disagreement": (
            round(S.spearman(units.loc[ok, "S_5model"], units.loc[ok, "vote_disagreement"]), 3)
            if ok.sum() >= 3 else None),
        "spearman_S_5model_vs_freq_std": (
            round(S.spearman(units.loc[ok, "S_5model"], units.loc[ok, "freq_std_thz"]), 3)
            if ok.sum() >= 3 else None),
    }

    # the configurations themselves must match the design, not just the model roster
    design_ok = bool(not args.smoke and args.seed == SEED and args.n_perm == N_PERM
                     and set(configs) == set(systems) and all(
        c["meta"]["rng_seed"][0] == SEED and
        c["meta"]["n_configs"] == N_CONFIGS and c["meta"]["fc_is_full_production_set"]
        and c["meta"]["freq_floor_thz"] == FREQ_FLOOR_THZ and c["meta"]["supercell"] == list(SUPERCELL)
        and [float(t) for t in c["meta"]["temps_K"]] == [float(t) for t in temps]
        for c in configs.values()))
    # The labels are the calls the ledger's model versions made; forces from another version
    # of a production model would score a model that never made those calls.
    version_ok = all(finfo.get(m, {}).get("matches_ledger") is True for m in PRODUCTION)
    design_ok = design_ok and version_ok
    doc = __doc__
    prereg = doc[doc.index("PRE-REGISTRATION"):doc.index("END PRE-REGISTRATION")]
    primary_text = " ".join(prereg[prereg.index("PRIMARY METRIC."):
                                   prereg.index("Secondary,")].split())
    summary = {
        "_provenance": _provenance("analyze", args),
        "smoke": bool(args.smoke),
        "preregistration": {"primary_metric": primary_text,
                            "block_sha256": hashlib.sha256(prereg.encode()).hexdigest(),
                            "text": prereg},
        "design_as_preregistered": design_ok,
        "production_versions_match_ledger": version_ok,
        "verdict": _verdict(cons["primary_S_5model"], cons["a_S_ex_orb_4model"], design_ok),
        # not part of the decision rule; placed beside it because the pre-registration names
        # (g) as the check that a positive primary is not the overlapped-atom artifact
        "g_artifact_check": {k: cons["g_S_5model_no_overlap"].get(k) for k in
                             ("auc", "ci_lo", "ci_hi", "n_units", "n_wrong", "n_clusters",
                              "status")},
        "primary": cons["primary_S_5model"],
        "consensus_level": cons,
        "per_model_level": permodel,
        "per_temperature_primary__descriptive": per_t,
        "reference_call_level_signals": reference,
        "descriptive": descriptive,
        "inputs": {
            "unit_set": {"n_consensus_units": int(n_u), "n_per_model_units": int(len(per)),
                         "systems": systems, "temps_K": temps,
                         "n_configs_requested": n_cfg_req,
                         "h3_guardrail_summary_reference": ref},
            "configs": {s: {k: c["meta"].get(k) for k in
                            ("config_hash", "fc_models", "fc_is_full_production_set",
                             "n_configs", "temps_K", "n_imaginary", "n_floored",
                             "min_freq_thz", "ref_pair_min_ang", "per_temperature",
                             "git_head", "utc")}
                        for s, c in configs.items()},
            "forces": finfo,
            "models_missing_from_design": [t for t in dict.fromkeys(
                (*PRODUCTION, *MACE_COMMITTEE, *MATTERSIM_COMMITTEE)) if t not in forces],
        },
        "notes": [
            "std across models is the population std (ddof=0), matching np.nanstd in "
            "analysis.h3_ensemble_guardrail.",
            "Ex-ORB-v2 consensus labels are rebuilt from four votes; a 2-2 tie is called stable "
            "(analysis.h3_ensemble_guardrail, stable-vote fraction >= 0.5).",
            "Configurations sit near the unrelaxed registry cell, not each model's relaxed "
            "cell where the production calls were made; that is the price of a common input.",
            "The MACE-MP-0 small/medium/large and MatterSim 1M/5M checkpoints differ in size "
            "and training run; they are family committees, not a deep ensemble of one model.",
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "summary.json", summary)
    units.to_csv(out / "units_consensus.csv", index=False)
    per.to_csv(out / "units_per_model.csv", index=False)

    p = summary["primary"]
    print(f"[analyze] forces for {sorted(forces)}; missing {summary['inputs']['models_missing_from_design']}")
    print(f"  primary S_5model: AUC {p.get('auc')}  clustered p {p.get('p_perm_clustered')}  "
          f"CI [{p.get('ci_lo')}, {p.get('ci_hi')}]  n={p.get('n_units')}  "
          f"complete={p.get('complete')}")
    for k, v in {**cons, **permodel}.items():
        print(f"  {k:34s} AUC {v.get('auc')}  p_cl {v.get('p_perm_clustered')}  "
              f"CI [{v.get('ci_lo')}, {v.get('ci_hi')}]  n={v.get('n_units')}  {v.get('status')}")
    print(f"  verdict: {summary['verdict']}")
    print(f"  (g) artifact check beside it: {summary['g_artifact_check']}")
    print(f"wrote {out / 'summary.json'}")


# --------------------------------------------------------------------- cli ----

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True,
                    choices=["fc", "configs", "forces", "analyze", "selftest"])
    ap.add_argument("--model", choices=list(PRODUCTION))
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint for a committee member (mace_mp0: small|large; "
                         "mattersim: mattersim-v1.0.0-1m). The production one maps to the "
                         "production file.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-configs", type=int, default=N_CONFIGS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--smoke", action="store_true",
                    help="2 systems x 2 temperatures x 2 configurations")
    ap.add_argument("--allow-partial-fc", action="store_true",
                    help="freeze configurations from fewer than five FC sets")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    out = args.out or (OUT_DEFAULT / "smoke" if args.smoke else OUT_DEFAULT)
    if args.stage in ("fc", "forces") and not args.model:
        ap.error(f"--stage {args.stage} needs --model")
    if args.ckpt is not None and args.ckpt not in ALLOWED_CKPT.get(args.model, ()):
        ap.error(f"--ckpt {args.ckpt} is not a checkpoint the analysis uses for {args.model}; "
                 f"allowed: {ALLOWED_CKPT.get(args.model, ())}")
    if args.stage == "selftest":
        print(json.dumps(selftest_sampler(), indent=2))
    elif args.stage == "fc":
        stage_fc(args, out)
    elif args.stage == "configs":
        stage_configs(args, out)
    elif args.stage == "forces":
        stage_forces(args, out)
    else:
        stage_analyze(args, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
