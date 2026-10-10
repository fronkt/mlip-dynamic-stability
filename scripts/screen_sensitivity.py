"""Systematic, compute-free sensitivity of the single-mode soft-mode screen (Referee 1.5).

Referee 1 (item 5) asked for the sensitivity of the soft-mode screen to the polynomial fit
window, the sampling range and the cell commensurability, "systematically reported". Nothing
here is a new measurement. Every number re-solves the CACHED, temperature-independent E(Q) maps
of the current generation (``results/cache/softmode_v3m24_*.json``; the v2 files are a
superseded generation and are ignored) with the production solver functions imported from
``mlip_dynstab.finite_t`` (``_fit_double_well``, ``_scha_branch``, ``_solve_scha``,
``_sym_curvature_freq``) and the production call logic of ``compute_finite_t_softmode``: the
cubic phase is unstable at T if ANY screened mode condenses. ``finite_t.py`` is not modified.

Baseline first. Re-solving every cached map at production settings must reproduce the ledger's
softmode calls unit by unit (``pred_stable`` per system/model/T) and mode by mode
(``ft_modes_stable``); the script aborts if it does not.

Axes (each reported as mode-level condensation-call flips / mode-T evaluations, unit-level call
flips / units, FE-perovskite recall k/30 at T <= 300 K by ``analysis.displacive_recall``, the
SrTiO3 gate per model, and softmode accuracy on the scored set, each with and without ORB-v2):

  (a) Fit window. ``_fit_double_well`` keeps points with dE <= emin + max(0.06 eV, 5|emin|).
      ``fit_double_well_param`` below is a verbatim copy with the multiplier and the floor
      exposed; it is asserted to reproduce production bit-for-bit at (5, 60 meV). Swept over
      multiplier {3, 5, 8} x floor {30, 60, 120} meV.
  (b) Sampling range. The production map has 10 points at Q = 0.45 (i/9)^2 A. Truncate to
      Q <= {0.25, 0.30, 0.35, 0.40, 0.45} A and refit with the production window. On that grid
      Q <= 0.30 and Q <= 0.35 keep the same 8 points (0.2722 A in, 0.3556 A out), so those two
      rows are identical by construction and are reported as such.
  (c) Frozen-cell normalisation convention (derivation below).
  (d) Implementation constants: the condensation threshold (Q0 > 1.5 grid steps = 0.0075 A),
      the centroid scan box (0.6 A, 121 points) and the brentq fallback in ``_scha_branch``.

Frozen-cell normalisation: the derivation
----------------------------------------
Meaning of Q. A screened mode is frozen into its minimal commensurate cell C (phonopy
modulation, ``_mode_pattern``): atom i moves by dr_i = Q u_i, with the pattern normalised so
max_{i,alpha} |u_{i,alpha}| = 1. Q (A) is therefore the largest Cartesian displacement
component of any atom in C (not the largest atomic displacement, which can be up to sqrt(3) Q).
The map is V(Q) = E_C(Q) - E_C(0), the energy of the whole cell C, and the collective kinetic
energy is (1/2) M Qdot^2 with M = sum_{i in C} m_i |u_i|^2, also summed over C.

Tiling. Let C_n be n copies of C carrying the same per-atom pattern (u repeated in every copy).
Then (i) max|u| is still 1, so the same Q describes the same physical distortion: Q, and every
Q-denominated constant (q_max 0.45 A, box 0.6 A, threshold 0.0075 A), is invariant; (ii) the
tiled configuration is the same periodic crystal, so its energy is extensive, V_n(Q) = n V(Q)
exactly, i.e. (a, b, c) -> n (a, b, c); (iii) M_n = sum over n copies = n M. Invariant under n:
the static minimum Q_min of V (V' = 0 is scale-free), the harmonic frequency
omega_h^2 = V''(0)/M = 2a/M, and the classical T = 0 call. Not invariant: the thermal state.
The n-cell Hamiltonian is

    H_n = p^2/(2 n M) + n V(Q) = n [ p^2/(2 (n^2 M)) + V(Q) ],

so exp(-H_n/kT) is the minimal-cell problem with mass n^2 M at temperature T/n. A larger
normalisation cell is simultaneously colder (barrier n dV against kT) and more classical (zero
point hbar omega/2 unchanged against barrier n dV). The SCHA Gibbs-Bogoliubov functional
inherits the mapping exactly: with K = <V''> unchanged at fixed sigma^2, omega_n^2 = nK/(nM) =
K/M and sigma_n^2 = hbar/(2 n M omega_n) coth(hbar omega_n / 2kT), which is identically the
width of the (n^2 M, T/n) problem, whose frequency is omega_n / n; hence F_n(Q0) = n F'(Q0), the
same argmin and the same call, and the reported frequency scales by n. Both identities are
checked numerically on one mode (``scaling_checks``). A single-mode treatment therefore has no
n-independent answer: as n -> infinity the fluctuation of a uniform order parameter vanishes
and every double well condenses at every T (the mean-field limit). The minimal-cell convention
fixes an implicit correlation volume; it is a model choice, not a numerical detail. Note also
that the 60 meV window floor is a per-cell energy, so a pipeline re-run in a doubled cell would
also halve the effective floor; that side of the question is axis (a)'s 30 meV floor.

Conventions re-solved with (a, b, c, M) -> n (a, b, c, M), fit points unchanged:
  minimal     n = 1 (production)
  common_fc   n = prod(fc_supercell) / prod(dim): the force-constant supercell the q-search used
  per_fu      n = 1 / (prod(dim) Z): per formula unit, Z = formula units in the input cell
  doubled     n = 2
  x8          n = 8 (supplementary; a fixed factor, to compare with the audit's 'N = 8' count)

Outputs results/screen_sensitivity.json and results/screen_sensitivity.md (ESI table).
Run from the repo root:

    python scripts/screen_sensitivity.py [--workers 6] [--no-mlip-check]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from functools import reduce
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))          # match the convention in the sibling scripts

import numpy as np                     # noqa: E402
import pandas as pd                    # noqa: E402
import scipy.optimize as _scipy_optimize  # noqa: E402

from mlip_dynstab import analysis as A  # noqa: E402
from mlip_dynstab import stats as S     # noqa: E402
from mlip_dynstab.finite_t import (    # noqa: E402
    _fit_double_well, _scha_branch, _solve_scha, _sym_curvature_freq)

CACHE_DIR = REPO / "results" / "cache"
CACHE_PREFIX = "softmode_v3m24_"
OUT_JSON = REPO / "results" / "screen_sensitivity.json"
OUT_MD = REPO / "results" / "screen_sensitivity.md"

TEMPS = (100.0, 300.0, 600.0, 900.0)
ORB = "orb_v2"
FE = ["batio3_cubic", "knbo3_cubic", "pbtio3_cubic"]
SRTIO3 = "srtio3_cubic"

# Production constants of finite_t.py, restated so that drift in the module trips an assert.
PROD_MULT, PROD_FLOOR = 5.0, 0.06      # _fit_double_well: window = max(0.06 eV, 5 |emin|)
PROD_QMAX, PROD_NPTS = 0.45, 10        # compute_finite_t_softmode: q_max, n_pts
PROD_QBOX, PROD_NQ = 0.6, 121          # _solve_scha: q_box, nq  (grid step 0.005 A)
PROD_THRESH = 1.5                      # _solve_scha: q_thresh = 1.5 grid steps = 0.0075 A


# ------------------------------------------------------------- brentq probe ----
# _scha_branch does `from scipy.optimize import brentq` at call time, so wrapping the module
# attribute lets us see, per centroid evaluation, whether the self-consistent width came from
# brentq or from the grid-nearest fallback (no sign change bracketed, or brentq raised). The
# wrapper passes arguments and results through untouched.

_REAL_BRENTQ = _scipy_optimize.brentq
_BQ = {"called": False, "ok": False}


def _counting_brentq(*args, **kwargs):
    _BQ["called"] = True
    root = _REAL_BRENTQ(*args, **kwargs)
    _BQ["ok"] = True
    return root


def _install_probe():
    import warnings
    _scipy_optimize.brentq = _counting_brentq
    # _scha_branch takes sqrt of K<0 on part of its sigma^2 grid and then drops the NaNs
    # (finite_t.py:701, :711); production emits the same warnings. Silence them in the log.
    warnings.filterwarnings("ignore", message="invalid value encountered in sqrt",
                            category=RuntimeWarning)


# ----------------------------------------------------------------- fitting ----

def fit_double_well_param(Qs, dE, mult: float = PROD_MULT, floor: float = PROD_FLOOR):
    """Verbatim copy of ``finite_t._fit_double_well`` with the window multiplier and floor
    exposed. Returns ((a, b, c), keep_mask). Asserted equal to production at (5, 0.06 eV)."""
    Qs = np.asarray(Qs, float); dE = np.asarray(dE, float)
    emin = float(dE.min())
    window = max(floor, mult * abs(min(emin, 0.0)))       # eV
    keep = dE <= emin + window
    if keep.sum() < 4:
        keep = np.zeros_like(dE, bool)
        keep[np.argsort(Qs)[:4]] = True
    Qf, Ef = Qs[keep], dE[keep]
    Am = np.stack([Qf ** 2, Qf ** 4, Qf ** 6], axis=1)
    coef, *_ = np.linalg.lstsq(Am, Ef, rcond=None)
    if coef[2] < 0 or (coef[1] < 0 and coef[2] <= 0):
        A2 = np.stack([Qf ** 2, Qf ** 4], axis=1)
        c2, *_ = np.linalg.lstsq(A2, Ef, rcond=None)
        coef = np.array([c2[0], c2[1], 0.0])
    return tuple(float(x) for x in coef), keep


def truncate(Qs, dE, qtrunc):
    Qs = np.asarray(Qs, float); dE = np.asarray(dE, float)
    if qtrunc is None:
        return Qs, dE
    m = Qs <= qtrunc + 1e-9
    return Qs[m], dE[m]


# ------------------------------------------------------------------ solving ----

def solve_scha_probe(a, b, c, m_eff, T, q_box=PROD_QBOX, nq=PROD_NQ) -> dict:
    """Replica of ``finite_t._solve_scha`` built on the production ``_scha_branch``: the same
    centroid grid, the same argmin and the same 1.5-step threshold, but it also returns the
    grid step (so other thresholds are pure post-processing), the argmin index and the brentq
    fallback bookkeeping. Asserted identical to ``_solve_scha`` on every baseline mode-T."""
    Q0s = np.linspace(0.0, q_box, nq)
    Fs = np.empty(nq); om = np.empty(nq)
    fb = np.zeros(nq, bool); inf = np.zeros(nq, bool)
    for j, q0 in enumerate(Q0s):
        _BQ["called"] = False; _BQ["ok"] = False
        F, _s2, omega, _K = _scha_branch(q0, a, b, c, m_eff, T)
        Fs[j] = F; om[j] = omega
        if not np.isfinite(F):
            inf[j] = True
        elif not (_BQ["called"] and _BQ["ok"]):
            fb[j] = True
    i = int(np.argmin(Fs))
    step = float(Q0s[1] - Q0s[0])
    Q0 = float(Q0s[i])
    stable = bool(Q0 <= PROD_THRESH * step)
    eff = float(np.sign(1 if stable else -1) * om[i] / (2 * np.pi) / 1e12)
    return {"Q0": Q0, "idx": i, "step": step, "stable_prod_thresh": stable, "eff": eff,
            "at_edge": bool(i == nq - 1), "n_fallback": int(fb.sum()), "n_inf": int(inf.sum()),
            "fallback_at_argmin": bool(fb[i]), "n_eval": int(nq)}


def _job(key):
    kind, a, b, c, m, T, q_box, nq = key
    if kind == "prod":
        eff, Q0, st = _solve_scha(a, b, c, m, T)
        return key, {"eff": float(eff), "Q0": float(Q0), "stable": bool(st)}
    r = solve_scha_probe(a, b, c, m, T, q_box, nq)
    curv = _sym_curvature_freq(a, b, c, m, T)
    r["curv"] = r["eff"] if curv is None else float(curv)
    return key, r


def _run_jobs(keys, workers):
    keys = list(keys)
    out, t0 = {}, time.time()
    if workers <= 1:
        _install_probe()
        for i, k in enumerate(keys):
            kk, r = _job(k); out[kk] = r
            if (i + 1) % 500 == 0:
                print(f"    {i + 1}/{len(keys)}  {time.time() - t0:.0f}s", flush=True)
        return out
    with Pool(workers, initializer=_install_probe) as pool:
        for i, (kk, r) in enumerate(pool.imap_unordered(_job, keys, chunksize=4)):
            out[kk] = r
            if (i + 1) % 1000 == 0:
                print(f"    {i + 1}/{len(keys)}  {time.time() - t0:.0f}s", flush=True)
    return out


# -------------------------------------------------------------------- data ----

@dataclass
class Mode:
    uid: int
    system: str
    model: str
    idx: int
    placeholder: bool
    harm_thz: float
    q: list
    band: int
    dim: list
    fc: list
    z_fu: int
    m_eff: float
    a: float
    b: float
    c: float
    Qs: np.ndarray
    dE: np.ndarray
    well_meV: float


def _formula_units(system: str) -> int:
    from mlip_dynstab.systems import build_atoms, get_spec
    atoms = build_atoms(get_spec(system))
    counts = {}
    for s in atoms.get_chemical_symbols():
        counts[s] = counts.get(s, 0) + 1
    return int(reduce(math.gcd, counts.values()))


def load_modes(sm: pd.DataFrame) -> tuple[list[Mode], dict]:
    pairs = sorted(set(zip(sm["system"], sm["model"])))
    modes, files, uid = [], {}, 0
    zcache = {}
    for system, model in pairs:
        hits = sorted(glob.glob(str(CACHE_DIR / f"{CACHE_PREFIX}{system}_{model}_sc*.json")))
        assert len(hits) == 1, (system, model, hits)
        files[(system, model)] = Path(hits[0]).name
        c = json.load(open(hits[0]))
        if system not in zcache:
            zcache[system] = _formula_units(system)
        for j, e in enumerate(c["modes"]):
            modes.append(Mode(uid=uid, system=system, model=model, idx=j,
                              placeholder=e["band"] < 0, harm_thz=float(e["harm_thz"]),
                              q=list(e["q"]), band=int(e["band"]), dim=[int(x) for x in e["dim"]],
                              fc=[int(x) for x in c["fc_supercell"]], z_fu=zcache[system],
                              m_eff=float(e["m_eff"]), a=float(e["a"]), b=float(e["b"]),
                              c=float(e["c"]), Qs=np.asarray(e["Qs"], float),
                              dE=np.asarray(e["dE"], float), well_meV=float(e["well_depth_meV"])))
            uid += 1
    all_files = {Path(p).name for p in glob.glob(str(CACHE_DIR / f"{CACHE_PREFIX}*.json"))}
    orphans = sorted(all_files - set(files.values()))
    assert not orphans, f"cache files with no ledger unit: {orphans}"
    return modes, files


# ---------------------------------------------------------------- settings ----

@dataclass(frozen=True)
class Setting:
    key: str
    axis: str
    label: str
    mult: float = PROD_MULT
    floor: float = PROD_FLOOR
    qtrunc: float | None = None
    cell: str = "minimal"
    q_box: float = PROD_QBOX
    nq: int = PROD_NQ
    thresh: float = PROD_THRESH

    @property
    def is_production(self) -> bool:
        return (self.mult == PROD_MULT and self.floor == PROD_FLOOR and self.qtrunc in (None, PROD_QMAX)
                and self.cell == "minimal" and self.q_box == PROD_QBOX and self.nq == PROD_NQ
                and self.thresh == PROD_THRESH)


def build_settings() -> list[Setting]:
    s = [Setting("production", "baseline", "Production: 5x / 60 meV, Q <= 0.45 A, minimal cell, "
                                           "1.5 steps, box 0.6 A")]
    for fl in (0.03, 0.06, 0.12):
        for mu in (3.0, 5.0, 8.0):
            s.append(Setting(f"window_m{int(mu)}_f{int(round(fl * 1000))}", "a_fit_window",
                             f"window {int(mu)}x well depth, floor {int(round(fl * 1000))} meV",
                             mult=mu, floor=fl))
    for qt in (0.25, 0.30, 0.35, 0.40, 0.45):
        s.append(Setting(f"trunc_{qt:.2f}", "b_sampling_range", f"sampled Q <= {qt:.2f} A",
                         qtrunc=qt))
    for cell, lab in (("minimal", "minimal commensurate cell"),
                      ("common_fc", "common FC supercell (q-search cell)"),
                      ("per_fu", "per formula unit"),
                      ("doubled", "doubled minimal cell"),
                      ("x8", "8x minimal cell (supplementary)")):
        s.append(Setting(f"cell_{cell}", "c_cell_normalisation", lab, cell=cell))
    for th in (0.5, 1.5, 3.0):
        s.append(Setting(f"thresh_{th:g}", "d_constants",
                         f"condensation threshold {th:g} steps ({th * 0.005:.4f} A)", thresh=th))
    s.append(Setting("box_0.6", "d_constants", "centroid box 0.6 A, 121 pts"))
    s.append(Setting("box_1.2", "d_constants", "centroid box 1.2 A, 241 pts (same step)",
                     q_box=1.2, nq=241))
    return s


def cell_factor(m: Mode, cell: str) -> float:
    nd = int(np.prod(m.dim))
    if cell == "minimal":
        return 1.0
    if cell == "common_fc":
        return float(np.prod(m.fc)) / nd
    if cell == "per_fu":
        return 1.0 / (nd * m.z_fu)
    if cell == "doubled":
        return 2.0
    if cell == "x8":
        return 8.0
    raise ValueError(cell)


def mode_problem(m: Mode, s: Setting):
    """(a, b, c, M) that setting ``s`` hands the solver for mode ``m``, plus the keep mask."""
    Qs, dE = truncate(m.Qs, m.dE, None if s.qtrunc in (None, PROD_QMAX) else s.qtrunc)
    (a, b, c), keep = fit_double_well_param(Qs, dE, s.mult, s.floor)
    n = cell_factor(m, s.cell)
    if n != 1.0:
        a, b, c, M = n * a, n * b, n * c, n * m.m_eff
    else:
        M = m.m_eff
    return (a, b, c, M), keep


# ----------------------------------------------------------------- metrics ----

def _kn(k, n):
    return {"k": int(k), "n": int(n), "fmt": f"{int(k)}/{int(n)}"}


def _rate(k, n):
    r = S.rate_ci(int(k), int(n))
    return r.as_dict() | {"fmt": S.fmt_rate(r)}


def unit_table(modes, mode_res, setting) -> pd.DataFrame:
    """Production call logic of compute_finite_t_softmode, per unit, from per-mode results."""
    by_unit = {}
    for m in modes:
        by_unit.setdefault((m.system, m.model), []).append(m)
    rows = []
    for (system, model), ms in by_unit.items():
        for T in TEMPS:
            solved = []
            for m in ms:
                if m.placeholder:
                    solved.append({"Q0": 0.0, "stable": True, "curv": m.harm_thz})
                    continue
                r = mode_res[(m.uid, T)]
                st = bool(r["Q0"] <= setting.thresh * r["step"])
                solved.append({"Q0": r["Q0"], "stable": st, "curv": r["curv"]})
            cond = [x for x in solved if not x["stable"]]
            decide = max(cond, key=lambda x: x["Q0"]) if cond else min(solved, key=lambda x: x["curv"])
            rows.append({"system": system, "model": model, "temperature_K": T,
                         "pred_new": len(cond) == 0, "min_eff_new": float(min(x["curv"] for x in solved)),
                         "decide_Q0": float(decide["Q0"]), "n_condensed": len(cond)})
    return pd.DataFrame(rows)


def mode_calls(modes, mode_res, setting) -> dict:
    return {(m.uid, T): not (mode_res[(m.uid, T)]["Q0"] <= setting.thresh * mode_res[(m.uid, T)]["step"])
            for m in modes if not m.placeholder for T in TEMPS}


def modified_ledger(df: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    sm = d["method"] == "softmode"
    key = units.set_index(["system", "model", "temperature_K"])
    idx = pd.MultiIndex.from_arrays([d.loc[sm, "system"], d.loc[sm, "model"], d.loc[sm, "temperature_K"]])
    d.loc[sm, "pred_stable"] = key.loc[idx, "pred_new"].to_numpy()
    d.loc[sm, "min_eff_freq_thz"] = key.loc[idx, "min_eff_new"].to_numpy()
    return d


def headline(dm: pd.DataFrame) -> dict:
    """FE recall (paper definition), scored-set accuracy and the SrTiO3 gate on a ledger view."""
    out = {}
    rec = A.displacive_recall(dm).set_index("method")
    out["fe_recall"] = _rate(rec.loc["softmode", "correct_unstable"], rec.loc["softmode", "n_valid"])
    t = A.low_t_false_stable(dm, exclude_bcc=True)
    k = int((t["TP"] + t["TN"]).sum()); n = int(t["n"].sum())
    out["accuracy"] = _rate(k, n)
    out["accuracy_per_model"] = {r.model: _kn(r.TP + r.TN, r.n) for r in t.itertuples()}
    out["false_stable"] = _rate(int(t["FP_false_stable"].sum()),
                                int((t["FP_false_stable"] + t["TN"]).sum()))
    out["false_unstable"] = _rate(int(t["FN_false_unstable"].sum()),
                                  int((t["FN_false_unstable"] + t["TP"]).sum()))
    # The T <= 300 K scored set is nearly blind to over-condensation: few of its truly stable
    # units carry a screened imaginary mode (diagnostics "le300K_scored_stable_units_with_modes";
    # the rest are harmonically stable controls with nothing to condense). The same scored systems over the full
    # ladder (100-900 K) add the 600/900 K rows where extra condensation shows up as
    # false-unstables, so a convention that condenses more cannot score well for free.
    ta = A.low_t_false_stable(dm, t_max=900.0, exclude_bcc=True)
    out["accuracy_all_T"] = _rate(int((ta["TP"] + ta["TN"]).sum()), int(ta["n"].sum()))
    out["false_stable_all_T"] = _rate(int(ta["FP_false_stable"].sum()),
                                      int((ta["FP_false_stable"] + ta["TN"]).sum()))
    out["false_unstable_all_T"] = _rate(int(ta["FN_false_unstable"].sum()),
                                        int((ta["FN_false_unstable"] + ta["TP"]).sum()))
    s = dm[(dm["method"] == "softmode") & (dm["system"] == SRTIO3)]
    gate = {}
    for model, g in s.groupby("model"):
        g = g.set_index("temperature_K")["pred_stable"].astype(bool)
        c100 = not bool(g.loc[100.0])
        hi = bool(g.loc[[300.0, 600.0, 900.0]].all())
        gate[model] = {"condenses_100K": c100, "stable_300K_plus": hi, "pass": c100 and hi}
    out["srtio3_gate"] = gate
    out["srtio3_gate_passing"] = _kn(sum(v["pass"] for v in gate.values()), len(gate))
    return out


def setting_metrics(modes, mode_res, setting, ref_mode_calls, ref_units, df) -> dict:
    mc = mode_calls(modes, mode_res, setting)
    units = unit_table(modes, mode_res, setting)
    model_of = {m.uid: m.model for m in modes}
    out = {"setting": asdict(setting) | {"is_production": setting.is_production}}

    def mflip(pred):
        ks = [k for k in mc if pred(k)]
        flips = [k for k in ks if mc[k] != ref_mode_calls[k]]
        return {"k": len(flips), "n": len(ks), "fmt": f"{len(flips)}/{len(ks)}",
                "to_condensed": sum(1 for k in flips if mc[k]),
                "to_uncondensed": sum(1 for k in flips if not mc[k])}
    out["mode_flips"] = {
        "all": mflip(lambda k: True),
        "le300K": mflip(lambda k: k[1] <= 300.0),
        "exORB": mflip(lambda k: model_of[k[0]] != ORB),
        "exORB_le300K": mflip(lambda k: model_of[k[0]] != ORB and k[1] <= 300.0)}

    u = units.merge(ref_units[["system", "model", "temperature_K", "pred_new"]],
                    on=["system", "model", "temperature_K"], suffixes=("", "_ref"))

    def uflip(g):
        f = g[g["pred_new"] != g["pred_new_ref"]]
        return {"k": int(len(f)), "n": int(len(g)), "fmt": f"{len(f)}/{len(g)}",
                "to_unstable": int((~f["pred_new"]).sum()), "to_stable": int(f["pred_new"].sum()),
                # flips are repeated measurements on few systems (Referee 3): say how few
                "n_systems": int(f["system"].nunique()), "systems": sorted(f["system"].unique()),
                "units": [f"{r.system}/{r.model}/{int(r.temperature_K)}K" for r in f.itertuples()]}
    lo = u["temperature_K"] <= 300.0
    out["unit_flips"] = {"all": uflip(u), "exORB": uflip(u[u["model"] != ORB]),
                         "le300K": uflip(u[lo]), "exORB_le300K": uflip(u[lo & (u["model"] != ORB)])}

    dm = modified_ledger(df, units)
    out["all_models"] = headline(dm)
    out["exORB"] = headline(dm[dm["model"] != ORB])

    edge_units = units[units["decide_Q0"] >= setting.q_box - 1e-9]
    out["scan_edge_units"] = {"all": _kn(len(edge_units), len(units)),
                              "exORB": _kn(int((edge_units["model"] != ORB).sum()),
                                           int((units["model"] != ORB).sum()))}
    real = [m for m in modes if not m.placeholder]
    ev = [mode_res[(m.uid, T)] for m in real for T in TEMPS]
    out["solver"] = {
        "mode_T_argmin_at_edge": _kn(sum(r["at_edge"] for r in ev), len(ev)),
        "centroid_evaluations": int(sum(r["n_eval"] for r in ev)),
        "brentq_fallback": int(sum(r["n_fallback"] for r in ev)),
        "unbound_centroids_F_inf": int(sum(r["n_inf"] for r in ev)),
        "fallback_at_argmin": _kn(sum(r["fallback_at_argmin"] for r in ev), len(ev))}
    out["_units"] = units
    out["_mode_calls"] = mc
    return out


# ------------------------------------------------------------- diagnostics ----

def kept_set_identity(modes) -> dict:
    real = [m for m in modes if not m.placeholder]
    out = {}
    for fl in (0.03, 0.06, 0.12):
        same = 0
        for m in real:
            ks = [fit_double_well_param(m.Qs, m.dE, mu, fl)[1] for mu in (3.0, 5.0, 8.0)]
            same += all(np.array_equal(ks[0], k) for k in ks[1:])
        out[f"floor_{int(round(fl * 1000))}meV"] = _kn(same, len(real))
    shallow = [m for m in real if m.well_meV <= 7.5]
    shallow_same = sum(all(np.array_equal(fit_double_well_param(m.Qs, m.dE, 3.0, 0.06)[1],
                                          fit_double_well_param(m.Qs, m.dE, mu, 0.06)[1])
                           for mu in (5.0, 8.0)) for m in shallow)
    out["modes_shallower_than_7p5meV"] = len(shallow)
    out["of_which_identical_at_60meV"] = int(shallow_same)
    return out


def unbracketed(modes) -> dict:
    """Modes whose sampled E(Q) is still decreasing at the largest sampled Q (dE[-1] < dE[-2])."""
    real = [m for m in modes if not m.placeholder]
    out = {}
    for qt in (0.25, 0.30, 0.35, 0.40, 0.45):
        hits = []
        for m in real:
            Qs, dE = truncate(m.Qs, m.dE, qt)
            if dE[-1] < dE[-2]:
                hits.append(m)
        by_sys = {}
        for m in hits:
            by_sys[m.system] = by_sys.get(m.system, 0) + 1
        out[f"Q<={qt:.2f}"] = _kn(len(hits), len(real)) | {
            "n_points": int(len(truncate(real[0].Qs, real[0].dE, qt)[0])),
            "exORB": _kn(sum(1 for m in hits if m.model != ORB),
                         sum(1 for m in real if m.model != ORB)),
            "by_system": dict(sorted(by_sys.items()))}
    return out


def fitted_without_sampled_well(modes, ref_mode_calls, ref_units) -> dict:
    """Modes with no sampled point below E(0) whose fitted polynomial nevertheless has a
    displaced minimum below zero inside the 0.6 A centroid box."""
    grid = np.linspace(0.0, PROD_QBOX, 6001)
    hits = []
    for m in modes:
        if m.placeholder or m.dE[1:].min() < 0:
            continue
        V = m.a * grid ** 2 + m.b * grid ** 4 + m.c * grid ** 6
        if V.min() < 0:
            hits.append((m, float(-V.min() * 1000), float(grid[int(np.argmin(V))])))
    bad = {m.uid for m, _, _ in hits}
    ncalls = sum(1 for m, _, _ in hits for T in TEMPS if ref_mode_calls[(m.uid, T)])
    # does any unit call depend on these modes alone?
    dependent = []
    by_unit = {}
    for m in modes:
        if not m.placeholder:
            by_unit.setdefault((m.system, m.model), []).append(m)
    for (system, model), ms in by_unit.items():
        for T in TEMPS:
            cond = [m for m in ms if ref_mode_calls[(m.uid, T)]]
            if cond and all(m.uid in bad for m in cond):
                dependent.append(f"{system}/{model}/{int(T)}K")
    by = {}
    for m, _, _ in hits:
        by[f"{m.system}/{m.model}"] = by.get(f"{m.system}/{m.model}", 0) + 1
    ti = [h[1] for h in hits if h[0].system == "ti_bcc" and h[0].model == ORB]
    return {"n_modes": len(hits), "by_unit": by,
            "definition": "every sampled dE >= 0, yet min of a Q^2 + b Q^4 + c Q^6 on a 1e-4 A grid "
                          "over [0, 0.6] A is < 0 (depth per minimal cell)",
            "modes": [{"unit": f"{m.system}/{m.model}", "mode_index": m.idx, "dim": m.dim,
                       "harm_thz": round(m.harm_thz, 3), "fitted_depth_meV": round(dd, 1),
                       "at_Q_A": round(qq, 4)} for m, dd, qq in hits],
            "fitted_depth_meV_range_ti_orb": [round(min(ti), 1), round(max(ti), 1)] if ti else None,
            "fitted_depth_meV_range": [round(min(h[1] for h in hits), 1),
                                       round(max(h[1] for h in hits), 1)] if hits else None,
            "fitted_min_Q_range_A": [round(min(h[2] for h in hits), 4),
                                     round(max(h[2] for h in hits), 4)] if hits else None,
            "mode_T_condensation_calls": int(ncalls),
            "unit_calls_resting_only_on_them": dependent}


def _scored_stable_with_modes(sm: pd.DataFrame, real) -> dict:
    """Truly stable units of the T <= 300 K scored set that have at least one screened imaginary
    mode, i.e. the only units on which a convention that condenses more can be penalised."""
    with_modes = {(m.system, m.model) for m in real}
    d = sm[(sm["temperature_K"] <= 300.0) & ~sm["system"].str.contains("bcc")
           & ~sm["system"].isin(A.borderline_systems()) & sm["gt_stable"].astype(bool)]
    hit = d[[(s, mo) in with_modes for s, mo in zip(d["system"], d["model"])]]
    return _kn(len(hit), len(d)) | {"systems": sorted(hit["system"].unique())}


def cell_sizes(modes) -> dict:
    real = [m for m in modes if not m.placeholder]
    fu = {}
    for m in real:
        k = int(np.prod(m.dim)) * m.z_fu
        fu[k] = fu.get(k, 0) + 1
    fam = {}
    for m in real:
        f = "bcc" if "bcc" in m.system else ("fluorite" if "o2_" in m.system else "perovskite")
        fam.setdefault(f, set()).add(int(np.prod(m.dim)) * m.z_fu)
    return {"formula_units_in_minimal_cell_histogram": dict(sorted(fu.items())),
            "range_by_family": {k: [min(v), max(v)] for k, v in fam.items()},
            "common_fc_factor_range": [min(cell_factor(m, "common_fc") for m in real),
                                       max(cell_factor(m, "common_fc") for m in real)],
            "per_fu_factor_range": [min(cell_factor(m, "per_fu") for m in real),
                                    max(cell_factor(m, "per_fu") for m in real)]}


def srtio3_mode_detail(modes, mode_res, setting) -> list:
    out = []
    for m in modes:
        if m.system != SRTIO3 or m.placeholder:
            continue
        row = {"model": m.model, "q": m.q, "harm_thz": round(m.harm_thz, 3), "dim": m.dim,
               "n_factor": cell_factor(m, setting.cell)}
        for T in TEMPS:
            r = mode_res[(m.uid, T)]
            row[f"Q0_{int(T)}K"] = r["Q0"]
            row[f"condensed_{int(T)}K"] = not (r["Q0"] <= setting.thresh * r["step"])
        out.append(row)
    return out


# ------------------------------------------------------------ scaling check ----

def scaling_checks(modes, do_mlip: bool) -> dict:
    """Numerical check of the normalisation transformation on one mode (SrTiO3 / MACE-MP-0,
    R-point tilt, minimal cell 2x2x2).

    (1) Solver identity: (n a, n b, n c, n M, T) must give the same centroid and call as
        (a, b, c, n^2 M, T/n), with the reported frequency larger by exactly n.
    (2) Harmonic-frequency invariance: sqrt(2a/M) is unchanged by the rescaling.
    (3) Optional, needs mace-torch on CPU: rebuild the R-point pattern with the production
        _harmonic_phonon/_mode_pattern, freeze it into the minimal 2x2x2 cell and into a
        4x2x2 cell, and check max|u| = 1 in both, M_4x2x2 / M_2x2x2 = 2 and
        E_4x2x2(Q) / E_2x2x2(Q) = 2 at several Q. (Unrelaxed cell; extensivity does not need
        the relaxed geometry, and this is not a reproduction of the cached map.)"""
    m = next(x for x in modes if x.system == SRTIO3 and x.model == "mace_mp0"
             and not x.placeholder and x.q == [0.5, 0.5, 0.5])
    out = {"mode": {"system": m.system, "model": m.model, "q": m.q, "dim": m.dim,
                    "m_eff": m.m_eff, "a": m.a, "b": m.b, "c": m.c}}
    rows = []
    for n in (2.0, 8.0, 0.125):
        for T in (100.0, 300.0):
            e1, q1, s1 = _solve_scha(n * m.a, n * m.b, n * m.c, n * m.m_eff, T)
            e2, q2, s2 = _solve_scha(m.a, m.b, m.c, n * n * m.m_eff, T / n)
            rows.append({"n": n, "T": T, "Q0_ncell": q1, "Q0_equiv": q2, "stable_ncell": s1,
                         "stable_equiv": s2, "freq_ratio": e1 / e2 if e2 else None,
                         "identical_call_and_Q0": bool(q1 == q2 and s1 == s2),
                         "freq_ratio_equals_n": bool(abs(e1 / e2 - n) < 1e-6 * n) if e2 else None})
    out["solver_identity"] = rows
    out["solver_identity_all_pass"] = bool(all(r["identical_call_and_Q0"] and r["freq_ratio_equals_n"]
                                               for r in rows))
    def fh(a, M):                     # signed omega_h = sign(a) sqrt(2|a|/M), in THz
        return math.copysign(math.sqrt(2 * abs(a) * 1.602176634e-19 / 1e-20
                                       / (M * 1.66053906660e-27)) / (2 * math.pi) / 1e12, a)
    out["harmonic_freq_thz"] = {"minimal": fh(m.a, m.m_eff), "doubled": fh(2 * m.a, 2 * m.m_eff),
                                "per_fu_(1/8)": fh(m.a / 8, m.m_eff / 8),
                                "cached_harm_thz": m.harm_thz}
    if not do_mlip:
        out["mlip_extensivity"] = {"skipped": "run without --no-mlip-check"}
        return out
    try:
        import ase
        from mlip_dynstab.calculators import get_calculator
        from mlip_dynstab.finite_t import _harmonic_phonon, _mode_pattern
        from mlip_dynstab.systems import build_atoms, get_spec
        handle = get_calculator("mace_mp0", device="cpu")
        calc = handle.calc
        prim = build_atoms(get_spec(SRTIO3))
        ph = _harmonic_phonon(prim, calc, (2, 2, 2))
        q, band = [0.5, 0.5, 0.5], 0
        dim1, base1, u1, M1 = _mode_pattern(ph, q, band, 2)
        ph.run_modulations(dimension=[4, 2, 2], phonon_modes=[[q, band, 1.0, 0.0]])
        mods, sc = ph.get_modulations_and_supercell()
        base2 = ase.Atoms(symbols=sc.symbols, scaled_positions=sc.scaled_positions,
                          cell=sc.cell, pbc=True)
        u2 = np.real(mods[0]); u2 = u2 / np.abs(u2).max()
        M2 = float(np.sum(base2.get_masses()[:, None] * u2 ** 2))

        def energy(base, u, Q):
            a = base.copy(); a.set_positions(base.get_positions() + Q * u); a.calc = calc
            return float(a.get_potential_energy())
        E01, E02 = energy(base1, u1, 0.0), energy(base2, u2, 0.0)
        erows = []
        for Q in (0.05, 0.1, 0.2, 0.3):
            v1 = energy(base1, u1, Q) - E01
            v2 = energy(base2, u2, Q) - E02
            erows.append({"Q": Q, "V_min_cell_eV": v1, "V_doubled_eV": v2,
                          "ratio": v2 / v1 if v1 else None})
        same_pattern = bool(np.allclose(np.sort(np.repeat(np.abs(u1).ravel(), 2)),
                                        np.sort(np.abs(u2).ravel()), atol=1e-10))
        out["mlip_extensivity"] = {
            "model_version": handle.version, "n_atoms": [len(base1), len(base2)],
            "dims": [list(map(int, dim1)), [4, 2, 2]],
            "max_abs_u": [float(np.abs(u1).max()), float(np.abs(u2).max())],
            "pattern_multiset_is_tiled": same_pattern,
            "M_eff_amu": [M1, M2], "M_ratio": M2 / M1,
            "E_per_cell": erows,
            "E0_ratio": E02 / E01,
            "pass": bool(same_pattern and abs(M2 / M1 - 2) < 1e-9
                         and all(abs(r["ratio"] - 2) < 1e-4 for r in erows))}
    except Exception as exc:                           # environment-dependent; record, don't die
        out["mlip_extensivity"] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


# ------------------------------------------------------------------ report ----

def _cell(x: dict) -> str:
    return x["fmt"]


def _rate_short(r: dict) -> str:
    return f"{r['k']}/{r['n']}"


def _rate_ci(r: dict) -> str:
    """Count with its Wilson 95% interval (Referee 3: every rate with counts and intervals)."""
    return f"{r['k']}/{r['n']} [{r['lo']:.2f}, {r['hi']:.2f}]"


def write_md(results: dict, path: Path):
    rows = results["settings"]
    lines = [
        "# Table S14 (draft). Sensitivity of the single-mode soft-mode screen (Referee 1.5)",
        "",
        "Compute-free: every row re-solves the cached E(Q) maps (`results/cache/softmode_v3m24_*.json`)",
        "with the production solver. Each cell gives all five models, then ORB-v2 excluded.",
        "Mode flips count condensation calls that change relative to production over all",
        "mode-temperature evaluations; unit flips count changed stability calls over all",
        "(system, model, T) units. FE recall is the paper's `displacive_recall` (BaTiO3, KNbO3,",
        "PbTiO3; T <= 300 K). The SrTiO3 gate passes when a model condenses at 100 K and is",
        "stable at 300, 600 and 900 K. Accuracy is on the scored finite-T set (T <= 300 K,",
        "bcc and KTaO3 excluded); 'Accuracy, all T' scores the same systems over 100-900 K and",
        "gives the false-unstable count in parentheses. Rates carry Wilson 95% intervals.",
        "",
        "| Axis | Setting | Mode flips (all T) | Unit flips (all T) | Unit flips (T <= 300 K) "
        "| FE recall | SrTiO3 gate | Accuracy | Accuracy, all T (false-unstable) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    axis_name = {"baseline": "Production", "a_fit_window": "(a) Fit window",
                 "b_sampling_range": "(b) Sampling range",
                 "c_cell_normalisation": "(c) Frozen-cell normalisation",
                 "d_constants": "(d) Constants"}
    for r in rows:
        s = r["setting"]
        prod = " (prod.)" if s["is_production"] and s["axis"] != "baseline" else ""
        lines.append(
            f"| {axis_name[s['axis']]} | {s['label']}{prod} "
            f"| {_cell(r['mode_flips']['all'])} · {_cell(r['mode_flips']['exORB'])} "
            f"| {_cell(r['unit_flips']['all'])} · {_cell(r['unit_flips']['exORB'])} "
            f"| {_cell(r['unit_flips']['le300K'])} · {_cell(r['unit_flips']['exORB_le300K'])} "
            f"| {_rate_ci(r['all_models']['fe_recall'])} · {_rate_ci(r['exORB']['fe_recall'])} "
            f"| {_cell(r['all_models']['srtio3_gate_passing'])} · {_cell(r['exORB']['srtio3_gate_passing'])} "
            f"| {_rate_ci(r['all_models']['accuracy'])} · {_rate_ci(r['exORB']['accuracy'])} "
            f"| {_rate_ci(r['all_models']['accuracy_all_T'])} ({_rate_short(r['all_models']['false_unstable_all_T'])})"
            f" · {_rate_ci(r['exORB']['accuracy_all_T'])} ({_rate_short(r['exORB']['false_unstable_all_T'])}) |")
    d = results["diagnostics"]
    st = {r["setting"]["key"]: r for r in rows}
    sc = results["scaling_checks"]
    mx = sc.get("mlip_extensivity", {})
    fbc = d["brentq_fallback_cached_coefficient_path"]
    lines += [
        "",
        "Notes.",
        "",
        f"- Baseline reproduction: {results['baseline']['unit_calls_match']['fmt']} unit calls and "
        f"{results['baseline']['mode_T_calls_match']['fmt']} mode-T condensation calls match the ledger.",
        f"- (a) The kept point set is identical at 3x, 5x and 8x (60 meV floor) for "
        f"{d['kept_set_identity']['floor_60meV']['fmt']} modes, including all "
        f"{d['kept_set_identity']['modes_shallower_than_7p5meV']} modes shallower than 7.5 meV; "
        f"the multiplier sweep cannot move them.",
        f"- (b) Wells still descending at the largest sampled Q (0.45 A): "
        f"{d['unbracketed']['Q<=0.45']['fmt']} modes. Q <= 0.30 and Q <= 0.35 A keep the same 8 "
        f"points and are identical by construction.",
        f"- (c) Rescaling the normalisation cell by n maps (a, b, c, M) to n(a, b, c, M), which is the "
        f"minimal-cell problem at mass n^2 M and temperature T/n; checked on the SrTiO3/MACE-MP-0 "
        f"R mode (solver identity {'passes' if sc['solver_identity_all_pass'] else 'FAILS'}"
        + (f"; MACE-MP-0 energy and mass ratios for a doubled cell "
           f"{mx['E_per_cell'][1]['ratio']:.6f} and {mx['M_ratio']:.6f}" if mx.get("pass") else "")
        + ").",
        f"- (d) Every non-zero centroid is >= {d['mode_Q0_distribution']['smallest_nonzero_A']} A, so "
        f"thresholds of 0.5-3 steps (0.0025-0.015 A) cannot change a call. Unit order parameters at "
        f"the 0.6 A box edge: {st['production']['scan_edge_units']['all']['fmt']}; beyond the sampled "
        f"0.45 A: {d['unit_Q0_beyond_sampled_range_0p45A']['fmt']}; widening the box to 1.2 A moves "
        f"none of them off condensation. brentq falls back to the grid-nearest width on "
        f"{fbc['fmt']} centroid evaluations ({100 * fbc['share']:.1f}%), at the deciding argmin on "
        f"{fbc['at_argmin']['fmt']} (production convention; "
        f"{st['cell_common_fc']['solver']['fallback_at_argmin']['fmt']} under the common-FC convention).",
    ]
    fl = [f"{r['setting']['label']}: {r['unit_flips']['all']['k']} over "
          f"{r['unit_flips']['all']['n_systems']} ({', '.join(r['unit_flips']['all']['systems'])})"
          for r in rows if r["unit_flips"]["all"]["k"] > 0]
    if fl:
        lines.append("- Unit flips are repeated measurements on few systems (all T): " + "; ".join(fl) + ".")
    le = d["le300K_scored_stable_units_with_modes"]

    def fu(key):
        return _rate_short(st[key]["all_models"]["false_unstable_all_T"])

    def acc(key):
        return _rate_short(st[key]["all_models"]["accuracy_all_T"])
    lines += [
        f"- The T <= 300 K accuracy column is nearly blind to over-condensation: only {le['fmt']} of "
        f"its truly stable units carry a screened imaginary mode ({', '.join(le['systems'])} at "
        f"300 K); the rest have no imaginary commensurate mode and cannot condense under any "
        f"convention. Intervals are unit-level Wilson and do not account for clustering by "
        f"system. Over the full ladder the extra condensation "
        f"of larger normalisation cells shows up as false-unstables at 600-900 K (production "
        f"{fu('production')}, common FC {fu('cell_common_fc')}, doubled {fu('cell_doubled')}, 8x "
        f"{fu('cell_x8')}), and all-T accuracy is {acc('production')}, {acc('cell_common_fc')}, "
        f"{acc('cell_doubled')} and {acc('cell_x8')} respectively. How accuracy ranks the "
        f"conventions depends on which temperatures are scored, so it is not a test of which "
        f"normalisation is physically right.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def audit_comparison(res: dict) -> list:
    """Every audit number this script can check, with the value computed here."""
    st = {r["setting"]["key"]: r for r in res["settings"]}
    d = res["diagnostics"]
    b = res["baseline"]
    cmp = []

    def add(claim, audit, computed, agree):
        cmp.append({"claim": claim, "audit": audit, "computed": computed,
                    "agrees": None if agree is None else bool(agree)})
    add("baseline reproduces ledger mode-T calls", "1512/1512", b["mode_T_calls_match"]["fmt"],
        b["mode_T_calls_match"]["fmt"] == "1512/1512")
    add("kept set identical at 3x/5x/8x, 60 meV floor", "261/378",
        d["kept_set_identity"]["floor_60meV"]["fmt"], d["kept_set_identity"]["floor_60meV"]["fmt"] == "261/378")
    add("modes shallower than 7.5 meV (all inside the inert set)", "101",
        f"{d['kept_set_identity']['modes_shallower_than_7p5meV']} "
        f"({d['kept_set_identity']['of_which_identical_at_60meV']} inert)",
        d["kept_set_identity"]["modes_shallower_than_7p5meV"] == 101
        and d["kept_set_identity"]["of_which_identical_at_60meV"] == 101)
    for fl, aud in ((30, "2/756"), (120, "8/756")):
        r = st[f"window_m5_f{fl}"]["mode_flips"]
        add(f"floor {fl} meV mode flips", aud, f"{r['le300K']['fmt']} (T<=300 K); {r['all']['fmt']} (all T)",
            aud in (r["le300K"]["fmt"], r["all"]["fmt"]))
    add("unbracketed wells at Q = 0.45 A", "77/378", d["unbracketed"]["Q<=0.45"]["fmt"],
        d["unbracketed"]["Q<=0.45"]["fmt"] == "77/378")
    add("unit Q0 at the 0.6 A scan-box edge", "26/400", st["production"]["scan_edge_units"]["all"]["fmt"],
        st["production"]["scan_edge_units"]["all"]["fmt"] == "26/400")
    tr = {k: st[f"trunc_{k}"] for k in ("0.25", "0.30", "0.35", "0.40")}
    add("truncation test: mode flips 3/756 and 9/756, 0 unit flips (levels and scope unstated "
        "in audit)", "3/756, 9/756; 0 unit flips",
        "; ".join(f"Q<={k}: modes {v['mode_flips']['le300K']['fmt']} (T<=300 K), "
                  f"{v['mode_flips']['all']['fmt']} (all T); units {v['unit_flips']['le300K']['fmt']} "
                  f"(T<=300 K), {v['unit_flips']['all']['fmt']} (all T)" for k, v in tr.items()),
        tr["0.40"]["mode_flips"]["le300K"]["k"] == 3 and tr["0.30"]["mode_flips"]["le300K"]["k"] == 9
        and tr["0.40"]["unit_flips"]["all"]["k"] == 0 and tr["0.30"]["unit_flips"]["all"]["k"] == 0)
    for key, aud_flips, aud_rec in (("cell_common_fc", 14, "28/30"), ("cell_per_fu", 39, "5/30"),
                                    ("cell_doubled", None, "26/30")):
        r = st[key]
        rec = _rate_short(r["all_models"]["fe_recall"])
        ok = rec == aud_rec and (aud_flips is None or r["unit_flips"]["le300K"]["k"] == aud_flips)
        add(f"{key}: unit flips / FE recall",
            f"{aud_flips if aud_flips is not None else 'n/a'} flips (denominator unstated); recall {aud_rec}",
            f"{r['unit_flips']['le300K']['fmt']} flips at T<=300 K, {r['unit_flips']['all']['fmt']} "
            f"at all T; recall {rec}", ok)
    add("production FE recall", "16/30", _rate_short(st["production"]["all_models"]["fe_recall"]),
        _rate_short(st["production"]["all_models"]["fe_recall"]) == "16/30")
    for key, aud in (("cell_doubled", "89/756"), ("cell_x8", "176/756")):
        r = st[key]["mode_flips"]
        add(f"{key} mode flips", aud, f"{r['le300K']['fmt']} (T<=300 K); {r['all']['fmt']} (all T)",
            aud in (r["le300K"]["fmt"], r["all"]["fmt"]))
    g2 = st["cell_doubled"]["all_models"]["srtio3_gate"]["mace_mp0"]
    add("SrTiO3 gate breaks for MACE-MP-0 under a doubled cell", "breaks", json.dumps(g2), not g2["pass"])
    gf = st["cell_per_fu"]["all_models"]["srtio3_gate"]
    add("SrTiO3 gate breaks under per-f.u. normalisation", "breaks",
        st["cell_per_fu"]["all_models"]["srtio3_gate_passing"]["fmt"] + " models pass",
        st["cell_per_fu"]["all_models"]["srtio3_gate_passing"]["k"]
        < st["production"]["all_models"]["srtio3_gate_passing"]["k"])
    ms = [x for x in st["cell_common_fc"]["srtio3_modes"]
          if x["model"] == "mattersim" and x["q"] == [0.0, 0.0, 0.0]]
    add("common FC cell: MatterSim SrTiO3 Gamma mode condenses at 100 K", "condenses",
        json.dumps([{k: v for k, v in x.items() if k in ("q", "n_factor", "Q0_100K", "condensed_100K")}
                    for x in ms]),
        bool(ms) and all(x["condensed_100K"] for x in ms))
    fb = d["brentq_fallback"]
    fbc = d["brentq_fallback_cached_coefficient_path"]
    add("brentq fallback share of centroid evaluations", "8,442 (4.6%)",
        f"{fbc['k']} of {fbc['n']} ({100 * fbc['k'] / fbc['n']:.2f}%) on the as-run cached-coefficient "
        f"path; {fb['k']} on the refit path", fbc["k"] == 8442)
    add("fallback never at the deciding argmin", "0/1512", fb["at_argmin"]["fmt"], fb["at_argmin"]["k"] == 0)
    fw = d["fitted_without_sampled_well"]
    add("fitted wells with no sampled well", "19 modes (18 Ti/ORB-v2 + 1 PbTiO3/ORB-v2), 100-376 meV, "
        "48 mode-T condensation calls, no unit call depends",
        f"{fw['n_modes']} modes {fw['by_unit']}, {fw['fitted_depth_meV_range']} meV, "
        f"{fw['mode_T_condensation_calls']} calls, dependent units {fw['unit_calls_resting_only_on_them']}",
        fw["n_modes"] == 19 and fw["mode_T_condensation_calls"] == 48
        and not fw["unit_calls_resting_only_on_them"]
        and fw["fitted_depth_meV_range_ti_orb"] == [100.0, 376.0])
    cs = d["cell_sizes"]["range_by_family"].get("perovskite")
    add("minimal-cell size in perovskites", "1 to 8 f.u.", str(cs), cs == [1, 8])
    return cmp


# -------------------------------------------------------------------- main ----

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-mlip-check", action="store_true",
                    help="skip the MACE-MP-0 CPU extensivity check of the normalisation derivation")
    ap.add_argument("--dry-run", action="store_true", help="count the solves and stop")
    ap.add_argument("--solve-cache", default=None,
                    help="optional pickle of raw solves, reused across runs (development aid)")
    args = ap.parse_args()
    workers = max(1, min(6, args.workers))
    t_start = time.time()
    _install_probe()

    df = A.load_canonical()
    sm = df[df["method"] == "softmode"].copy()
    assert len(sm) == 400 and sorted(sm["temperature_K"].unique()) == list(TEMPS), "unexpected ledger"
    modes, files = load_modes(sm)
    real = [m for m in modes if not m.placeholder]
    print(f"{len(files)} cache files, {len(real)} screened modes, "
          f"{len(modes) - len(real)} harmonically-stable placeholders", flush=True)

    # --- fit reproduction: parameterised copy == production == cached coefficients
    n_copy_eq = n_cache_eq = 0
    for m in real:
        prod = _fit_double_well(m.Qs, m.dE)
        mine, _ = fit_double_well_param(m.Qs, m.dE, PROD_MULT, PROD_FLOOR)
        n_copy_eq += prod == mine
        n_cache_eq += prod == (m.a, m.b, m.c)
    assert n_copy_eq == len(real), "parameterised fit copy does not reproduce _fit_double_well"
    for m in real:
        assert len(m.Qs) == PROD_NPTS and abs(m.Qs[-1] - PROD_QMAX) < 1e-12

    settings = build_settings()
    problems = {}                    # (setting.key, uid) -> (a, b, c, M)
    keys = set()
    for s in settings:
        for m in real:
            (a, b, c, M), _ = mode_problem(m, s)
            problems[(s.key, m.uid)] = (a, b, c, M)
            for T in TEMPS:
                keys.add(("probe", a, b, c, M, T, s.q_box, s.nq))
    prod_keys = {("prod", m.a, m.b, m.c, m.m_eff, T, PROD_QBOX, PROD_NQ) for m in real for T in TEMPS}
    # the replica on the CACHED coefficients too, so replica == _solve_scha is tested on
    # identical inputs (the refit differs from the cached coefficients in the last bits)
    keys |= {("probe", m.a, m.b, m.c, m.m_eff, T, PROD_QBOX, PROD_NQ) for m in real for T in TEMPS}
    print(f"{len(keys)} unique probe solves + {len(prod_keys)} production-function solves "
          f"on {workers} workers", flush=True)
    if args.dry_run:
        return
    res = {}
    if args.solve_cache and Path(args.solve_cache).exists():
        import pickle
        res = pickle.load(open(args.solve_cache, "rb"))
        print(f"reused {len(res)} solves from {args.solve_cache}", flush=True)
    todo = sorted((keys | prod_keys) - set(res), key=lambda k: (k[0], k[5], k[1:]))
    n_reused = len((keys | prod_keys) & set(res))
    res.update(_run_jobs(todo, workers))
    if args.solve_cache:
        import pickle
        pickle.dump(res, open(args.solve_cache, "wb"))
    print(f"solves done in {time.time() - t_start:.0f}s", flush=True)

    def mode_res_for(s):
        return {(m.uid, T): res[("probe", *problems[(s.key, m.uid)], T, s.q_box, s.nq)]
                for m in real for T in TEMPS}

    # --- baseline: replica == _solve_scha, and both == ledger
    base_s = settings[0]
    base_res = mode_res_for(base_s)
    n_rep = n_ref_vs_cache = 0
    cached_calls = {}
    for m in real:
        for T in TEMPS:
            p = res[("prod", m.a, m.b, m.c, m.m_eff, T, PROD_QBOX, PROD_NQ)]
            rc = res[("probe", m.a, m.b, m.c, m.m_eff, T, PROD_QBOX, PROD_NQ)]
            r = base_res[(m.uid, T)]
            n_rep += (p["Q0"] == rc["Q0"] and p["stable"] == rc["stable_prod_thresh"] and p["eff"] == rc["eff"])
            n_ref_vs_cache += (p["Q0"] == r["Q0"] and p["stable"] == r["stable_prod_thresh"])
            cached_calls[(m.uid, T)] = not p["stable"]
    coef_rel = max(abs(x - y) / max(abs(y), 1e-300)
                   for m in real
                   for x, y in zip(_fit_double_well(m.Qs, m.dE), (m.a, m.b, m.c)) if x != y) \
        if n_cache_eq < len(real) else 0.0
    ref_mode_calls = mode_calls(modes, base_res, base_s)
    ref_units = unit_table(modes, base_res, base_s)
    led = sm.set_index(["system", "model", "temperature_K"])
    u = ref_units.set_index(["system", "model", "temperature_K"])
    unit_match = int((u["pred_new"] == led.loc[u.index, "pred_stable"].astype(bool)).sum())
    curv_diff = float(np.max(np.abs(u["min_eff_new"] - led.loc[u.index, "min_eff_freq_thz"].astype(float))))
    q0_diff = float(np.max(np.abs(u["decide_Q0"] - led.loc[u.index, "ft_order_param_Q_ang"].astype(float))))
    mode_match = mode_match_cached = n_mode = 0
    unit_match_cached = 0
    by_unit = {}
    for m in modes:
        by_unit.setdefault((m.system, m.model), []).append(m)
    for (system, model), ms in by_unit.items():
        for T in TEMPS:
            flags = led.loc[(system, model, T), "ft_modes_stable"].split(";")
            assert len(flags) == len(ms), (system, model, T)
            for m, f in zip(ms, flags):
                if m.placeholder:
                    continue
                n_mode += 1
                mode_match += (f == "0") == ref_mode_calls[(m.uid, T)]
                mode_match_cached += (f == "0") == cached_calls[(m.uid, T)]
            unit_cached = not any(cached_calls[(m.uid, T)] for m in ms if not m.placeholder)
            unit_match_cached += unit_cached == bool(led.loc[(system, model, T), "pred_stable"])
    baseline = {
        "parameterised_fit_equals_production": _kn(n_copy_eq, len(real)),
        "production_refit_equals_cached_coefficients_bitwise": _kn(n_cache_eq, len(real)),
        "refit_vs_cached_max_relative_coefficient_difference": coef_rel,
        "replica_solver_equals__solve_scha_on_cached_coefficients": _kn(n_rep, len(real) * len(TEMPS)),
        "refit_path_equals__solve_scha_cached_path_Q0_and_call": _kn(n_ref_vs_cache, len(real) * len(TEMPS)),
        "cached_coefficient_path_unit_calls_match": _kn(unit_match_cached, len(u)),
        "cached_coefficient_path_mode_T_calls_match": _kn(mode_match_cached, n_mode),
        "unit_calls_match": _kn(unit_match, len(u)),
        "mode_T_calls_match": _kn(mode_match, n_mode),
        "max_abs_diff_min_eff_freq_thz_vs_ledger": curv_diff,
        "max_abs_diff_unit_Q0_vs_ledger_A": q0_diff,
    }
    print("baseline:", json.dumps(baseline), flush=True)
    assert (unit_match == len(u) and mode_match == n_mode and n_rep == len(real) * len(TEMPS)
            and unit_match_cached == len(u) and mode_match_cached == n_mode), \
        "baseline does not reproduce the ledger; refusing to report sensitivities"

    # --- every setting
    out_rows = []
    for s in settings:
        mr = base_res if s.key == "production" else mode_res_for(s)
        r = setting_metrics(modes, mr, s, ref_mode_calls, ref_units, df)
        r["srtio3_modes"] = srtio3_mode_detail(modes, mr, s)
        changed = sum(1 for m in real if problems[(s.key, m.uid)] != problems[("production", m.uid)])
        r["modes_with_changed_solver_input"] = _kn(changed, len(real))
        r.pop("_units"); r.pop("_mode_calls")
        out_rows.append(r)
        print(f"  {s.key:22s} modes {r['mode_flips']['all']['fmt']:>9s}  units "
              f"{r['unit_flips']['all']['fmt']:>7s}  FE {r['all_models']['fe_recall']['k']}/"
              f"{r['all_models']['fe_recall']['n']}  gate {r['all_models']['srtio3_gate_passing']['fmt']}"
              f"  acc {r['all_models']['accuracy']['k']}/{r['all_models']['accuracy']['n']}", flush=True)

    ev = [base_res[(m.uid, T)] for m in real for T in TEMPS]
    fb_k = sum(r["n_fallback"] for r in ev); fb_n = sum(r["n_eval"] for r in ev)
    evc = [res[("probe", m.a, m.b, m.c, m.m_eff, T, PROD_QBOX, PROD_NQ)] for m in real for T in TEMPS]
    fbc_k = sum(r["n_fallback"] for r in evc)
    nz = sorted(r["Q0"] for r in ev if r["Q0"] > 0)
    diagnostics = {
        "brentq_fallback_cached_coefficient_path": _kn(fbc_k, sum(r["n_eval"] for r in evc)) | {
            "share": fbc_k / sum(r["n_eval"] for r in evc),
            "at_argmin": _kn(sum(r["fallback_at_argmin"] for r in evc), len(evc)),
            "note": "the as-run production path (cached a, b, c); the sensitivity rows use the refit "
                    "path, whose coefficients differ in the last bits"},
        "mode_Q0_distribution": {
            "exactly_zero": _kn(sum(1 for r in ev if r["Q0"] == 0.0), len(ev)),
            "smallest_nonzero_A": nz[0] if nz else None,
            "in_(0,_0.015]_A": int(sum(1 for q in nz if q <= 0.015 + 1e-12)),
            "note": "threshold sweep 0.5/1.5/3 steps (0.0025/0.0075/0.015 A) can only move the "
                    "evaluations with 0 < Q0 <= 0.015 A"},
        "kept_set_identity": kept_set_identity(modes),
        "unbracketed": unbracketed(modes),
        "fitted_without_sampled_well": fitted_without_sampled_well(modes, ref_mode_calls, ref_units),
        "cell_sizes": cell_sizes(modes),
        "brentq_fallback": _kn(fb_k, fb_n) | {
            "share": fb_k / fb_n,
            "at_argmin": _kn(sum(r["fallback_at_argmin"] for r in ev), len(ev)),
            "unbound_centroids_F_inf": int(sum(r["n_inf"] for r in ev))},
        "unit_Q0_beyond_sampled_range_0p45A": _kn(int((ref_units["decide_Q0"] > PROD_QMAX + 1e-9).sum()),
                                                  len(ref_units)),
        "ledger_unit_Q0_at_box_edge": _kn(int((sm["ft_order_param_Q_ang"] >= PROD_QBOX - 1e-9).sum()), len(sm)),
        "le300K_scored_stable_units_with_modes": _scored_stable_with_modes(sm, real),
        "truncation_levels_identical": "Q<=0.30 and Q<=0.35 keep the same 8 sampled points "
                                       "(0.2722 A in, 0.3556 A out) and are identical by construction",
    }
    results = {
        "meta": {
            "script": "scripts/screen_sensitivity.py",
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "inputs": {"ledger": "results/ledger.parquet via analysis.load_canonical()",
                       "cache": f"results/cache/{CACHE_PREFIX}*.json ({len(files)} files)"},
            "n_screened_modes": len(real), "n_placeholders": len(modes) - len(real),
            "n_units": int(len(sm)), "temperatures_K": list(TEMPS),
            "production_constants": {"window_multiplier": PROD_MULT, "window_floor_eV": PROD_FLOOR,
                                     "q_max_A": PROD_QMAX, "n_pts": PROD_NPTS, "q_box_A": PROD_QBOX,
                                     "nq": PROD_NQ, "threshold_steps": PROD_THRESH,
                                     "threshold_A": PROD_THRESH * PROD_QBOX / (PROD_NQ - 1)},
            "definitions": {
                "mode_flips": "condensation-call changes vs production over mode-T evaluations "
                              "(all = 4 T; le300K = 100 and 300 K)",
                "unit_flips": "pred_stable changes vs production over (system, model, T) units",
                "fe_recall": "analysis.displacive_recall softmode row (BaTiO3/KNbO3/PbTiO3, T<=300 K)",
                "accuracy": "analysis.low_t_false_stable(exclude_bcc=True): T<=300 K, KTaO3 and bcc "
                            "excluded, pooled over models; Wilson 95% interval",
                "srtio3_gate": "per model: condenses at 100 K and stable at 300, 600 and 900 K",
                "exORB": "the same quantity with ORB-v2 rows removed",
                "cell_normalisation": "(a,b,c,M) -> n(a,b,c,M) on the production fit; see module docstring"},
            "n_solves": len(keys | prod_keys), "n_solves_reused_from_solve_cache": n_reused,
            "runtime_s": None,
            "runtime_note": "a cold run (no --solve-cache) takes ~18 min on 6 workers"},
        "baseline": baseline,
        "settings": out_rows,
        "diagnostics": diagnostics,
        "scaling_checks": scaling_checks(modes, not args.no_mlip_check),
    }
    results["audit_comparison"] = audit_comparison(results)
    results["meta"]["runtime_s"] = round(time.time() - t_start, 1)
    OUT_JSON.write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    write_md(results, OUT_MD)
    print(f"wrote {OUT_JSON.relative_to(REPO)} and {OUT_MD.relative_to(REPO)} "
          f"in {results['meta']['runtime_s']}s", flush=True)
    for c in results["audit_comparison"]:
        print(f"  [{'ok ' if c['agrees'] else ('?? ' if c['agrees'] is None else 'DIFF')}] "
              f"{c['claim']}: audit {c['audit']} | computed {c['computed']}", flush=True)


if __name__ == "__main__":
    main()
