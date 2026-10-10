"""The converged-recipe SSCHA grid against the production SSCHA claims of Sec. 3.3.

The manuscript's SSCHA numbers come from the PRODUCTION ledger (2x2x2, 256 configurations, the
cumulative max_ka = 20; it never converged). ``scripts/sscha_seed_study.py --preset grid`` re-runs
the units those numbers rest on with the converged recipe (start A) and ``--preset grid
--summarize`` writes results/revision/sscha_converged_grid/summary.csv. This script asks, for each
production claim, what the converged recipe says on the units the grid covers:

  (i)   "SSCHA calls deep displacive wells stable against unstable labels in 57 units, and on the
        same MLIP energies the screen's free-energy comparison finds a lower displaced minimum in
        46 of them" -- Table S16 lower part, ``stats_hardening.criterion_blindness``: non-bcc
        (system, model, T) units with both a softmode and an sscha row, SSCHA pred_stable and
        gt_stable False; the screen's pred_stable is the free-energy comparison. Blow-ups are
        NOT excluded here.
  (ii)  ferroelectric-perovskite recall "16/30 for the screen vs 5/27 for SSCHA at T <= 300 K" --
        ``analysis.displacive_recall``: BaTiO3, KNbO3, PbTiO3 at T <= 300 K; recall of the
        unstable class; SSCHA blow-ups (|f| > 50 THz) leave the denominator and are counted
        separately.
  (iii) bcc screen-vs-SSCHA call agreement 31/45 (25/36 without ORB-v2) -- Table S16 upper part,
        ``stats_hardening.bcc_agreement``: Ti, Zr, Hf at 100/300/600 K where both ran; the screen's
        pred_stable against SSCHA's. Blow-ups are included.
  (iv)  the non-bcc SSCHA false-unstables by temperature and the SrTiO3 units above T_c -- Table
        S17, ``stats_hardening.sscha_high_t``: SSCHA calls unstable where the label is stable.
        Blow-ups are included in the counts and tallied alongside.

Every one of these is computed with the repo's own functions. The converged side is the same
function applied to a ledger-shaped frame whose SSCHA rows are replaced by the converged
Hessian minima (pred_stable = Hessian minimum >= imag_tol, the production rule); the screen rows,
the labels and the harmonic rows are the ledger's. So the blow-up rule, the ORB-v2 split, the
unit sets and the denominators are the production ones by construction, and the production
numbers recomputed here are checked against results/stats_hardening.json.

Four frames are compared, each with and without ORB-v2:

  production_full     the canonical ledger (the manuscript's numbers);
  production_matched  the production rows on exactly the units the grid evaluated;
  converged_matched   the converged values on those same units;
  converged_all       the converged values on every unit evaluated, including units production
                      never returned (ORB-v2 on PbTiO3 failed there).

Two variants of "evaluated": ``converged_only`` (the primary: status ok and the library's own
stopping test passed) and ``all_ok`` (also the runs the wall cap or another limit ended).
Two SrTiO3/MACE-MP-0 units (600 and 900 K, 2x2x2) are read from the C1c converged-mode directory
(results/revision/sscha_converged/); they are the grid's only coverage above 300 K.

    python scripts/grid_compare.py [SUMMARY_CSV] [--c1c-dir DIR] [--out PATH]

Writes results/revision/grid_compare.json (refuses that default path for a --dry-run grid) and
prints a markdown summary. Read-only on every input.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np                              # noqa: E402
import pandas as pd                             # noqa: E402

from mlip_dynstab import analysis as A          # noqa: E402
from mlip_dynstab import stats as S             # noqa: E402
import stats_hardening as SH                    # noqa: E402

GRID_DIR = REPO / "results" / "revision" / "sscha_converged_grid"
SUMMARY = GRID_DIR / "summary.csv"
C1C_DIR = REPO / "results" / "revision" / "sscha_converged"
OUT = REPO / "results" / "revision" / "grid_compare.json"
LEDGER = REPO / "results" / "ledger.parquet"
STATS = REPO / "results" / "stats_hardening.json"
SEED_STUDY = REPO / "scripts" / "sscha_seed_study.py"

KEYS = ["system", "model", "temperature_K"]
TEMPS = SH.TEMPS
MODEL_SETS = SH.MODEL_SETS
FE = SH.FE
BCC = SH.BCC_METALS
BLOWUP_THZ = SH.BLOWUP_THZ
# The two converged units that sit outside the grid (C1c converged mode, start A, 2x2x2): the
# high-T false-unstable runaway of Table S17. They are the grid's only coverage above 300 K.
EXTRA_UNITS = (("srtio3_cubic", "mace_mp0", 600.0), ("srtio3_cubic", "mace_mp0", 900.0))
VARIANTS = {
    "converged_only": "status ok and the unit's own stopping test passed (the primary set)",
    "all_ok": "status ok, whatever ended the relaxation (also the wall-cap runs)",
}
FRAMES = (("production_full", "Production, full set"),
          ("production_matched", "Production, same units as the grid"),
          ("converged_matched", "Converged, same units"),
          ("converged_all", "Converged, all units evaluated"))
MODEL_TAGS = (("all_models", "all five"), ("excl_orb_v2", "without ORB-v2"))
STATE_WORDS = {"ok_not_converged": "finished without meeting the stopping test",
               "ok_no_value": "finished without a value", "failed": "failed",
               "running": "still running", "todo": "not run", "other_recipe": "run with another recipe",
               "blowup": "converged to a blow-up", "not_in_grid": "not in the grid",
               "still_false_stable": "still called stable", "now_unstable": "now called unstable"}
BOOL_COLS = ("recipe_matches", "converged", "stable_call", "blowup", "gt_stable", "label_scored",
             "call_matches_label", "screen_stable_call", "call_matches_screen",
             "prod_sc222_stable_call", "prod_sc222_blowup", "same_cell_as_prod",
             "call_changed_vs_prod")
NUM_COLS = ("T_K", "n_atoms", "n_populations", "final_aux_min_thz", "hessian_min_thz",
            "boot_sd_thz", "imag_tol_thz", "prod_sc222_hessian_min_thz", "fresh_R_over_expected",
            "kl_ratio_at_gradient_end", "wall_min", "n_boot")


# ------------------------------------------------------------------------ utilities ----

def rel(p) -> str:
    p = Path(p).resolve()
    try:
        return p.relative_to(REPO).as_posix()
    except ValueError:
        return p.as_posix()


def sha256(p, text: bool = False) -> str:
    """text=True hashes the file as git stores it (CRLF -> LF), so the digest does not depend on
    a Windows autocrlf checkout."""
    data = Path(p).read_bytes()
    return hashlib.sha256(data.replace(b"\r\n", b"\n") if text else data).hexdigest()


def tobool(v):
    """True/False/None from a CSV cell ('True', 'False', '' or NaN)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    s = str(v).strip().lower()
    return True if s == "true" else False if s == "false" else None


def istrue(v) -> bool:
    """True for a real True (Python or numpy), False for False, None and NaN."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return False
    return bool(v)


def clean(v):
    """None for None and NaN, else the value (a CSV's empty cells arrive as NaN)."""
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else v


def kn(k, n) -> dict:
    """k of n with a Wilson interval; an empty denominator gives nulls (strict JSON)."""
    k, n = int(k), int(n)
    if n == 0:
        return {"k": k, "n": 0, "p": None, "lo": None, "hi": None, "fmt": "0/0", "short": "0/0"}
    r = S.rate_ci(k, n)
    return r.as_dict() | {"fmt": S.fmt_rate(r), "short": f"{k}/{n}"}


def with_short(d: dict) -> dict:
    """A stats_hardening rate dict (k, n, Wilson interval, fmt) plus a 'k/n' string."""
    return d | {"short": f"{d['k']}/{d['n']}"}


def drop_models(df: pd.DataFrame, excl) -> pd.DataFrame:
    return df[~df["model"].isin(excl)] if excl else df


def keyset(df: pd.DataFrame) -> set:
    return set(zip(df["system"], df["model"], df["temperature_K"].astype(float)))


def sel(df: pd.DataFrame, keys: set) -> pd.DataFrame:
    if df.empty:
        return df
    mask = [k in keys for k in zip(df["system"], df["model"], df["temperature_K"].astype(float))]
    return df[np.asarray(mask, dtype=bool)]


def _py(o):
    """json.dumps default: numpy scalars and arrays, NaN as null."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def _nan_to_none(x):
    if isinstance(x, float) and not np.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: _nan_to_none(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_nan_to_none(v) for v in x]
    return x


def load_seed_study():
    """scripts/sscha_seed_study.py as a module (definitions only; its main is guarded). Used for
    the unit-JSON digest, the recipe and the CSV column list, so none of them is re-implemented."""
    spec = importlib.util.spec_from_file_location("sscha_seed_study", SEED_STUDY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sscha_seed_study"] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- inputs ----

def load_summary(path: Path, sss) -> pd.DataFrame:
    """summary.csv with typed columns; start A rows only (the grid is start A; a B row would be
    a second value for the same unit). Refuses a file that lacks any GRID_CSV_COLUMNS column."""
    df = pd.read_csv(path)
    missing = [c for c in sss.GRID_CSV_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"{rel(path)} lacks the columns {missing}; it was not written by "
                         "scripts/sscha_seed_study.py --preset grid --summarize")
    for c in BOOL_COLS:
        df[c] = pd.Series([tobool(v) for v in df[c]], index=df.index, dtype=object)
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("stop_reason", "status", "error", "prod_status"):
        df[c] = df[c].astype(object).where(df[c].notna(), None)
    return df


def first_unit_json(directory: Path, tags) -> tuple:
    """(path, doc) of the first unit JSON beside the summary that exists and is a dict."""
    for tag in tags:
        p = directory / f"{tag}_startA.json"
        if p.exists():
            try:
                return p, json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
    return None, None


def load_extras(c1c_dir: Path, sss, dry: bool, expected: dict) -> tuple:
    """The converged-mode units outside the grid (EXTRA_UNITS), start A. Returns (rows, info):
    rows are pool rows for the ones usable here, info says what was found for every one."""
    rows, info = [], []
    for system, model, T in EXTRA_UNITS:
        tag = sss.unit_tag(system, model, T, (2, 2, 2))
        p = c1c_dir / f"{tag}_startA.json"
        e = {"unit_tag": tag, "system": system, "model": model, "T_K": T, "path": rel(p),
             "used": False}
        if not p.exists():
            info.append(e | {"reason": "no JSON yet"})
            continue
        doc = json.loads(p.read_text(encoding="utf-8"))
        unit = {"system": system, "model": model, "T": T, "supercell": [2, 2, 2]}
        if doc.get("schema") != sss.CONV_SCHEMA:
            info.append(e | {"reason": f"schema {doc.get('schema')}"})
        elif bool(doc.get("dry_run")) != bool(dry):
            info.append(e | {"reason": f"dry_run={doc.get('dry_run')} against a "
                                       f"{'dry-run' if dry else 'real'} grid; not mixed"})
        elif doc.get("unit") != unit:
            info.append(e | {"reason": f"the JSON holds unit {doc.get('unit')}"})
        elif doc.get("recipe") != expected:
            info.append(e | {"reason": "another recipe than the grid's"})
        else:
            dg = sss._start_digest(doc)
            h = dg["hessian_min_thz"] if dg["status"] == "ok" else None
            rows.append({"unit_tag": tag, "system": system, "model": model, "T_K": T,
                         "supercell": "2x2x2", "family": "cubic-perovskite", "status": dg["status"],
                         "converged": bool(dg["converged"]) if dg["converged"] is not None else None,
                         "stop_reason": dg["stop_reason"],
                         "hessian_min_thz": np.nan if h is None else float(h),
                         "imag_tol_thz": float(doc["recipe"]["imag_tol_thz"]),
                         "source": "c1c"})
            info.append(e | {"used": True, "status": dg["status"], "converged": dg["converged"],
                             "stop_reason": dg["stop_reason"], "hessian_min_thz": h,
                             "sha256": sha256(p)})
    return rows, info


def c1c_overlap(rows: pd.DataFrame, c1c_dir: Path, sss, dry: bool, expected: dict) -> list:
    """Grid units that also have a C1c converged-mode start-A JSON of the same recipe: the same
    computation twice (other seed stream is not involved: both use seed 0), a reproducibility
    check of the converged value."""
    out = []
    for r in rows.itertuples():
        p = c1c_dir / f"{r.unit_tag}_startA.json"
        if r.status != "ok" or not p.exists():
            continue
        doc = json.loads(p.read_text(encoding="utf-8"))
        if (doc.get("schema") != sss.CONV_SCHEMA or bool(doc.get("dry_run")) != bool(dry)
                or doc.get("recipe") != expected):
            continue
        dg = sss._start_digest(doc)
        if dg["status"] != "ok":
            continue
        itol = float(r.imag_tol_thz)
        out.append({"unit_tag": r.unit_tag, "grid_hessian_thz": float(r.hessian_min_thz),
                    "c1c_hessian_thz": float(dg["hessian_min_thz"]),
                    "abs_diff_thz": abs(float(r.hessian_min_thz) - float(dg["hessian_min_thz"])),
                    "grid_converged": r.converged, "c1c_converged": bool(dg["converged"]),
                    "same_call": bool((r.hessian_min_thz >= itol) == (dg["hessian_min_thz"] >= itol))})
    return out


# ----------------------------------------------------------------------- the frames ----

def label_tables(ledger: pd.DataFrame) -> tuple:
    """The ledger's label per (system, T) and transition temperature per system."""
    d = ledger[ledger["method"].isin(["softmode", "sscha"])]
    gt = d.groupby(["system", "temperature_K"])["gt_stable"].first()
    tc = d.groupby("system")["transition_T_K"].first()
    return gt, tc


def conv_sscha_rows(pool: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    """Ledger-shaped SSCHA rows carrying the converged Hessian minima: min_eff_freq_thz is the
    Hessian minimum and pred_stable the production rule (stable iff it is >= imag_tol_thz)."""
    if pool.empty:
        return pd.DataFrame(columns=["method", "system", "model", "temperature_K",
                                     "min_eff_freq_thz", "pred_stable", "gt_stable",
                                     "transition_T_K"])
    gt, tc = label_tables(ledger)
    T = pool["T_K"].astype(float)
    h = pool["hessian_min_thz"].astype(float)
    return pd.DataFrame({
        "method": "sscha", "system": pool["system"].to_numpy(), "model": pool["model"].to_numpy(),
        "temperature_K": T.to_numpy(), "min_eff_freq_thz": h.to_numpy(),
        "pred_stable": (h >= pool["imag_tol_thz"].astype(float)).to_numpy(),
        "gt_stable": [bool(gt.get((s, t))) for s, t in zip(pool["system"], T)],
        "transition_T_K": [float(tc.get(s, np.nan)) for s in pool["system"]],
        "imag_tol_thz": pool["imag_tol_thz"].astype(float).to_numpy()})


def evaluated_mask(pool: pd.DataFrame, variant: str) -> pd.Series:
    ok = (pool["status"] == "ok") & pool["hessian_min_thz"].notna()
    if variant == "converged_only":
        ok &= pool["converged"].map(istrue)
    return ok


def build_frames(ledger: pd.DataFrame, pool_eval: pd.DataFrame) -> dict:
    """The three frames that depend on what the grid evaluated (production_full is the ledger)."""
    sm = ledger[ledger["method"] == "softmode"]
    hm = ledger[ledger["method"] == "harmonic"]
    ps = ledger[ledger["method"] == "sscha"]
    ev = set(zip(pool_eval["system"], pool_eval["model"], pool_eval["T_K"].astype(float)))
    matched = ev & keyset(ps)
    pseudo = conv_sscha_rows(pool_eval, ledger)
    pseudo_m = sel(pseudo, matched)

    def frame(*parts) -> pd.DataFrame:
        f = pd.concat([p for p in parts if len(p)], ignore_index=True)
        for c in ("min_eff_freq_thz", "temperature_K"):     # object when a part was empty
            f[c] = pd.to_numeric(f[c], errors="coerce")
        return f

    return {
        "production_matched": frame(sel(sm, matched), hm, sel(ps, matched)),
        "converged_matched": frame(sel(sm, matched), hm, pseudo_m),
        "converged_all": frame(sel(sm, ev), hm, pseudo),
    }


# ------------------------------------------------------------------ the four claims ----

def _paired_nonbcc(frame: pd.DataFrame) -> pd.DataFrame:
    sm = frame[frame["method"] == "softmode"][KEYS + ["pred_stable", "gt_stable"]]
    ss = frame[frame["method"] == "sscha"][KEYS + ["pred_stable", "min_eff_freq_thz"]]
    m = sm.merge(ss, on=KEYS, suffixes=("_scr", "_ss"))
    return m[~m["system"].str.contains("bcc")]


def block_i(frame: pd.DataFrame) -> dict:
    """Claim (i). stats_hardening.criterion_blindness for the two model sets, plus the
    denominators it does not carry (all paired non-bcc units, those labelled unstable)."""
    cb = SH.criterion_blindness(frame)
    pair = _paired_nonbcc(frame)
    out = {}
    for tag, excl in MODEL_SETS:
        e, m = cb[tag], drop_models(pair, excl)
        unst = m[~m["gt_stable"].astype(bool)]
        fs = unst[unst["pred_stable_ss"].astype(bool)]
        if len(fs) != e["n_sscha_false_stable_nonbcc"]:
            raise SystemExit(f"claim (i): {len(fs)} false-stables here, "
                             f"{e['n_sscha_false_stable_nonbcc']} from criterion_blindness")
        out[tag] = {
            "n_paired_nonbcc": int(len(m)), "n_label_unstable": int(len(unst)),
            "n_sscha_false_stable": int(len(fs)),
            "sscha_false_stable_of_label_unstable": kn(len(fs), len(unst)),
            "screen_call_unstable_on_those": with_short(e["screen_call_unstable"]),
            "by_system": e["by_system"],
            "by_T": {str(int(t)): int((fs["temperature_K"] == t).sum()) for t in TEMPS},
            "n_blowup_among_false_stable": int((fs["min_eff_freq_thz"].abs() > BLOWUP_THZ).sum()),
            "paired_T_le_300": e["paired_nonbcc_T_le_300"],
        }
    return out


def block_ii(frame: pd.DataFrame) -> dict:
    """Claim (ii). analysis.displacive_recall on each model set (the call stats_hardening.orb_split
    makes): the recall of the unstable class on the ferroelectric perovskites at T <= 300 K."""
    out = {}
    for tag, excl in MODEL_SETS:
        rec = A.displacive_recall(drop_models(frame, excl))
        res = {}
        for meth in ("softmode", "sscha"):
            r = rec[rec["method"] == meth] if len(rec) else rec
            if len(r):
                r = r.iloc[0]
                n, c, b = int(r["n_valid"]), int(r["correct_unstable"]), int(r["n_numerical_blowup"])
            else:
                n = c = b = 0
            res[meth] = {"n_valid": n, "correct_unstable": c, "n_numerical_blowup": b,
                         "recall": kn(c, n)}
        out[tag] = res
    return out


def block_iii(frame: pd.DataFrame) -> dict:
    """Claim (iii). stats_hardening.bcc_agreement: call agreement on Ti/Zr/Hf."""
    bcc = frame[frame["system"].isin(BCC)]
    sm = bcc[bcc["method"] == "softmode"][KEYS]
    ss = bcc[bcc["method"] == "sscha"][KEYS]
    empty = {"n_paired": 0, "call_agreement": kn(0, 0), "curvature_sign_agreement": kn(0, 0),
             "non_trivial_call": kn(0, 0), "trivial_call": kn(0, 0), "per_model_call": {}}
    if sm.merge(ss, on=KEYS).empty:
        return {tag: dict(empty) for tag, _ in MODEL_SETS}
    with warnings.catch_warnings():       # the frequency correlations it also computes are unused
        warnings.simplefilter("ignore", RuntimeWarning)
        ba = SH.bcc_agreement(frame)
    out = {}
    for tag, _ in MODEL_SETS:
        e = ba[tag]
        out[tag] = {
            "n_paired": int(e["n_paired"]), "call_agreement": with_short(e["call_agreement"]),
            "curvature_sign_agreement": with_short(e["curvature_sign_agreement"]),
            "non_trivial_call": with_short(e["trivial_split"]["non_trivial"]["call"]),
            "trivial_call": with_short(e["trivial_split"]["trivial"]["call"]),
            "n_trivial_pairs": len(e["trivial_split"]["trivial_pairs"]),
            "per_model_call": {m: with_short(v["call"]) for m, v in e["per_model"].items()}}
    return out


def _false_unstable_by_T(frame: pd.DataFrame, excl) -> dict:
    """The false_unstable_by_T rule of stats_hardening.sscha_high_t, for a frame with no stable-
    labelled SrTiO3 SSCHA row (where the repo function cannot run). Checked against the repo
    function on the production frame in check_production."""
    ss = drop_models(frame[(frame["method"] == "sscha") & ~frame["system"].str.contains("bcc")], excl)
    by_t = {}
    for t in TEMPS:
        x = ss[ss["temperature_K"] == t]
        gs = x[x["gt_stable"].astype(bool)]
        fu = gs[~gs["pred_stable"].astype(bool)]
        by_t[str(int(t))] = kn(len(fu), len(gs)) | {
            "n_units_at_T": int(len(x)),
            "n_blowup_among_false_unstable": int((fu["min_eff_freq_thz"].abs() > BLOWUP_THZ).sum()),
            "by_system": {s: int(k) for s, k in fu["system"].value_counts().sort_index().items()}}
    return by_t


def block_iv(frame: pd.DataFrame) -> dict:
    """Claim (iv). stats_hardening.sscha_high_t: the non-bcc false-unstables by temperature and the
    SrTiO3 units above T_c. Falls back to the same rule without the SrTiO3 part when the frame has
    no stable-labelled SrTiO3 unit (the repo function indexes its first row)."""
    try:
        sh, note = SH.sscha_high_t(frame), None
    except (IndexError, KeyError, ValueError) as exc:
        sh, note = None, f"stats_hardening.sscha_high_t not applicable ({type(exc).__name__}: {exc})"
    out = {}
    for tag, excl in MODEL_SETS:
        if sh is None:
            out[tag] = {"false_unstable_by_T": _false_unstable_by_T(frame, excl),
                        "srtio3_above_tc": None, "note": note}
            continue
        e = sh[tag]
        so = e["srtio3_above_tc"]
        out[tag] = {
            "false_unstable_by_T": {t: with_short(v) for t, v in e["false_unstable_by_T"].items()},
            "srtio3_above_tc": {
                "transition_T_K": so["transition_T_K"], "n_returned": so["n_returned"],
                "false_unstable": with_short(so["false_unstable"]),
                "sscha_freq_range_thz": so["sscha_freq_range_thz"],
                "units": [{"model": u["model"], "T": u["T"],
                           "sscha_min_eff_freq_thz": u["sscha_min_eff_freq_thz"],
                           "false_unstable": u["false_unstable"], "blow_up": u["blow_up"]}
                          for u in so["units"]]}}
    return out


def block_v(frame: pd.DataFrame) -> dict:
    """Claim (v). stats_hardening.screen_vs_sscha_paired on the three displacive system sets."""
    out = {}
    for tag, excl in MODEL_SETS:
        out[tag] = {label: SH.screen_vs_sscha_paired(frame, systems, exclude_models=tuple(excl))
                    for label, systems in (("fe_oxide", FE), ("fluorite", SH.FLUORITE),
                                           ("displacive_combined", FE + SH.FLUORITE))}
    return out


def analyse(frame: pd.DataFrame) -> dict:
    """The four claims on one frame. A claim the frame cannot answer is recorded with its error,
    never dropped silently."""
    out = {}
    for key, fn in (("i", block_i), ("ii", block_ii), ("iii", block_iii), ("iv", block_iv),
                    ("v", block_v)):
        try:
            out[key] = fn(frame)
        except (KeyError, IndexError, ValueError, AssertionError) as exc:
            out[key] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


# -------------------------------------------------------------------------- coverage ----

def classify(r, variant: str) -> str:
    """One pool row against a variant: evaluated | ok_not_converged | failed | running | todo |
    other_recipe | ok_no_value."""
    if r is None:
        return "not_in_grid"
    if r.status != "ok":
        return str(r.status)
    if not np.isfinite(r.hessian_min_thz):
        return "ok_no_value"
    if variant == "converged_only" and not istrue(r.converged):
        return "ok_not_converged"
    return "evaluated"


def coverage_for(keys, pool: pd.DataFrame, variant: str, what: str) -> dict:
    """Which of the production units ``keys`` the grid planned and evaluated."""
    idx = {(r.system, r.model, float(r.T_K)): r for r in pool.itertuples()}
    keys = sorted(keys)
    states = [classify(idx.get(k), variant) for k in keys]
    by_state: dict = {}
    for s in states:
        by_state[s] = by_state.get(s, 0) + 1
    out_grid = [k for k, s in zip(keys, states) if s == "not_in_grid"]
    behind = [{"system": k[0], "model": k[1], "T": k[2], "state": s}
              for k, s in zip(keys, states) if s not in ("evaluated", "not_in_grid")]
    return {
        "what": what, "n": len(keys), "in_grid_plan": len(keys) - len(out_grid),
        "evaluated": by_state.get("evaluated", 0), "by_state": by_state,
        "not_in_grid_by_T": {str(int(t)): sum(1 for k in out_grid if k[2] == t)
                             for t in sorted({k[2] for k in out_grid})},
        "not_in_grid_by_system": {s: sum(1 for k in out_grid if k[0] == s)
                                  for s in sorted({k[0] for k in out_grid})},
        "planned_not_evaluated": behind}


def production_sets(ledger: pd.DataFrame) -> dict:
    """The unit sets the four production claims are about, as (system, model, T) keys."""
    sm = ledger[ledger["method"] == "softmode"]
    ss = ledger[ledger["method"] == "sscha"]
    pair = _paired_nonbcc(ledger)
    fs = pair[pair["pred_stable_ss"].astype(bool) & ~pair["gt_stable"].astype(bool)]
    bcc_pair = sm[sm["system"].isin(BCC)][KEYS].merge(ss[KEYS], on=KEYS)
    nb = ss[~ss["system"].str.contains("bcc")]
    return {
        "i": ("the SSCHA false-stables of Table S16 (non-bcc, label unstable, SSCHA stable)",
              {tuple((r.system, r.model, float(r.temperature_K))) for r in fs.itertuples()}),
        "ii": ("the ferroelectric-perovskite screen units at T <= 300 K (the SSCHA rows are a "
               "subset)", keyset(sm[sm["system"].isin(FE) & (sm["temperature_K"] <= 300.0)])),
        "iii": ("the Ti/Zr/Hf units with both a screen and an SSCHA row", keyset(bcc_pair)),
        "iv": ("the non-bcc SSCHA units whose label is stable (the false-unstable denominators)",
               keyset(nb[nb["gt_stable"].astype(bool)])),
    }


def fs_table(ledger: pd.DataFrame, pool: pd.DataFrame, variant: str) -> tuple:
    """The 57 units one by one against the grid, and their outcome counts for one variant."""
    pair = _paired_nonbcc(ledger)
    fs = pair[pair["pred_stable_ss"].astype(bool) & ~pair["gt_stable"].astype(bool)]
    idx = {(r.system, r.model, float(r.T_K)): r for r in pool.itertuples()}
    rows, counts = [], {}
    scr_unst = 0
    for r in fs.sort_values(KEYS).itertuples():
        g = idx.get((r.system, r.model, float(r.temperature_K)))
        state = classify(g, variant)
        h = None if g is None or not np.isfinite(g.hessian_min_thz) else float(g.hessian_min_thz)
        call = None if h is None else bool(h >= float(g.imag_tol_thz))
        if state == "evaluated":
            outcome = ("blowup" if abs(h) > BLOWUP_THZ else
                       "still_false_stable" if call else "now_unstable")
        else:
            outcome = state
        counts[outcome] = counts.get(outcome, 0) + 1
        scr_unst += int(outcome == "still_false_stable" and not bool(r.pred_stable_scr))
        rows.append({"system": r.system, "model": r.model, "T": float(r.temperature_K),
                     "production_sscha_thz": float(r.min_eff_freq_thz),
                     "screen_call_stable": bool(r.pred_stable_scr),
                     "grid_state": state,
                     "converged_hessian_thz": h, "converged_call_stable": call,
                     "converged": None if g is None else clean(g.converged),
                     "stop_reason": None if g is None else clean(g.stop_reason),
                     "outcome": outcome})
    return rows, {"n": len(rows), "outcomes": counts,
                  "still_false_stable_screen_unstable": kn(scr_unst, counts.get("still_false_stable", 0))}


# ------------------------------------------------------------------- run accounting ----

def run_status_block(rows: pd.DataFrame, sss) -> dict:
    """What the grid did: statuses, convergence, stop reasons, timeouts, failures, blow-ups."""
    n = len(rows)
    ok = rows[rows["status"] == "ok"]
    conv = ok[ok["converged"].map(istrue)]
    cap = ok[ok["stop_reason"] == "wall_cap"]
    blow = ok[ok["hessian_min_thz"].abs() > BLOWUP_THZ]
    unfinished = rows[~rows["status"].isin(["ok", "failed", "other_recipe"])]

    def per(col):
        out = {}
        for k, g in rows.groupby(col):
            o = g[g["status"] == "ok"]
            out[str(k)] = {"planned": int(len(g)), "ok": int(len(o)),
                           "converged": int(o["converged"].map(istrue).sum()),
                           "wall_cap": int((o["stop_reason"] == "wall_cap").sum()),
                           "failed": int((g["status"] == "failed").sum()),
                           "not_finished": int(g["status"].isin(["todo", "running"]).sum()),
                           "blowups": int((o["hessian_min_thz"].abs() > BLOWUP_THZ).sum())}
        return out

    return {
        "n_planned_units": n, "by_status": {k: int(v) for k, v in rows["status"].value_counts().items()},
        "n_ok": int(len(ok)), "n_converged": int(len(conv)),
        "converged_fraction_of_ok": (len(conv) / len(ok)) if len(ok) else None,
        "converged_fraction_of_planned": (len(conv) / n) if n else None,
        "stop_reasons_of_ok": {str(k): int(v) for k, v in
                               ok["stop_reason"].fillna("none").value_counts().items()},
        "wall_cap_timeouts": [{"unit_tag": r.unit_tag, "hessian_thz": float(r.hessian_min_thz),
                               "n_populations": None if not np.isfinite(r.n_populations)
                               else int(r.n_populations)} for r in cap.itertuples()],
        "failed": [{"unit_tag": r.unit_tag, "error": (str(clean(r.error))[:300]
                                                      if clean(r.error) else None)}
                   for r in rows[rows["status"] == "failed"].itertuples()],
        "not_finished": [{"unit_tag": r.unit_tag, "status": r.status} for r in unfinished.itertuples()],
        "other_recipe": [r.unit_tag for r in rows[rows["status"] == "other_recipe"].itertuples()],
        "blowups": [{"unit_tag": r.unit_tag, "hessian_thz": float(r.hessian_min_thz),
                     "converged": r.converged, "production_thz":
                     None if not np.isfinite(r.prod_sc222_hessian_min_thz)
                     else float(r.prod_sc222_hessian_min_thz)} for r in blow.itertuples()],
        "blowup_threshold_thz": BLOWUP_THZ,
        "wall_min": ({"sum": float(ok["wall_min"].sum()), "median": float(ok["wall_min"].median()),
                      "max": float(ok["wall_min"].max())} if len(ok) else None),
        "by_model": per("model"), "by_family": per("family"),
        "failed_in_production": int((rows["prod_status"].fillna("").str.startswith("no ")).sum()),
    }


def changed_calls_block(rows: pd.DataFrame, extras: pd.DataFrame, ledger: pd.DataFrame) -> dict:
    """The units whose stability call differs from production, with old and new Hessian values.
    The grid rows carry their production value, label and screen call from summary.csv; the
    converged-mode units outside the grid (``extras``) get them from the ledger. Calls are the
    production rule: stable iff the Hessian minimum is >= imag_tol_thz."""
    recs = []
    for r in rows[(rows["status"] == "ok") & rows["hessian_min_thz"].notna()
                  & rows["prod_sc222_hessian_min_thz"].notna()].itertuples():
        recs.append(dict(
            unit_tag=r.unit_tag, system=r.system, model=r.model, T_K=float(r.T_K), cell=r.supercell,
            same_cell_as_prod=bool(r.same_cell_as_prod), converged=clean(r.converged),
            stop_reason=clean(r.stop_reason), hessian=float(r.hessian_min_thz),
            production_hessian=float(r.prod_sc222_hessian_min_thz), tol=float(r.imag_tol_thz),
            label_scored=bool(r.label_scored), label_stable=clean(r.gt_stable),
            screen_call_stable=clean(r.screen_stable_call), family=r.family, source="grid"))
    ss = ledger[ledger["method"] == "sscha"].set_index(KEYS)
    sm = ledger[ledger["method"] == "softmode"].set_index(KEYS)
    gt, _ = label_tables(ledger)
    for r in extras[extras["status"] == "ok"].itertuples() if len(extras) else []:
        k = (r.system, r.model, float(r.T_K))
        if k not in ss.index or not np.isfinite(r.hessian_min_thz):
            continue
        recs.append(dict(
            unit_tag=r.unit_tag, system=r.system, model=r.model, T_K=float(r.T_K), cell="2x2x2",
            same_cell_as_prod=True, converged=clean(r.converged), stop_reason=clean(r.stop_reason),
            hessian=float(r.hessian_min_thz), production_hessian=float(ss.loc[k, "min_eff_freq_thz"]),
            tol=float(r.imag_tol_thz), label_scored=True, label_stable=bool(gt.get((r.system, k[2]))),
            screen_call_stable=bool(sm.loc[k, "pred_stable"]) if k in sm.index else None,
            family=r.family, source="c1c"))
    out = []
    for u in sorted(recs, key=lambda u: (u["family"], u["system"], u["T_K"], u["model"])):
        ncall, pcall = u["hessian"] >= u["tol"], u["production_hessian"] >= u["tol"]
        u["blowup"], u["production_blowup"] = abs(u["hessian"]) > BLOWUP_THZ, \
            abs(u["production_hessian"]) > BLOWUP_THZ
        u["production_call_stable"], u["call_stable"] = pcall, ncall
        gtl = u["label_stable"] if u["label_scored"] else None
        u["production_matches_label"] = None if gtl is None else bool(pcall == gtl)
        u["converged_matches_label"] = None if gtl is None else bool(ncall == gtl)
        u["direction"] = ("S->U" if pcall and not ncall else "U->S") if pcall != ncall else None
        u["production_hessian_thz"], u["hessian_thz"] = u.pop("production_hessian"), u.pop("hessian")
        out.append(u)
    ok = out
    out = [u for u in out if u["direction"]]
    comp = [u for u in ok if istrue(u["converged"])]

    def tally(lst):
        return {"n": len(lst), "S_to_U": sum(1 for u in lst if u["direction"] == "S->U"),
                "U_to_S": sum(1 for u in lst if u["direction"] == "U->S")}

    good = [u for u in out if istrue(u["converged"]) and not u["blowup"]
            and not u["production_blowup"]]
    return {
        "n_compared_ok": len(ok), "n_compared_converged": len(comp),
        "all_changed": tally(out), "converged_no_blowup_changed": tally(good),
        "same_cell": tally([u for u in good if u["same_cell_as_prod"]]),
        "bcc_3x3x3_vs_2x2x2": tally([u for u in good if not u["same_cell_as_prod"]]),
        "nonbcc_corrected_vs_label": sum(1 for u in good if u["label_scored"] and
                                         u["production_matches_label"] is False and
                                         u["converged_matches_label"] is True),
        "nonbcc_worsened_vs_label": sum(1 for u in good if u["label_scored"] and
                                        u["production_matches_label"] is True and
                                        u["converged_matches_label"] is False),
        "units": out}


# ------------------------------------------------------------- production check ----

def check_production(prod: dict, ledger: pd.DataFrame, st: dict | None) -> dict:
    """The production numbers recomputed here against results/stats_hardening.json, and the local
    false-unstable rule against the repo's."""
    checks = []

    def add(name, got, want):
        checks.append({"check": name, "got": got, "committed": want, "ok": got == want})

    for tag, excl in MODEL_SETS:
        loc = _false_unstable_by_T(ledger, excl)
        add(f"local false-unstable rule == sscha_high_t [{tag}]",
            {t: (v["k"], v["n"]) for t, v in loc.items()},
            {t: (v["k"], v["n"]) for t, v in prod["iv"][tag]["false_unstable_by_T"].items()})
    if st is not None:
        for tag, _ in MODEL_SETS:
            cb, ba, sh = st["criterion_blindness"][tag], st["bcc_agreement"][tag], st["sscha_high_t"][tag]
            i, iii, iv = prod["i"][tag], prod["iii"][tag], prod["iv"][tag]
            add(f"(i) false-stables [{tag}]", i["n_sscha_false_stable"], cb["n_sscha_false_stable_nonbcc"])
            add(f"(i) screen unstable among them [{tag}]",
                (i["screen_call_unstable_on_those"]["k"], i["screen_call_unstable_on_those"]["n"]),
                (cb["screen_call_unstable"]["k"], cb["screen_call_unstable"]["n"]))
            add(f"(iii) bcc call agreement [{tag}]",
                (iii["call_agreement"]["k"], iii["call_agreement"]["n"]),
                (ba["call_agreement"]["k"], ba["call_agreement"]["n"]))
            add(f"(iv) false-unstable by T [{tag}]",
                {t: (v["k"], v["n"]) for t, v in iv["false_unstable_by_T"].items()},
                {t: (v["k"], v["n"]) for t, v in sh["false_unstable_by_T"].items()})
            rec = st["orb_split"][tag]["displacive_recall"]
            for meth in ("softmode", "sscha"):
                r = prod["ii"][tag][meth]
                add(f"(ii) recall {meth} [{tag}]", (r["correct_unstable"], r["n_valid"]),
                    (rec[meth]["k"], rec[meth]["n"]))
    return {"stats_hardening_json": rel(STATS) if st is not None else None,
            "n_checks": len(checks), "all_ok": all(c["ok"] for c in checks),
            "failed": [c for c in checks if not c["ok"]], "checks": checks}


def check_grid_vs_ledger(rows: pd.DataFrame, ledger: pd.DataFrame) -> dict:
    """summary.csv carries the label, the screen call and the production value it read from the
    ledger when it was written; confirm they are this ledger's."""
    ss = ledger[ledger["method"] == "sscha"].set_index(KEYS)
    sm = ledger[ledger["method"] == "softmode"].set_index(KEYS)
    gt, _ = label_tables(ledger)
    bad = {"production_value": [], "production_call": [], "screen_call": [], "label": []}
    for r in rows.itertuples():
        k = (r.system, r.model, float(r.T_K))
        if np.isfinite(r.prod_sc222_hessian_min_thz):
            if k not in ss.index or abs(ss.loc[k, "min_eff_freq_thz"] - r.prod_sc222_hessian_min_thz) > 1e-9:
                bad["production_value"].append(r.unit_tag)
            elif r.prod_sc222_stable_call is not None and bool(ss.loc[k, "pred_stable"]) != r.prod_sc222_stable_call:
                bad["production_call"].append(r.unit_tag)
        if r.screen_stable_call is not None and (k not in sm.index
                                                 or bool(sm.loc[k, "pred_stable"]) != r.screen_stable_call):
            bad["screen_call"].append(r.unit_tag)
        if r.gt_stable is not None and bool(gt.get((r.system, k[2]))) != r.gt_stable:
            bad["label"].append(r.unit_tag)
    return {"n_rows_checked": int(len(rows)), "mismatches": bad,
            "all_ok": not any(bad.values())}


# ----------------------------------------------------------------------- assembling ----

DEFINITIONS = {
    "i": ("Table S16 lower part (stats_hardening.criterion_blindness): non-bcc (system, model, T) "
          "units with both a softmode and an sscha row where SSCHA's pred_stable is True and the "
          "label gt_stable is False (57 in production, 50 without ORB-v2); the second count is how "
          "many of them the screen's pred_stable (its free-energy comparison) calls unstable "
          "(46 / 39). Blow-ups are included. The last row of the claim table is all paired "
          "non-bcc units at T <= 300 K with SSCHA's call equal to the screen's."),
    "ii": ("analysis.displacive_recall: BaTiO3, KNbO3, PbTiO3 at T <= 300 K, the share of units a "
           "method calls unstable (the label is unstable throughout); SSCHA rows with "
           "|f| > 50 THz are removed from the denominator and counted separately; the screen "
           "has 30 units and SSCHA 27 plus one blow-up in production (16/30 against 5/27)."),
    "iii": ("Table S16 upper part (stats_hardening.bcc_agreement): Ti, Zr, Hf at 100/300/600 K, the "
            "(system, model, T) units with both a screen and an SSCHA row (45; 36 without ORB-v2): "
            "the screen's pred_stable against SSCHA's (31/45; 25/36). Blow-ups are included. In "
            "the grid the bcc cell is 3x3x3, in production 2x2x2."),
    "iv": ("Table S17 (stats_hardening.sscha_high_t): non-bcc SSCHA rows, false-unstable = SSCHA "
           "calls unstable and the label is stable, per temperature; blow-ups are included and "
           "tallied. Production has 0/0, 5/5, 10/13 and 15/19 at 100/300/600/900 K. Since the "
           "2026-10-03 extension the grid covers the non-bcc systems at 100-900 K (2x2x2), so the "
           "two SrTiO3/MACE-MP-0 units of the converged-mode directory are replicates of grid "
           "units (c1c_overlap), not extra coverage."),
    "v": ("stats_hardening.screen_vs_sscha_paired: the paired screen-v-SSCHA contrast at "
          "T <= 300 K on the ferroelectric oxides (BaTiO3, KNbO3, PbTiO3), the fluorites and both "
          "together (units with an SSCHA value, |f| <= 50 THz); screen right and SSCHA wrong "
          "against the reverse, with stats.cluster_exact_paired over systems. Production: 33 v 3 "
          "on 47 units, p = 0.125."),
    "matched": ("production_matched uses the production rows of exactly the (system, model, T) "
                "units the grid evaluated (those with a production SSCHA row); converged_matched "
                "the converged values on the same units; converged_all every evaluated unit, "
                "including the ones production never returned."),
    "converged_value": ("A converged unit's SSCHA value is its free-energy Hessian minimum "
                        "(hessian_min_thz); its call is stable iff that is >= imag_tol_thz "
                        "(-0.1 THz, the production rule)."),
}


def compare(summary: Path, c1c_dir: Path, out_is_default: bool) -> dict:
    sss = load_seed_study()
    if not summary.exists():
        raise SystemExit(f"{rel(summary)} does not exist: run `python scripts/sscha_seed_study.py "
                         "--preset grid --summarize` on the grid's output first")
    rows_all = load_summary(summary, sss)
    rows = rows_all[rows_all["start"] == "A"].copy()
    if rows[["system", "model", "T_K"]].duplicated().any():
        raise SystemExit(f"{rel(summary)} has duplicate start-A rows")
    rows["T_K"] = rows["T_K"].astype(float)

    # the grid's real recipe and settings, from the first unit JSON beside the summary
    jp, jdoc = first_unit_json(summary.parent, rows["unit_tag"])
    expected = sss._clean(sss.converged_recipe(sss.parse_args(["--preset", sss.GRID_PRESET])))
    recipe = (jdoc or {}).get("recipe") or expected
    dry = bool((jdoc or {}).get("dry_run")) if jdoc else False
    if dry and out_is_default:
        raise SystemExit(f"{rel(summary)} was written by a --dry-run grid (mocked SSCHA values); "
                         f"pass --out to write it somewhere other than {rel(OUT)}")
    settings = ((jdoc or {}).get("run") or {}).get("settings") or {}

    ledger = A.canonical(pd.read_parquet(LEDGER))
    st = json.loads(STATS.read_text(encoding="utf-8")) if STATS.exists() else None

    extra_rows, extra_info = load_extras(c1c_dir, sss, dry, expected)
    # Once the grid itself covers an extra unit (the 600/900 K set added 2026-10-03), the grid row
    # is the one counted; the converged-mode JSON is then a replicate, reported by c1c_overlap.
    in_grid = set(rows["unit_tag"])
    extra_rows = [r for r in extra_rows if r["unit_tag"] not in in_grid]
    for e in extra_info:
        if e["unit_tag"] in in_grid and e.get("used"):
            e.update(used=False, reason="covered by the grid; counted there, replicate in overlap")
    pool = pd.concat([rows.assign(source="grid"), pd.DataFrame(extra_rows)], ignore_index=True)
    overlap = c1c_overlap(rows, c1c_dir, sss, dry, expected)

    prod_full = analyse(ledger)
    check = check_production(prod_full, ledger, st)
    variants = {}
    for variant, desc in VARIANTS.items():
        ev = pool[evaluated_mask(pool, variant)]
        frames = build_frames(ledger, ev)
        sets = production_sets(ledger)
        fs_rows, fs_counts = fs_table(ledger, pool, variant)
        entry = {"description": desc, "n_units_evaluated": int(len(ev)),
                 "n_extra_units_evaluated": int((ev["source"] == "c1c").sum()),
                 "coverage": {k: coverage_for(v[1], pool, variant, v[0]) for k, v in sets.items()},
                 "fs57_outcomes": fs_counts}
        for name, frame in frames.items():
            entry[name] = analyse(frame)
        if variant == "converged_only":
            entry["fs57_units"] = fs_rows
        variants[variant] = entry

    return {
        "_generated_by": "scripts/grid_compare.py",
        "meta": {
            "dry_run": dry,
            "summary_csv": {"path": rel(summary), "sha256": sha256(summary, text=True),
                            "n_rows": int(len(rows_all)), "n_start_A_rows": int(len(rows))},
            "ledger": {"path": rel(LEDGER), "sha256": sha256(LEDGER)},
            "c1c_extra_units": extra_info, "c1c_dir": rel(c1c_dir),
            "recipe": recipe, "recipe_source": rel(jp) if jp else "converged_recipe() default",
            "recipe_is_converged_recipe_default": bool(recipe == expected),
            "n_boot": sorted({int(x) for x in rows["n_boot"].dropna().unique()}),
            "unit_timeout_s": settings.get("unit_timeout_s", sss.GRID_DEFAULTS["unit_timeout_s"]),
            "unit_timeout_source": "unit JSON settings" if "unit_timeout_s" in settings
            else "GRID_DEFAULTS",
            "starts": ["A"], "production_cell": list(sss.GRID_PROD_SC),
            "bcc_grid_cell": [3, 3, 3], "imag_tol_thz": float(recipe["imag_tol_thz"]),
            "blowup_threshold_thz": BLOWUP_THZ,
            "grid_vs_ledger": check_grid_vs_ledger(rows, ledger),
            "production_reproduces_stats_hardening": check,
        },
        "definitions": DEFINITIONS,
        "run_status": run_status_block(rows, sss),
        "production_full": prod_full,
        "variants": variants,
        "changed_calls": changed_calls_block(rows, pd.DataFrame(extra_rows), ledger),
        "c1c_overlap": overlap,
    }


# ---------------------------------------------------------------------------- tables ----
# Shared with scripts/build_esi_tables.py (Table S22): both render from the JSON above.

def _frame_block(cmp: dict, variant: str, frame: str) -> dict:
    return cmp["production_full"] if frame == "production_full" else cmp["variants"][variant][frame]


def _g_i_fs(tag):
    return lambda b: f"{b['i'][tag]['n_sscha_false_stable']}/{b['i'][tag]['n_label_unstable']}"


def _g_i_scr(tag):
    return lambda b: b["i"][tag]["screen_call_unstable_on_those"]["short"]


def _g_i_agree(tag):
    return lambda b: (f"{b['i'][tag]['paired_T_le_300']['sscha_call_eq_screen_call']}/"
                      f"{b['i'][tag]['paired_T_le_300']['n']}")


def _g_ii(tag, meth):
    def g(b):
        r = b["ii"][tag][meth]
        s = f"{r['correct_unstable']}/{r['n_valid']}"
        return s + (f" (+{r['n_numerical_blowup']} blow-up)" if r["n_numerical_blowup"] else "")
    return g


def _g_iii(tag):
    return lambda b: b["iii"][tag]["call_agreement"]["short"]


def _g_iv(tag, T):
    return lambda b: b["iv"][tag]["false_unstable_by_T"][T]["short"]


def claim_table(cmp: dict, variant: str = "converged_only", typographic: bool = False) -> tuple:
    """(header, rows) of the claim-by-claim comparison: each production number beside the same
    number on the units the grid covers, and the converged value. Cells whose frame could not
    answer read 'n/a'."""
    le, x = ("≤", "×") if typographic else ("<=", "x")
    quantities = [
        ("(i)", "non-bcc units SSCHA calls stable against an unstable label (of those labelled "
                "unstable)", lambda tag: _g_i_fs(tag)),
        ("(i)", "of those, the screen's free-energy comparison calls unstable",
         lambda tag: _g_i_scr(tag)),
        ("(i)", f"non-bcc, T {le} 300 K: SSCHA call = screen call", lambda tag: _g_i_agree(tag)),
        ("(ii)", f"FE perovskites, T {le} 300 K: unstable recalled by the screen",
         lambda tag: _g_ii(tag, "softmode")),
        ("(ii)", f"FE perovskites, T {le} 300 K: unstable recalled by SSCHA",
         lambda tag: _g_ii(tag, "sscha")),
        ("(iii)", f"bcc, 100-600 K: screen call = SSCHA call (production 2{x}2{x}2, grid "
                  f"3{x}3{x}3)", lambda tag: _g_iii(tag))]
    for T in ("100", "300", "600", "900"):
        quantities.append(("(iv)", f"non-bcc SSCHA false-unstable at {T} K (of stable-labelled units)",
                           lambda tag, T=T: _g_iv(tag, T)))
    header = ["Claim", "Quantity", "Models"] + [lab for _, lab in FRAMES]
    rows = []
    for cl, q, make in quantities:
        for tag, tl in MODEL_TAGS:
            getter, cells = make(tag), []
            for fr, _ in FRAMES:
                try:
                    cells.append(getter(_frame_block(cmp, variant, fr)))
                except (KeyError, TypeError):
                    cells.append("n/a")
            rows.append([cl, q, tl] + cells)
    return header, rows


def coverage_table(cmp: dict, variant: str = "converged_only", typographic: bool = False) -> tuple:
    le = "≤" if typographic else "<="
    cov = cmp["variants"][variant]["coverage"]
    header = ["Claim", "Production units", "Count", "In the grid plan", "Evaluated",
              "Not in the grid (by T)", "Planned, not evaluated"]
    rows = []
    for k, c in cov.items():
        nie = ", ".join(f"{t} K: {n}" for t, n in c["not_in_grid_by_T"].items()) or "none"
        pne = ", ".join(f"{STATE_WORDS.get(s, s)} {n}" for s, n in sorted(c["by_state"].items())
                        if s not in ("evaluated", "not_in_grid")) or "none"
        rows.append([f"({k})", c["what"].replace("<=", le), c["n"], c["in_grid_plan"],
                     c["evaluated"], nie, pne])
    return header, rows


def status_table(cmp: dict) -> tuple:
    rs = cmp["run_status"]
    header = ["Model", "Planned", "ok", "Converged", "Wall cap", "Failed", "Not finished",
              "Blow-ups (over 50 THz)"]
    rows = []
    for m, v in sorted(rs["by_model"].items()):
        rows.append([m, v["planned"], v["ok"], v["converged"], v["wall_cap"], v["failed"],
                     v["not_finished"], v["blowups"]])
    t = {k: sum(v[k] for v in rs["by_model"].values())
         for k in ("planned", "ok", "converged", "wall_cap", "failed", "not_finished", "blowups")}
    rows.append(["total", t["planned"], t["ok"], t["converged"], t["wall_cap"], t["failed"],
                 t["not_finished"], t["blowups"]])
    return header, rows


def changed_table(cmp: dict, typographic: bool = False) -> tuple:
    arrow, x = ("→", "×") if typographic else ("->", "x")
    header = ["System", "Model", "T (K)", "Cell", "Production Hessian (THz)",
              "Converged Hessian (THz)", "Call", "Label", "Screen", "Converged", "Note"]

    def num(v):
        s = f"{v:+.2f}"
        return s.replace("-", "−") if typographic else s

    rows = []
    for u in cmp["changed_calls"]["units"]:
        lab = "n/s" if not u["label_scored"] else ("stable" if u["label_stable"] else "unstable")
        note = []
        if u.get("source") == "c1c":
            note.append("converged-mode unit outside the grid")
        if not u["same_cell_as_prod"]:
            note.append("cell and convergence change together")
        if u["blowup"]:
            note.append("converged blow-up")
        if u["production_blowup"]:
            note.append("production blow-up")
        rows.append([u["system"], u["model"], f"{u['T_K']:.0f}", u["cell"].replace("x", x),
                     num(u["production_hessian_thz"]), num(u["hessian_thz"]),
                     ("stable" if u["production_call_stable"] else "unstable") + f" {arrow} "
                     + ("stable" if u["call_stable"] else "unstable"), lab,
                     "-" if u["screen_call_stable"] is None else
                     ("stable" if u["screen_call_stable"] else "unstable"),
                     "yes" if u["converged"] else ("no" + (f" ({u['stop_reason']})" if u["stop_reason"] else "")),
                     "; ".join(note)])
    return header, rows


def mdtable(header, rows) -> str:
    out = ["| " + " | ".join(str(h) for h in header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def render_markdown(cmp: dict) -> str:
    m, rs, ch = cmp["meta"], cmp["run_status"], cmp["changed_calls"]
    L = []
    L.append("# Converged-recipe SSCHA grid against the production SSCHA claims")
    L.append("")
    L.append(f"- grid summary: `{m['summary_csv']['path']}` ({m['summary_csv']['n_start_A_rows']} "
             f"start-A rows, sha256 {m['summary_csv']['sha256'][:12]}); ledger "
             f"`{m['ledger']['path']}`" + ("; **DRY RUN, mocked SSCHA values: plumbing only**"
                                           if m["dry_run"] else ""))
    L.append(f"- recipe from `{m['recipe_source']}`: n_configs {m['recipe']['n_configs']}, max_pop "
             f"{m['recipe']['max_pop']}, n_hessian {m['recipe']['n_hessian']}, <= "
             f"{m['recipe']['max_steps_per_pop']} steps per population, meaningful_factor "
             f"{m['recipe']['meaningful_factor']} on the real error, imag tol {m['imag_tol_thz']} "
             f"THz; n_boot {m['n_boot']}; relaxation cap {m['unit_timeout_s']:g} s; start A only; "
             f"bcc grid cell 3x3x3, production 2x2x2")
    pr = m["production_reproduces_stats_hardening"]
    gl = m["grid_vs_ledger"]
    L.append(f"- production numbers recomputed here vs results/stats_hardening.json: "
             f"{'all ' + str(pr['n_checks']) + ' checks agree' if pr['all_ok'] else 'MISMATCH ' + str(pr['failed'])}"
             f"; summary.csv label/screen/production columns vs this ledger: "
             f"{'agree' if gl['all_ok'] else 'MISMATCH ' + str(gl['mismatches'])}")
    ex = m["c1c_extra_units"]
    L.append("- SrTiO3/MACE-MP-0 converged-mode units outside the grid: " + ("; ".join(
        f"{e['unit_tag']} " + (f"{e['status']}, converged {e['converged']}, {e['hessian_min_thz']:+.2f} THz"
                               if e["used"] and e.get("hessian_min_thz") is not None
                               else (e["status"] if e["used"] else f"not used ({e['reason']})"))
        for e in ex) or "none"))
    L.append("")
    L.append("## Run status")
    L.append("")
    cf = rs["converged_fraction_of_ok"]
    L.append(f"{rs['n_planned_units']} planned units: " + ", ".join(
        f"{k} {v}" for k, v in sorted(rs["by_status"].items())) + f". Of the {rs['n_ok']} ok, "
             f"{rs['n_converged']} converged"
             + ("" if cf is None else f" ({cf:.1%})") + "; stop reasons "
             + ", ".join(f"{k} {v}" for k, v in sorted(rs["stop_reasons_of_ok"].items()))
             + f". Wall-cap timeouts {len(rs['wall_cap_timeouts'])}, failed {len(rs['failed'])}, "
             f"not finished {len(rs['not_finished'])}, other recipe {len(rs['other_recipe'])}, "
             f"blow-ups {len(rs['blowups'])}, units production never returned "
             f"{rs['failed_in_production']}.")
    for f in rs["failed"]:
        L.append(f"- FAILED {f['unit_tag']}: {f['error']}")
    for f in rs["wall_cap_timeouts"]:
        L.append(f"- WALL CAP {f['unit_tag']}: {f['hessian_thz']:+.2f} THz after "
                 f"{f['n_populations']} populations")
    for f in rs["blowups"]:
        pv = "none" if f["production_thz"] is None else format(f["production_thz"], "+.1f")
        L.append(f"- BLOW-UP {f['unit_tag']}: {f['hessian_thz']:+.1f} THz (production {pv}; "
                 f"converged {f['converged']})")
    L.append("")
    L.append(mdtable(*status_table(cmp)))
    for variant in VARIANTS:
        v = cmp["variants"][variant]
        L.append("")
        L.append(f"## Variant `{variant}`: {v['description']}")
        L.append("")
        L.append(f"{v['n_units_evaluated']} units evaluated ({v['n_extra_units_evaluated']} of them "
                 "from the converged-mode directory).")
        L.append("")
        L.append("### Coverage of the production units")
        L.append("")
        L.append(mdtable(*coverage_table(cmp, variant)))
        fo = v["fs57_outcomes"]
        L.append("")
        L.append(f"The {fo['n']} SSCHA false-stables of Table S16, one by one: " + ", ".join(
            f"{STATE_WORDS.get(k, k)} {n}" for k, n in sorted(fo["outcomes"].items()))
            + f"; of the ones still false-stable the screen calls "
            f"{fo['still_false_stable_screen_unstable']['short']} unstable.")
        L.append("")
        L.append("### Claims")
        L.append("")
        L.append(mdtable(*claim_table(cmp, variant)))
    L.append("")
    L.append("## Calls that changed against production")
    L.append("")
    a, c, sc, b3 = ch["all_changed"], ch["converged_no_blowup_changed"], ch["same_cell"], ch["bcc_3x3x3_vs_2x2x2"]
    L.append(f"{a['n']} of {ch['n_compared_ok']} compared units differ (S->U {a['S_to_U']}, U->S "
             f"{a['U_to_S']}); among converged non-blow-up units {c['n']} (S->U {c['S_to_U']}, U->S "
             f"{c['U_to_S']}): same cell {sc['n']}, bcc 3x3x3 against 2x2x2 {b3['n']}; non-bcc "
             f"corrected against the label {ch['nonbcc_corrected_vs_label']}, worsened "
             f"{ch['nonbcc_worsened_vs_label']}.")
    L.append("")
    L.append(mdtable(*changed_table(cmp)) if ch["units"] else "(none)")
    if cmp["c1c_overlap"]:
        L.append("")
        L.append("## Grid units also run in the converged-mode directory (same recipe, start A)")
        L.append("")
        L.append(mdtable(["Unit", "Grid (THz)", "C1c (THz)", "|diff|", "Same call"],
                         [[o["unit_tag"], f"{o['grid_hessian_thz']:+.3f}", f"{o['c1c_hessian_thz']:+.3f}",
                           f"{o['abs_diff_thz']:.3f}", o["same_call"]] for o in cmp["c1c_overlap"]]))
    L.append("")
    L.append("## Definitions")
    L.append("")
    for k, d in cmp["definitions"].items():
        L.append(f"- **{k}**: {d}")
    return "\n".join(L)


# -------------------------------------------------------------------------------- CLI ----

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("summary", nargs="?", default=str(SUMMARY),
                    help=f"the grid's summary.csv (default {rel(SUMMARY)})")
    ap.add_argument("--c1c-dir", default=str(C1C_DIR),
                    help=f"the converged-mode directory holding the SrTiO3 600/900 K JSONs "
                         f"(default {rel(C1C_DIR)})")
    ap.add_argument("--out", default=str(OUT), help=f"output JSON (default {rel(OUT)})")
    ap.add_argument("--quiet", action="store_true", help="write the JSON, print no markdown")
    args = ap.parse_args(argv)
    out = Path(args.out).resolve()
    cmp = compare(Path(args.summary).resolve(), Path(args.c1c_dir).resolve(),
                  out_is_default=(out == OUT.resolve()))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_nan_to_none(cmp), indent=2, default=_py, allow_nan=False) + "\n",
                   encoding="utf-8")
    if not args.quiet:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        print(render_markdown(cmp))
    print(f"\nwrote {rel(out)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
