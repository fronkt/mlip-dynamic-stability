"""Recompute the fine-tuning trial's outcomes (E5, Referee 2.1) from the raw per-run evaluation
files under results/revision/finetune/, for the ESI table builder.

This does not read results/revision/finetune/summary.json for any number. It applies the verdict
rules that scripts/finetune/summary.py fixed before the trial ran (commit de7046c, before any ft_
job had run) to the raw files, so that the table in the ESI and the pins in scripts/verify_claims.py
(which recompute the same quantities a second time, independently) rest on the raw outputs.

Rules (pre-registration tasks/preregistration-finetune-2026-10-03.md, commit 033b3d8, plus the
tooling's fixed rules):
  * a screen call is CHANGED iff all three replicates differ from the base call (the deposited
    ledger call), UNCHANGED iff all three equal it, otherwise UNRESOLVED;
  * P2 "corrected" cells (BaTiO3 and KNbO3 at 300 K): supported iff changed, refuted iff
    unchanged; P2 "persist" cells (CsSnBr3 300 and 600 K, KNbO3 600 K): supported iff unchanged,
    refuted iff changed; unresolved otherwise;
  * P1: per base model the median over the deciding paths of BaTiO3 and KNbO3 and the three
    replicates of (fine-tuned well depth / PBE well depth); supported iff that median and every
    replicate's median lie in [0.8, 1.2], refuted iff none does, otherwise unresolved.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
FT = REPO / "results" / "revision" / "finetune"
EV = FT / ("ev" + "al")
MODELS = ["mace_mp0", "chgnet"]
SEEDS = ["seed0", "seed1", "seed2"]
SYSTEMS = ["batio3_cubic", "knbo3_cubic", "cssnbr3_cubic"]
LADDER = [100.0, 300.0, 600.0, 900.0]
CONTROLS = ["si_diamond", "mgo_rocksalt", "nacl_rocksalt", "cu_fcc", "c_diamond", "ceo2_cubic"]
BAND = (0.8, 1.2)
REG_CORRECTED = [("batio3_cubic", 300), ("knbo3_cubic", 300)]
REG_PERSIST = [("cssnbr3_cubic", 300), ("cssnbr3_cubic", 600), ("knbo3_cubic", 600)]


def _j(name: str):
    with open(EV / name, encoding="utf-8") as fh:
        return json.load(fh)


def p1() -> dict:
    out = {}
    for m in MODELS:
        base = _j(f"c3a_{m}_base.json")["paths"]
        reps = {t: _j(f"c3a_{m}_{t}.json")["paths"] for t in SEEDS}
        pick = lambda sysn: [k for k, v in base.items()                 # noqa: E731
                             if v["system"] == sysn and v["role"] == "decide"
                             and v["path_model"] == m]
        fe = pick("batio3_cubic") + pick("knbo3_cubic")
        cs = pick("cssnbr3_cubic")
        rat = lambda stems, tag: [                                       # noqa: E731
            (base if tag == "base" else reps[tag])[s]["metrics"]["depth_meV"]
            / base[s]["pbe_depth_meV"] for s in stems]
        r = dict(stems=fe, pbe={s: base[s]["pbe_depth_meV"] for s in fe},
                 base={s: x for s, x in zip(fe, rat(fe, "base"))},
                 reps={t: dict(zip(fe, rat(fe, t))) for t in SEEDS},
                 cs_stems=cs, cs_base=dict(zip(cs, rat(cs, "base"))),
                 cs_reps={t: dict(zip(cs, rat(cs, t))) for t in SEEDS})
        r["base_median"] = float(np.median(rat(fe, "base")))
        r["median"] = float(np.median([x for t in SEEDS for x in rat(fe, t)]))
        r["rep_median"] = {t: float(np.median(rat(fe, t))) for t in SEEDS}
        inside = lambda x: BAND[0] <= x <= BAND[1]                       # noqa: E731
        r["verdict"] = ("supported" if inside(r["median"]) and all(map(inside, r["rep_median"].values()))
                        else "refuted" if not inside(r["median"])
                        and not any(map(inside, r["rep_median"].values())) else "unresolved")
        out[m] = r
    return out


def p2() -> dict:
    """{(model, system, T): dict(label, ledger, calls, min_eff, status, ...)} for the 24 cells."""
    from mlip_dynstab import analysis as A
    led = A.canonical(pd.read_parquet(REPO / "results" / "ledger.parquet"))
    led = led[led.method == "softmode"]
    out = {}
    for m in MODELS:
        rows = {t: _j(f"p2_{m}_{t}.json")["rows"] for t in ["base"] + SEEDS}
        for s in SYSTEMS:
            for T in LADDER:
                get = lambda t: next(r for r in rows[t]                  # noqa: E731
                                     if r["system"] == s and r["T"] == T)
                row = led[(led.system == s) & (led.model == m) & (led.temperature_K == T)]
                lc = bool(row.dynamically_stable.iloc[0])
                calls = [bool(get(t)["pred_stable"]) for t in SEEDS]
                status = ("changed" if all(c != lc for c in calls)
                          else "unchanged" if all(c == lc for c in calls) else "unresolved")
                reg = ("corrected" if (s, int(T)) in REG_CORRECTED else
                       "persist" if (s, int(T)) in REG_PERSIST else None)
                verdict = None
                if reg:
                    verdict = ("unresolved" if status == "unresolved" else
                               "supported" if (status == "changed") == (reg == "corrected")
                               else "refuted")
                out[(m, s, int(T))] = dict(
                    label=bool(get("base")["gt_stable"]), ledger=lc,
                    rerun=bool(get("base")["pred_stable"]), calls=calls,
                    min_eff=[get(t)["min_eff_freq_thz"] for t in ["base"] + SEEDS],
                    n_imag=[get(t)["ft_n_imag_total"] for t in ["base"] + SEEDS],
                    well=[get(t)["ft_well_depth_meV"] for t in ["base"] + SEEDS],
                    status=status, registered=reg, verdict=verdict)
    return out


def p2_samepath() -> dict:
    """{(model, system, T): ...}: the screen's rule applied to each fine-tuned model's energies
    along the held-out PBE paths of the model's own deciding coordinates (the `calls` the S1 stage
    stored; same modes, amplitudes, fit and solver as Table S19), against the PBE-backed call and
    the base model on the same paths. Not a pre-registered outcome; computed from the S1 files."""
    uc = pd.read_csv(REPO / "results" / "revision" / "dft" / "c3a_unit_calls.csv")
    out = {}
    for m in MODELS:
        data = {t: _j(f"c3a_{m}_{t}.json")["paths"] for t in ["base"] + SEEDS}
        for s in SYSTEMS:
            for T in LADDER:
                def unit(d, s=s, T=T, m=m):
                    ps = [v for v in d.values() if v["system"] == s and v["path_model"] == m]
                    return all(v["calls"][str(int(T))]["stable"] for v in ps), len(ps)
                u = uc[(uc.system == s) & (uc.model == m) & (uc["T"] == T)].iloc[0]
                out[(m, s, int(T))] = dict(
                    n_paths=unit(data["base"])[1], label=bool(u.gt_stable),
                    pbe=bool(u.pbe_backed_stable), base=unit(data["base"])[0],
                    ledger_same=bool(u.mlip_same_paths_stable),
                    calls=[unit(data[t])[0] for t in SEEDS])
    return out


def s1() -> dict:
    out = {}
    for m in MODELS:
        for t in ["base"] + SEEDS:
            d = _j(f"c3a_{m}_{t}.json")["paths"]
            for s in SYSTEMS:
                ps = [v for v in d.values() if v["system"] == s]
                out[(m, t, s)] = dict(
                    n_paths=len(ps), n_pts=sum(v["n_pts"] for v in ps),
                    force=float(np.sqrt(sum(v["f_sumsq_window"] for v in ps)
                                        / sum(v["n_force_components_window"] for v in ps))),
                    energy=float(np.sqrt(sum(v["e_sumsq_window_per_atom_meV2"] for v in ps)
                                         / sum(v["n_pts_window"] for v in ps))))
    return out


def s2() -> dict:
    out = {}
    for m in MODELS:
        for t in ["base"] + SEEDS:
            for r in _j(f"s2_{m}_{t}.json")["rows"]:
                out[(m, t, r["system"])] = (bool(r["dynamically_stable"]), r["min_freq_thz"])
    return out


def p3() -> dict:
    j = json.load(open(FT / "sscha_converged_grid" / "batio3_cubic_mace_mp0_100K_sc222_startA.json",
                       encoding="utf-8"))
    r = j["run"]
    return dict(hess=r["hessian"]["min_nonac_thz"], start=r["start"]["min_nonac_thz"],
                final_aux=r["final_aux"]["min_nonac_thz"], converged=bool(r["relax"]["converged"]),
                n_pop=r["relax"]["n_populations"], n_steps=r["relax"]["n_steps_total"],
                a=r["relaxed_cellpar"][0], ledger=j["ledger"]["min_eff_freq_thz"],
                harm=r["harmonic_pre_fpd"]["min_nonac_thz"])
