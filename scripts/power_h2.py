"""Power projection for the H2 transfer-asymmetry test, and Holm adjustment of the paper's p-value families.

Everything here is CPU-only, numpy + pandas, and reads the deposited ledger and the derived JSONs
under results/. It changes no existing number: it writes two NEW files,

    results/power_h2.json       (part 1)
    results/holm_adjusted.json  (part 2)

    python scripts/power_h2.py [--only power|holm] [--n-resamples 2000] [--seed 0]

PART 1. POWER PROJECTION FOR THE H2 TEST
----------------------------------------
H2 is the system-clustered exact paired test of transfer asymmetry on the matched set (15 systems x
5 models): b = harmonic call right and screen call wrong at T, c = the reverse; b v c = 17 v 4 at
300 K and the clustered p is 0.152 (``stats.cluster_exact_paired``, ESI Table S15). The test depends
on the data only through each system's net discordance net_i = b_i - c_i: it asks how often a random
sign flip per system gives |sum of signed nets| at least as large as the observed |sum net_i|. So a
"study" is a list of per-system nets, and a larger study is a longer list.

(i) The observed pattern is rebuilt from the ledger with the repo's own functions
(``stats_hardening._h2_pairs`` / ``_h2_test``, which call ``stats.cluster_exact_paired``) and must
reproduce p = 0.15234 at 300 K and the value stored in results/stats_hardening.json, or the script
stops.

(ii) How likely is p < 0.05 with n_sys systems? For every n_sys in {15, 20, 25, 30, 40, 50, 60}, 2000
synthetic studies are drawn, by resampling WHOLE SYSTEMS (their observed (b_i, c_i)) with
replacement, and each is tested exactly. The fraction with p < 0.05 is the projected power. The
resampling is done separately at 300 K and at 600 K. Three ways of drawing the n_sys systems:

  bootstrap_all      Draw n_sys systems with replacement from all 15 observed systems. The simple
                     design projection: "if the population of systems looks like these 15, how many
                     would be enough?"
  stratified_shared  The CONSERVATIVE variant. The systems the screen mis-calls for at least four of
                     the five models at that T (``shared-screen-error`` systems: BaTiO3, KNbO3,
                     CsSnBr3 at 300 K; five systems at 600 K, taken from the same function the paper
                     uses) carry nearly all of b (13 of 17 at 300 K, 22 of 23 at 600 K), and the
                     paper reads them as the screen's own single-mode error rather than as an MLIP
                     effect. Here they enter every synthetic study at EXACTLY their observed share
                     of systems (3/15 at 300 K, i.e. round(n_sys x 3/15) of the n_sys), their
                     patterns drawn with replacement from those three; all remaining systems are
                     drawn with replacement ONLY from the systems outside that set (12 at 300 K),
                     whose observed b v c is 4 v 4 at 300 K. So the shared-error systems cannot
                     be over-represented by luck, and every other system is an ordinary one.
                     Read the output with this in mind: fixing the share removes the lucky draws
                     that lift bootstrap_all at small n_sys, so stratified_shared is the lower of
                     the two there (15-20 systems), but it also removes the unlucky ones, so once
                     n_sys is large enough for the fixed share to be several systems it can be
                     the higher of the two. Its guarantee is only that the share of
                     screen-error systems is not inflated, not that its power is lower at every n.
  non_shared_only    EXTRA, not asked for: the shared-error systems left out altogether. The
                     power to see an asymmetry in the systems where the screen is not itself at
                     fault, the real lower bound. At 300 K those 12 systems net to zero (4 v 4),
                     so it sits at or below the nominal 5% whatever n_sys is.

What this is and is not. It is a bootstrap design projection that takes the observed per-system
pattern as the truth. It is not a power calculation for a stated effect size: the observed pattern is
noisy, was examined after the fact, and 13 of the 17 b come from three systems, so a projection
built on it carries the same winner's-curse caveat as any post-hoc power. Read it as "how large
would a replication have to be if the sample is representative", not as a probability about the
world. Monte-Carlo standard error of each cell is at most 0.011 (2000 resamples).

Exact p at any n_sys. ``stats.cluster_exact_paired`` enumerates 2^k sign vectors and refuses k > 20,
which cannot reach n_sys = 60. Because the nets are small integers the same null is obtained exactly
by convolving the k two-point distributions (+/- |net_i| with probability 1/2 each), O(k * sum|net|);
``p_clustered_from_nets`` does that, and the script checks it against the repo function on 300
random studies of up to 14 systems before using it (they must agree to the 5 decimals the repo
function returns).

PART 2. HOLM ADJUSTMENT
-----------------------
The paper states most of its p-values as one of a set tested together, and some sets explicitly
unadjusted ("None is adjusted for multiple comparisons", the 19 force-spread scores of ESI Table
S18). The families below are the sets the paper reports side by side; Holm's step-down adjustment
(m p_(1), (m-1) p_(2), ..., made monotone, capped at 1) is applied within each. Families:

  h2_ladder_all_models / h2_ladder_excl_orb_v2   the four-temperature H2 ladder (ESI Table S15, the
                                                 manuscript's ladder table), clustered exact p;
  h2_convention_x_T                              H2 under the 5 distinct frozen-cell conventions x 4 T
                                                 = 20 (``results/h2_by_convention.json``; the
                                                 ``cell_minimal`` entry is the production convention
                                                 again and is not counted twice), clustered exact p;
  h2_phi_association_by_T                        the phi column of the manuscript's ladder table, the
                                                 system-clustered permutation p of the b/c association,
                                                 four temperatures (``results/estimator_noise.json``);
  screen_vs_sscha_S10                            the six clustered p of ESI Table S10 (three system
                                                 sets x with/without ORB-v2); "combined displacive
                                                 only" is also given for the two rows that carry the
                                                 claim;
  h3_guardrail_auc                               the two ensemble-disagreement AUCs (vote split,
                                                 frequency spread) x with/without ORB-v2, clustered
                                                 permutation p (manuscript 3.4, ESI Table S9);
  force_spread_S18 / force_spread_S18_secondary  the 19 pre-registered force-spread scores of ESI
                                                 Table S18 (primary + 18 secondary), and the 18
                                                 secondary scores alone.

Not adjusted, because the paper reports them as descriptive or as companions: the unit-level McNemar
p-values (labelled "do not quote"), the leave-one-system-out p-values, and Spearman rho (no test).
The p-values are read from the JSON files as stored (rounded to 4-5 decimals by their writers).

Both outputs are deterministic (fixed seed, no timestamp).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from mlip_dynstab import analysis as A   # noqa: E402
from mlip_dynstab import stats as S      # noqa: E402
import stats_hardening as SH             # noqa: E402

LEDGER = REPO / "results" / "ledger.parquet"
STATS_JSON = REPO / "results" / "stats_hardening.json"
CONV_JSON = REPO / "results" / "h2_by_convention.json"
NOISE_JSON = REPO / "results" / "estimator_noise.json"
FSPREAD_JSON = REPO / "results" / "revision" / "force_spread" / "summary.json"
OUT_POWER = REPO / "results" / "power_h2.json"
OUT_HOLM = REPO / "results" / "holm_adjusted.json"

ALPHA = 0.05
N_SYS_GRID = (15, 20, 25, 30, 40, 50, 60)
POWER_TEMPS = (300.0, 600.0)
VARIANTS = ("bootstrap_all", "stratified_shared", "non_shared_only")
N_VALIDATE = 300


# ------------------------------------------------------------------ exact clustered p ----

def p_clustered_from_nets(nets) -> float:
    """Two-sided exact system-clustered p for per-system net discordances ``nets``.

    The null of ``stats.cluster_exact_paired``: each system's net is multiplied by an independent
    random sign, and p = P(|sum| >= |observed sum|). Computed by convolution rather than by
    enumerating 2^k sign vectors, so it is exact for any number of systems.
    """
    nets = np.asarray(nets, dtype=int)
    obs = abs(int(nets.sum()))
    a = np.abs(nets[nets != 0])
    if a.size == 0:
        return 1.0
    s = int(a.sum())
    pmf = np.zeros(2 * s + 1)
    pmf[s] = 1.0
    for v in a:
        new = np.zeros_like(pmf)
        new[v:] += 0.5 * pmf[:-v]
        new[:-v] += 0.5 * pmf[v:]
        pmf = new
    support = np.abs(np.arange(-s, s + 1))
    return float(pmf[support >= obs].sum())


def validate_convolution(seed: int = 0, n_cases: int = N_VALIDATE) -> dict:
    """Check ``p_clustered_from_nets`` against ``stats.cluster_exact_paired`` on random studies."""
    rng = np.random.default_rng([seed, 99])
    worst = 0.0
    for _ in range(n_cases):
        k = int(rng.integers(2, 15))
        nets = rng.integers(-5, 6, size=k)
        a, b, cl = [], [], []
        for i, n in enumerate(nets):
            if n > 0:                      # n units where method A wins
                a += [True] * n
                b += [False] * n
                cl += [i] * n
            elif n < 0:                    # |n| units where method B wins
                a += [False] * -n
                b += [True] * -n
                cl += [i] * -n
            else:                          # a tied system: one unit both get right
                a.append(True)
                b.append(True)
                cl.append(i)
        ref = S.cluster_exact_paired(a, b, cl)["p_exact_clustered"]
        worst = max(worst, abs(round(p_clustered_from_nets(nets), 5) - ref))
    assert worst < 1e-9, f"convolution p disagrees with stats.cluster_exact_paired by {worst}"
    return {"cases": n_cases, "max_abs_difference_at_5_decimals": worst,
            "reference": "stats.cluster_exact_paired"}


# --------------------------------------------------------------- observed H2 pattern ----

def observed_pattern(df: pd.DataFrame, t: float) -> dict:
    """The matched-set (b_i, c_i) per system at T, tested with the repo's own functions."""
    m = SH._h2_pairs(df, t)
    res = SH._h2_test(m)
    per = {}
    for s, g in m.groupby("system"):
        b = int((g["harm_ok"] & ~g["ft_ok"]).sum())
        c = int((~g["harm_ok"] & g["ft_ok"]).sum())
        per[s] = {"b": b, "c": c, "net": b - c}
    shared = sorted(SH._shared_screen_errors(df, t))
    return {"n_pairs": res["n_pairs"], "n_systems": res["n_systems"],
            "b": res["b_harm_right_ft_wrong"], "c": res["c_harm_wrong_ft_right"],
            "p_clustered": res["clustered_by_system"]["p_exact_clustered"],
            "p_mcnemar_unit_level_companion_only": res["mcnemar_exact_p_UNIT_LEVEL"],
            "per_system": per, "shared_screen_error_systems": shared}


def check_against_stats_json(obs: dict, t: float) -> None:
    """Fail unless the rebuilt pattern is the one results/stats_hardening.json already records."""
    rec = json.loads(STATS_JSON.read_text(encoding="utf-8"))["h2_clustered"][str(int(t))]
    ref = rec["all_models"]
    assert obs["p_clustered"] == ref["clustered_by_system"]["p_exact_clustered"], (t, obs["p_clustered"])
    assert (obs["b"], obs["c"]) == (ref["b_harm_right_ft_wrong"], ref["c_harm_wrong_ft_right"]), t
    assert {s: v for s, v in obs["per_system"].items()} == ref["per_system_discordance"], t
    assert obs["shared_screen_error_systems"] == sorted(rec["shared_screen_error_systems"]), t


# ------------------------------------------------------------------------ simulation ----

def _draw(rng, pool: np.ndarray, n: int) -> np.ndarray:
    return pool[rng.integers(0, len(pool), size=n)] if n > 0 else pool[:0]


def simulate(per_system: dict, shared: list[str], variant: str, n_sys: int,
             n_resamples: int, rng: np.random.Generator) -> dict:
    """Projected P(clustered p < alpha) at n_sys systems, by resampling whole systems."""
    names = sorted(per_system)
    all_pool = np.array([[per_system[s]["b"], per_system[s]["c"]] for s in names])
    sh_pool = np.array([[per_system[s]["b"], per_system[s]["c"]] for s in names if s in shared])
    rest_pool = np.array([[per_system[s]["b"], per_system[s]["c"]] for s in names if s not in shared])
    if variant == "bootstrap_all":
        draw = lambda: _draw(rng, all_pool, n_sys)                                    # noqa: E731
    elif variant == "stratified_shared":
        n_sh = int(round(n_sys * len(sh_pool) / len(all_pool)))
        draw = lambda: np.vstack([_draw(rng, sh_pool, n_sh), _draw(rng, rest_pool, n_sys - n_sh)])  # noqa: E731
    elif variant == "non_shared_only":
        draw = lambda: _draw(rng, rest_pool, n_sys)                                   # noqa: E731
    else:
        raise ValueError(variant)
    ps = np.empty(n_resamples)
    bs = np.empty(n_resamples)
    cs = np.empty(n_resamples)
    for r in range(n_resamples):
        x = draw()
        ps[r] = p_clustered_from_nets(x[:, 0] - x[:, 1])
        bs[r], cs[r] = x[:, 0].sum(), x[:, 1].sum()
    power = float((ps < ALPHA).mean())
    return {"power": round(power, 4),
            "mc_se": round(float(np.sqrt(power * (1 - power) / n_resamples)), 4),
            "median_p": round(float(np.median(ps)), 5),
            "mean_b": round(float(bs.mean()), 2), "mean_c": round(float(cs.mean()), 2)}


def run_power(df: pd.DataFrame, n_resamples: int, seed: int) -> dict:
    out: dict = {
        "_generated_by": "scripts/power_h2.py",
        "_seed": seed, "_n_resamples": n_resamples, "_alpha": ALPHA,
        "_n_sys_grid": list(N_SYS_GRID),
        "_what": ("Projected P(system-clustered exact p < 0.05) for the H2 transfer-asymmetry test as the "
                  "number of systems grows, by resampling whole systems with replacement from the "
                  "observed per-system (b, c) at each T. A bootstrap design projection that takes the "
                  "observed pattern as the truth, not a power calculation for a stated effect size; "
                  "see the module docstring of scripts/power_h2.py."),
        "_variants": {
            "bootstrap_all": "n_sys systems drawn with replacement from all 15",
            "stratified_shared": ("the requested conservative variant: the shared-screen-error systems "
                                  "(screen mis-calls >= 4 of 5 models at that T) enter at exactly their "
                                  "observed share of systems, drawn from those systems; every other system "
                                  "is drawn only from the systems outside that set. Lower than bootstrap_all "
                                  "at small n_sys, not necessarily at large n_sys (fixing the share also "
                                  "removes unlucky draws)"),
            "non_shared_only": "extra: shared-screen-error systems excluded altogether (lower bound)",
        },
        "dp_validation": validate_convolution(seed),
        "observed": {}, "power": {}, "n_sys_for_80pct_power": {},
    }
    for ti, t in enumerate(POWER_TEMPS):
        key = str(int(t))
        obs = observed_pattern(df, t)
        check_against_stats_json(obs, t)
        out["observed"][key] = obs
        out["power"][key] = {}
        out["n_sys_for_80pct_power"][key] = {}
        for vi, variant in enumerate(VARIANTS):
            cells = {}
            for n_sys in N_SYS_GRID:
                rng = np.random.default_rng([seed, int(t), vi, n_sys])
                cells[str(n_sys)] = simulate(obs["per_system"], obs["shared_screen_error_systems"],
                                             variant, n_sys, n_resamples, rng)
            out["power"][key][variant] = cells
            first = next((n for n in N_SYS_GRID if cells[str(n)]["power"] >= 0.8), None)
            out["n_sys_for_80pct_power"][key][variant] = first   # None: not reached by 60 systems
    # (i) the reproduction the task asks for, stated at the top level too
    out["reproduces_p_300K"] = out["observed"]["300"]["p_clustered"]
    return out


def print_power(res: dict) -> None:
    print(f"H2 power projection: P(system-clustered exact p < {ALPHA}), "
          f"{res['_n_resamples']} resamples per cell, seed {res['_seed']}")
    for k in ("300", "600"):
        o = res["observed"][k]
        print(f"  observed {k} K: b v c = {o['b']} v {o['c']}, clustered p = {o['p_clustered']:.5f}; "
              f"shared-screen-error systems: {', '.join(o['shared_screen_error_systems'])}")
    short = {"bootstrap_all": "boot-all", "stratified_shared": "strat", "non_shared_only": "non-shr"}
    hdr = "  n_sys | " + " | ".join(f"{k} K {short[v]:>8}" for k in ("300", "600") for v in VARIANTS)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for n in N_SYS_GRID:
        cells = [f"{res['power'][k][v][str(n)]['power']:>12.3f}" for k in ("300", "600") for v in VARIANTS]
        print(f"  {n:>5} | " + " | ".join(cells))
    print("  boot-all = bootstrap_all; strat = stratified_shared (the conservative variant); "
          "non-shr = non_shared_only (extra)")
    for k in ("300", "600"):
        print(f"  first n_sys with power >= 0.8 at {k} K: {res['n_sys_for_80pct_power'][k]}")


# ------------------------------------------------------------------------------ Holm ----

def holm_adjust(p) -> np.ndarray:
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adj[idx] = running
    return adj


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_families() -> list[dict]:
    sh, conv, noise, fs = _load(STATS_JSON), _load(CONV_JSON), _load(NOISE_JSON), _load(FSPREAD_JSON)
    temps = ("100", "300", "600", "900")
    fams: list[dict] = []

    def fam(fid, desc, reported_in, test, items, note=None):
        fams.append({"id": fid, "description": desc, "reported_in": reported_in, "test": test,
                     "items": [{"label": lab, "p_raw": float(p), "source": src} for lab, p, src in items],
                     **({"note": note} if note else {})})

    for tag, nice in (("all_models", "all five models"), ("excl_orb_v2", "without ORB-v2")):
        fam(f"h2_ladder_{tag}", f"H2 transfer asymmetry b v c at the four ladder temperatures, {nice}",
            "manuscript 3.2 ladder table; ESI Table S15", "system-clustered exact (2^k sign flips)",
            [(f"{t} K", sh["h2_clustered"][t][tag]["clustered_by_system"]["p_exact_clustered"],
              f"results/stats_hardening.json h2_clustered/{t}/{tag}/clustered_by_system/p_exact_clustered")
             for t in temps])

    items = []
    for cname, c in conv["conventions"].items():
        if cname == "cell_minimal":      # the production convention under its other name
            assert c["by_T"] == conv["conventions"]["production"]["by_T"]
            continue
        for t in temps:
            items.append((f"{cname} @ {t} K", c["by_T"][t]["p_clustered"],
                          f"results/h2_by_convention.json conventions/{cname}/by_T/{t}/p_clustered"))
    assert len(items) == 20
    fam("h2_convention_x_T", "H2 under each frozen-cell convention of the screen at each ladder T (all five models)",
        "ESI S1.3 ('the 20 convention and temperature combinations'), Table S14",
        "system-clustered exact (2^k sign flips)", items,
        note="production and cell_minimal are the same convention and are counted once")

    fam("h2_phi_association_by_T",
        "phi (harmonic-right vs finite-T-right association on the matched pairs) at the four ladder temperatures",
        "manuscript 3.2 ladder table (last column)", "system-clustered permutation, 10,000 draws, seed 0",
        [(f"{t} K", noise["C_association"]["by_temperature"][t]["phi_perm_p_two_sided"],
          f"results/estimator_noise.json C_association/by_temperature/{t}/phi_perm_p_two_sided") for t in temps],
        note="Monte-Carlo permutation p-values (10,000 draws), so the adjusted values carry that noise")

    s10 = [(f"{lab} / {tag}", sh["screen_vs_sscha_paired"][f"{key}__{tag}"]["clustered_by_system"]["p_exact_clustered"],
            f"results/stats_hardening.json screen_vs_sscha_paired/{key}__{tag}/clustered_by_system/p_exact_clustered")
           for key, lab in (("fe_oxide", "fe oxide"), ("fluorite", "fluorite"),
                            ("displacive_combined", "displacive combined"))
           for tag in ("all_models", "excl_orb_v2")]
    fam("screen_vs_sscha_S10", "Screen vs SSCHA paired contrast, the six rows of ESI Table S10",
        "manuscript 3.3; ESI Table S10", "system-clustered exact (randomising the method label over systems)", s10,
        note=("with 5, 3 and 2 systems the smallest attainable p is 0.0625, 0.25 and 0.5, so no row can "
              "reach 0.05 before or after adjustment"))
    fam("screen_vs_sscha_S10_combined_only",
        "The two combined-displacive rows of Table S10, the ones that carry the claim", "manuscript 3.3",
        "system-clustered exact", [i for i in s10 if i[0].startswith("displacive combined")])

    g = sh["orb_split_s3"]["guardrail"]
    fam("h3_guardrail_auc", "Ensemble-disagreement AUCs (vote split, frequency spread), with and without ORB-v2",
        "manuscript 3.4; ESI Table S9", "system-clustered permutation, 10,000 draws, seed 0",
        [(f"{sc} / {tag}", g[tag][f"auc_{sc}"]["p_perm_clustered"],
          f"results/stats_hardening.json orb_split_s3/guardrail/{tag}/auc_{sc}/p_perm_clustered")
         for sc in ("vote_disagreement", "freq_std") for tag in ("all_models", "excl_orb_v2")])

    s18 = []
    for sec in ("consensus_level", "per_model_level"):
        for k, v in fs[sec].items():
            if v.get("p_perm_clustered") is not None:
                s18.append((k, v["p_perm_clustered"],
                            f"results/revision/force_spread/summary.json {sec}/{k}/p_perm_clustered"))
    assert len(s18) == 19 and s18[0][0] == "primary_S_5model", len(s18)
    fam("force_spread_S18", "The 19 force-spread scores of ESI Table S18 that carry a clustered p (primary + 18 secondary)",
        "manuscript 3.4 ('None is adjusted for multiple comparisons'); ESI Table S18",
        "system-clustered permutation, 10,000 draws, seed 0", s18,
        note="score (g) has no p-value (unequal blocks) and is not in the family")
    fam("force_spread_S18_secondary", "The 18 secondary scores of ESI Table S18 (the pre-registered primary tested alone)",
        "manuscript 3.4; ESI Table S18", "system-clustered permutation, 10,000 draws, seed 0", s18[1:])
    return fams


def run_holm() -> dict:
    out: dict = {
        "_generated_by": "scripts/power_h2.py",
        "_alpha": ALPHA,
        "_method": "Holm step-down: adj_(i) = max over j<=i of min(1, (m - j + 1) p_(j)), p sorted ascending",
        "_note": ("Additional output. No existing number is changed. Families are the sets of p-values the "
                  "paper reports side by side; descriptive and companion p-values (unit-level McNemar, "
                  "leave-one-system-out, Spearman) are not adjusted. Several p-values are discrete or "
                  "Monte-Carlo, see each family's note."),
        "families": {},
    }
    for f in build_families():
        p = [i["p_raw"] for i in f["items"]]
        adj = holm_adjust(p)
        for it, a in zip(f["items"], adj):
            it["p_holm"] = round(float(a), 5)
            it["below_alpha_raw"] = bool(it["p_raw"] < ALPHA)
            it["below_alpha_holm"] = bool(a < ALPHA)
        f["m"] = len(p)
        f["n_below_alpha_raw"] = sum(i["below_alpha_raw"] for i in f["items"])
        f["n_below_alpha_holm"] = sum(i["below_alpha_holm"] for i in f["items"])
        out["families"][f["id"]] = f
    return out


def print_holm(res: dict) -> None:
    print(f"\nHolm adjustment within families (alpha = {res['_alpha']}); raw -> Holm, "
          f"count below alpha raw/Holm")
    for fid, f in res["families"].items():
        sm = sorted(f["items"], key=lambda i: i["p_raw"])
        head = ", ".join(f"{i['label']}: {i['p_raw']:.4g}->{i['p_holm']:.3g}" for i in sm[:4])
        print(f"  {fid:<34} m={f['m']:<2} below alpha {f['n_below_alpha_raw']}/{f['n_below_alpha_holm']}  "
              f"smallest: {head}")


# ------------------------------------------------------------------------------ main ----

def _write(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {path.relative_to(REPO).as_posix()}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", choices=("power", "holm"), help="run one part (default: both)")
    ap.add_argument("--n-resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.only != "holm":
        df = A.load_canonical(LEDGER)
        res = run_power(df, args.n_resamples, args.seed)
        print_power(res)
        _write(OUT_POWER, res)
    if args.only != "power":
        hol = run_holm()
        print_holm(hol)
        _write(OUT_HOLM, hol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
