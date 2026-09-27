"""Regenerate the paper figures from results/ledger.parquet. Every plotted number is computed
here from the ledger (through ``analysis.canonical``), independent of when or where a unit ran.

Figures, in the order the manuscript first cites them:
  Fig. 1  fig_tolerance_sweep     harmonic false-stable / false-unstable calls vs imaginary tolerance
  Fig. 2  fig_softmode_heat       soft-mode screen at 100 K: symmetric-point curvature per unit
  Fig. 3  fig_sscha_bcc           SSCHA lowest free-energy-Hessian frequency vs T, bcc Ti/Zr/Hf
  Fig. 4  fig_method_agreement    screen curvature vs SSCHA Hessian frequency on the bcc metals
  Fig. 5  fig_displacive_recall   recall of the unstable cubic phase, FE perovskites, T <= 300 K
  Fig. 6  fig_ensemble_guardrail  consensus error on split-vote vs unanimous units, +/- ORB-v2

Each figure is written as a 600 dpi PNG (the copy embedded in the DOCX) and a 600 dpi LZW TIFF,
both flattened to RGB on white, plus a numbered upload copy results/figures/upload/FigN.tif.
The numbering is checked against the figure links in paper/manuscript.md before anything is
written. Run from the repository root: python scripts/make_figures.py
"""
from __future__ import annotations
import os
import re
import sys
from io import BytesIO
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from PIL import Image

LEDGER = os.environ.get("LEDGER", "results/ledger.parquet")
OUT = os.environ.get("FIGDIR", "results/figures")
MANUSCRIPT = "paper/manuscript.md"
SSCHA_GRID = "results/sscha_units_v1.csv"     # the attempted SSCHA grid (208 units)
os.makedirs(OUT, exist_ok=True)
from mlip_dynstab import analysis as A
from mlip_dynstab import stats as S
from mlip_dynstab.systems import load_specs
# The ledger is append-only and holds BOTH the legacy single-mode softmode grid and the current
# multi-mode one. `canonical` keeps only the current generation; reading the parquet directly
# would double-count every softmode unit and blend two different measurements in one figure.
df = A.canonical(pd.read_parquet(LEDGER))
_sm = df[df["method"] == "softmode"]
print(f"[figures] canonical ledger: {len(df)} rows, {len(_sm)} softmode "
      f"({'multi-mode' if _sm.get('ft_n_imag_total') is not None and _sm['ft_n_imag_total'].notna().any() else 'legacy'})")

# Model order and names as the paper writes them; one colour per model across every figure
# (Okabe-Ito, colour-blind safe).
MODELS = ["mace_mp0", "chgnet", "orb_v2", "sevennet0", "mattersim"]
NAME = {"mace_mp0": "MACE-MP-0", "chgnet": "CHGNet", "orb_v2": "ORB-v2",
        "sevennet0": "SevenNet-0", "mattersim": "MatterSim"}
COLOR = {"mace_mp0": "#0072B2", "chgnet": "#E69F00", "orb_v2": "#CC79A7",
         "sevennet0": "#009E73", "mattersim": "#D55E00"}
C_A, C_B = "#0072B2", "#D55E00"               # two-category bars (blue / vermillion)
KEY = ["system", "model", "temperature_K"]
DPI = 600
N_RESAMPLE, SEED = 10000, 0                    # as scripts/stats_hardening.py

# Upload numbering: order of first citation in the manuscript (checked by _check_numbering).
FIG_NUMBER = {"fig_tolerance_sweep": 1, "fig_softmode_heat": 2, "fig_sscha_bcc": 3,
              "fig_method_agreement": 4, "fig_displacive_recall": 5,
              "fig_ensemble_guardrail": 6}

_SPECS = {s.id: s for s in load_specs()}
_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def sys_label(sid: str) -> str:
    """Display name for a system id, e.g. batio3_cubic -> BaTiO₃, zr_bcc -> bcc Zr."""
    s = _SPECS.get(sid)
    if s is None:
        return sid
    f = s.formula.translate(_SUB)
    if s.klass == "bcc-metal":
        return f"bcc {f}"
    if s.prototype == "alpha-AgI":
        return f"α-{f}"
    if s.prototype in ("diamond", "fcc"):
        return f"{f} ({s.prototype})"
    return f


def _trivial_bcc_pairs() -> set:
    """(system, model) bcc pairs whose harmonic layer has no instability at the production
    tolerance: there both the screen and SSCHA agree without any thermal stabilisation being
    tested (the same definition as stats_hardening.bcc_agreement and verify_claims)."""
    h = df[(df["method"] == "harmonic") & df["system"].str.contains("bcc")]
    h = h[h["pred_stable"].astype(bool)]
    return set(zip(h["system"], h["model"]))


def _check_numbering() -> None:
    """Refuse to write numbered upload copies that disagree with the manuscript's figure links."""
    if not os.path.exists(MANUSCRIPT):
        print(f"[figures] WARNING: {MANUSCRIPT} not found; upload numbering not checked")
        return
    text = open(MANUSCRIPT, encoding="utf-8").read()
    found = {name: int(n) for n, name in re.findall(
        r"!\[\*\*Fig\.\s*(\d+)\*\*.*?\]\([^)]*?results/figures/(\w+)\.png\)", text, flags=re.S)}
    if not found:
        print("[figures] WARNING: no figure links parsed from the manuscript; numbering not checked")
        return
    if found != FIG_NUMBER:
        raise SystemExit(f"[figures] upload numbering {FIG_NUMBER} disagrees with the manuscript "
                         f"figure links {found}; fix FIG_NUMBER before regenerating")
    print(f"[figures] upload numbering matches the manuscript: "
          + ", ".join(f"Fig{n}={k}" for k, n in sorted(found.items(), key=lambda kv: kv[1])))


def _save(fig, name: str) -> None:
    """Write <name>.png and <name>.tiff at 600 dpi, flattened to RGB on white (no alpha; the TIFF
    LZW-compressed), and the numbered upload copy upload/FigN.tif."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    with Image.open(buf) as im:
        im.load()
        rgb = Image.new("RGB", im.size, (255, 255, 255))
        rgb.paste(im, mask=im.getchannel("A") if "A" in im.getbands() else None)
    rgb.save(f"{OUT}/{name}.png", dpi=(DPI, DPI), optimize=True)
    rgb.save(f"{OUT}/{name}.tiff", dpi=(DPI, DPI), compression="tiff_lzw")
    n = FIG_NUMBER.get(name)
    if n is not None:
        os.makedirs(f"{OUT}/upload", exist_ok=True)
        rgb.save(f"{OUT}/upload/Fig{n}.tif", dpi=(DPI, DPI), compression="tiff_lzw")
    print(f"wrote {OUT}/{name}.png/.tiff" + (f" and upload/Fig{n}.tif" if n else "")
          + f"  ({rgb.size[0]}x{rgb.size[1]} px, {DPI} dpi, RGB)")


def fig_sscha_bcc():
    """Fig. 3. SSCHA on the three bcc metals, one panel each: the lowest free-energy-Hessian
    frequency (the default criterion, a bubble-level Hessian evaluated at the bcc reference)
    against temperature. SSCHA here runs on each MLIP's own energies, so it inherits that PES.
    Dashed lines with open markers are the (metal, model) pairs whose harmonic layer has no
    instability at all; a positive frequency there tests no thermal stabilisation."""
    metals = ["ti_bcc", "zr_bcc", "hf_bcc"]
    have = [s for s in metals if not df[(df["method"] == "sscha") & (df["system"] == s)].empty]
    if not have:
        return
    trivial = _trivial_bcc_pairs()
    # Sized for the 17.1 cm double column, so the text prints at its set size (not shrunk ~56%).
    with plt.rc_context({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8.5,
                         "xtick.labelsize": 8, "ytick.labelsize": 8}):
        fig, axes = plt.subplots(1, len(have), figsize=(6.7, 3.3), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, s in zip(axes, have):
            piv = (df[(df["method"] == "sscha") & (df["system"] == s)]
                   .pivot_table(index="temperature_K", columns="model", values="min_eff_freq_thz"))
            for m in [c for c in MODELS if c in piv.columns]:
                triv = (s, m) in trivial
                ax.plot(piv.index, piv[m], marker="o", ms=4, lw=1.3, color=COLOR[m],
                        ls="--" if triv else "-", mfc="white" if triv else COLOR[m])
            ax.axhline(0, color="k", lw=0.8, ls=":")
            ax.set_xticks([0, 200, 400, 600])
            ax.set_xlabel("Temperature (K)")
            ax.set_title(sys_label(s))
        axes[0].set_ylabel("SSCHA lowest free-energy\nHessian frequency (THz)")
        handles = [Line2D([], [], color=COLOR[m], marker="o", ms=4, label=NAME[m])
                   for m in MODELS]
        if trivial:
            handles.append(Line2D([], [], color="0.35", marker="o", ms=4, ls="--", mfc="white",
                                  label="no harmonic instability on this model's PES"))
        fig.legend(handles=handles, fontsize=7.5, loc="lower center", ncol=3,
                   frameon=False, bbox_to_anchor=(0.5, 0.0))
        fig.suptitle("SSCHA on bcc Ti, Zr and Hf: free-energy Hessian at the bcc reference",
                     fontsize=9.5)
        fig.tight_layout(rect=(0, 0.14, 1, 1))
        _save(fig, "fig_sscha_bcc")


def fig_softmode_heat():
    """Fig. 2. The soft-mode screen at the lowest ladder temperature (100 K): for every system
    and model, the symmetric-point curvature frequency, i.e. the signed curvature of the
    single-mode SCHA free energy at Q = 0, minimised over the screened modes (negative =
    imaginary). This is the screen's curvature observable; it is neither a harmonic frequency
    nor the screen's stability call, which is the free-energy comparison over the centroid.
    Boxed cells are the units the screen calls unstable (some mode condenses). For a system with
    no imaginary commensurate mode the cell holds the softest harmonic commensurate frequency."""
    sm = df[df["method"] == "softmode"]
    if sm.empty:
        return
    t0 = sm["temperature_K"].min()
    d = sm[sm["temperature_K"] == t0]
    rows = [s for s in _SPECS if s in set(d["system"])] + sorted(set(d["system"]) - set(_SPECS))
    cols = [c for c in MODELS if c in set(d["model"])]
    piv = d.pivot_table(index="system", columns="model", values="min_eff_freq_thz").loc[rows, cols]
    called_stable = (d.assign(ps=d["pred_stable"].astype(float))
                     .pivot_table(index="system", columns="model", values="ps").loc[rows, cols])
    borderline = A.borderline_systems()
    fig, ax = plt.subplots(figsize=(6.5, 7.4))
    # Symmetric log colour scale: linear within +/-1 THz so the sign of the near-zero cells reads
    # clearly, logarithmic out to +/-10 THz; the few larger values (C diamond, HfO2/CHGNet) sit at
    # the saturated end and their printed values give the true numbers.
    norm = SymLogNorm(linthresh=1.0, linscale=1.0, vmin=-10.0, vmax=10.0, base=10)
    cmap = plt.get_cmap("RdBu")                  # low (negative, imaginary) = red, high = blue
    im = ax.imshow(piv.values, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([NAME[c] for c in cols], rotation=35, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([sys_label(s) + ("*" if s in borderline else "") for s in rows], fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isfinite(v):
                continue
            dark = abs(float(norm(v)) - 0.5) > 0.36
            ax.text(j, i, f"{v:.2f}".replace("-", "−"), ha="center", va="center",
                    fontsize=6.5, color="white" if dark else "black")
            if called_stable.values[i, j] == 0:
                ax.add_patch(Rectangle((j - 0.44, i - 0.42), 0.88, 0.84, fill=False,
                                       ec="black", lw=1.3, zorder=3))
    ax.set_title(f"Soft-mode screen, {int(t0)} K: symmetric-point curvature")
    cb = fig.colorbar(im, ax=ax, extend="both", fraction=0.05, pad=0.03)
    cb.set_ticks([-10, -3, -1, 0, 1, 3, 10])
    cb.set_ticklabels(["−10", "−3", "−1", "0", "1", "3", "10"])
    cb.set_label("curvature frequency (THz); negative = imaginary\n"
                 "(red: negative, blue: positive)")
    note = f"Boxed: the screen calls the phase unstable at {int(t0)} K (a mode condenses)."
    if any(s in borderline for s in rows):
        note += "\n* borderline label, excluded from scoring."
    fig.text(0.02, 0.01, note, fontsize=7.5, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, "fig_softmode_heat")


def fig_method_agreement():
    """Fig. 4. Screen vs SSCHA on the bcc metals at the temperatures both ran: the screen's
    symmetric-point curvature frequency against the SSCHA lowest free-energy-Hessian frequency,
    both computed on the same MLIP energies. Points in the upper-right and lower-left quadrants
    agree in curvature sign. Reported on the panel: curvature-sign agreement and stability-call
    agreement as k/n with Wilson 95% intervals, and the Spearman rho of the magnitudes as a
    descriptive number (no test; the pairs cluster by system and model). Open markers are pairs
    with no harmonic instability on that model's PES, where agreement is trivial. A point outside
    the plotting window is drawn at its edge and labelled with its value from the ledger."""
    dfb = df[df["system"].str.contains("bcc")]
    m = A.method_agreement(dfb)
    if m.empty:
        return
    summ = A.method_agreement_summary(dfb)
    calls = dfb[dfb["method"] == "softmode"][KEY + ["pred_stable"]].merge(
        dfb[dfb["method"] == "sscha"][KEY + ["pred_stable"]], on=KEY, suffixes=("_scr", "_ss"))
    m = m.merge(calls, on=KEY)
    call_agree = m["pred_stable_scr"].astype(bool) == m["pred_stable_ss"].astype(bool)
    sign = S.rate_ci(int(m["agree"].sum()), len(m))
    call = S.rate_ci(int(call_agree.sum()), len(m))
    trivial = _trivial_bcc_pairs()
    temps = sorted(m["temperature_K"].unique())
    x, y = m["min_eff_freq_thz_softmode"], m["min_eff_freq_thz_sscha"]

    # Window on the bulk of the points; anything beyond it is drawn at the edge and labelled.
    lo, hi = -2.0, max(3.0, float(x.max()) + 1.6)   # right margin holds the legend
    ylo, yhi = min(-0.3, float(y.min()) - 0.3), max(2.5, float(y.max()) + 0.3)
    fig, ax = plt.subplots(figsize=(6.0, 5.6))
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.axvline(0, color="grey", lw=0.8, ls="--")
    for mod in [c for c in MODELS if c in set(m["model"])]:
        g = m[m["model"] == mod]
        triv = np.array([(s, mod) in trivial for s in g["system"]])
        gx = g["min_eff_freq_thz_softmode"].clip(lower=lo + 0.06)
        gy = g["min_eff_freq_thz_sscha"]
        off = (g["min_eff_freq_thz_softmode"] < lo).to_numpy()
        for sel, face in ((~triv & ~off, COLOR[mod]), (triv & ~off, "white")):
            ax.scatter(gx[sel], gy[sel], s=42, marker="o", facecolors=face,
                       edgecolors=COLOR[mod], linewidths=1.2, alpha=0.9, zorder=3)
        ax.scatter(gx[off], gy[off], s=60, marker="<", color=COLOR[mod], zorder=3)
        for (_, r), yy in zip(g[off].iterrows(), gy[off]):
            val = f"{r['min_eff_freq_thz_softmode']:+.1f}".replace("-", "\u2212")
            ax.annotate(f"{NAME[mod]}, {sys_label(r['system'])}, {int(r['temperature_K'])} K:\n"
                        f"screen {val} THz (off scale)",
                        xy=(lo + 0.06, yy), xytext=(lo + 0.25, yy - 0.35), fontsize=7,
                        color=COLOR[mod], arrowprops=dict(arrowstyle="-", color=COLOR[mod], lw=0.6))
    ax.set_xlim(lo, hi)
    ax.set_ylim(ylo, yhi)
    ax.text(hi - 0.05, yhi - 0.08, "curvature signs agree", ha="right", va="top", fontsize=7.5,
            color="0.4")
    ax.text(lo + 0.05, yhi - 0.08, "signs disagree", ha="left", va="top", fontsize=7.5,
            color="0.4")
    ax.text(0.98, 0.03,
            f"curvature-sign agreement {S.fmt_rate(sign)}\n"
            f"stability-call agreement {S.fmt_rate(call)}\n"
            f"Spearman \u03c1 = {summ.get('spearman_freq', float('nan')):.2f} (descriptive)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7"))
    ax.set_xlabel("screen: symmetric-point curvature frequency (THz)")
    ax.set_ylabel("SSCHA: lowest free-energy Hessian frequency (THz)")
    ax.set_title("bcc Ti, Zr, Hf at " + ", ".join(str(int(t)) for t in temps)
                 + f" K: screen vs SSCHA (n = {len(m)})")
    handles = [Line2D([], [], ls="", marker="o", ms=6, color=COLOR[c], label=NAME[c])
               for c in MODELS if c in set(m["model"])]
    handles.append(Line2D([], [], ls="", marker="o", ms=6, mfc="white", mec="0.35",
                          label="open: no harmonic instability"))
    ax.legend(handles=handles, fontsize=7.5, loc="upper right", bbox_to_anchor=(1.0, 0.93),
              framealpha=0.9)
    fig.tight_layout()
    _save(fig, "fig_method_agreement")
    print(f"  bcc: sign {S.fmt_rate(sign)}, call {S.fmt_rate(call)}, summary {summ}")


def fig_displacive_recall():
    """Fig. 5. Recall of the unstable cubic phase on the ferroelectric oxide perovskites
    (BaTiO3, KNbO3, PbTiO3) at T <= 300 K, below every transition temperature: the fraction of
    (system, model, T) units each method calls unstable, as k/n with a Wilson 95% interval.
    SSCHA units that blew up numerically (|f| > 50 THz) or returned no result are left out of its
    denominator and counted under the bar. Both methods see the same units, so the comparison is
    a paired one; its system-clustered test is in the text, and two marginal intervals are not
    that test."""
    r = A.displacive_recall(df)
    if r.empty:
        return
    fe = ["batio3_cubic", "knbo3_cubic", "pbtio3_cubic"]
    n_failed = None
    if os.path.exists(SSCHA_GRID):
        grid = pd.read_csv(SSCHA_GRID)
        grid = grid[grid["system"].isin(fe) & (grid["temperature_K"] <= 300.0)]
        ran = df[(df["method"] == "sscha") & df["system"].isin(fe) & (df["temperature_K"] <= 300.0)]
        n_failed = int(len(grid.merge(ran[KEY], on=KEY, how="left", indicator=True)
                           .query("_merge == 'left_only'")))
    label = {"softmode": "soft-mode\nscreen", "sscha": "SSCHA\n(default criterion)"}
    fig, ax = plt.subplots(figsize=(4.6, 4.9))
    note = ""
    for i, (_, row) in enumerate(r.iterrows()):
        rc = S.rate_ci(int(row["correct_unstable"]), int(row["n_valid"]))
        ax.bar(i, rc.p, width=0.55, color=C_A if row["method"] == "softmode" else C_B)
        ax.errorbar(i, rc.p, yerr=[[rc.p - rc.lo], [rc.hi - rc.p]], color="k", capsize=5, lw=1)
        ax.text(i, rc.hi + 0.025, f"{rc.k}/{rc.n}", ha="center", va="bottom", fontsize=9)
        if row["method"] == "sscha":
            parts = []
            if int(row["n_numerical_blowup"]):
                parts.append(f"{int(row['n_numerical_blowup'])} numerical blow-up "
                             "(|f| > 50 THz)")
            if n_failed:
                parts.append(f"{n_failed} failed run{'s' if n_failed != 1 else ''}")
            if parts:
                note = "SSCHA denominator excludes " + " and ".join(parts) + "."
    ax.set_xticks(range(len(r)))
    ax.set_xticklabels([label.get(mth, mth) for mth in r["method"]])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("recall: cubic phase called unstable")
    ax.set_title("FE oxide perovskites, T \u2264 300 K\n(below every T$_\\mathrm{c}$; "
                 "Wilson 95% intervals)", fontsize=10)
    if note:
        fig.text(0.02, 0.01, note, fontsize=7, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.035 if note else 0, 1, 1))
    _save(fig, "fig_displacive_recall")
    print(f"  recall {r.to_dict('records')}, SSCHA failed runs {n_failed}")


def fig_tolerance_sweep():
    """Fig. 1. Harmonic false-stable and false-unstable call counts against the imaginary
    tolerance (a call is 'stable' iff the minimum harmonic frequency >= -tol), summed over the
    five models on the scored systems. The default 0.1 THz sits between the strict regime
    (tol = 0, where finite-displacement noise near Gamma floods false-unstables) and the loose
    regime (inflated false-stables)."""
    s = A.harmonic_tolerance_sweep(df)
    if s.empty:
        return
    h = df[(df["method"] == "harmonic") & ~df["system"].isin(A.borderline_systems())]
    n_calls, n_mod, n_sys = int(s["n_calls"].iloc[0]), h["model"].nunique(), h["system"].nunique()
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(s["tol_THz"], s["false_stable"], "o-", color=C_B, label="false-stable")
    ax.plot(s["tol_THz"], s["false_unstable"], "s-", color=C_A, label="false-unstable")
    ax.axvline(0.1, color="k", lw=0.8, ls="--", label="default tolerance (0.1 THz)")
    ax.set_xlabel("imaginary tolerance |tol| (THz)")
    ax.set_ylabel(f"calls (of {n_calls}: {n_mod} models \u00d7 {n_sys} systems)")
    ax.set_title("Harmonic stability calls vs imaginary tolerance")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "fig_tolerance_sweep")
    print(f"  sweep {s.to_dict('records')}")


def _guardrail_stats(d: pd.DataFrame) -> dict:
    """Guardrail numbers for one model set, computed exactly as stats_hardening.guardrail_clustered
    does (same unit set, same row order, same resampling seed), so the intervals match the
    deposited results/stats_hardening.json."""
    g = A.h3_ensemble_guardrail(d, method="softmode")
    g = g[~g["system"].str.contains("bcc") & ~g["system"].isin(A.borderline_systems())]
    g = g.sort_values(["system", "T"]).reset_index(drop=True)
    err = (~g["consensus_correct"].astype(bool)).to_numpy()
    split, unan = g[g["disagreement"] > 0], g[g["disagreement"] == 0]
    out = {"n_units": len(g), "n_systems": g["system"].nunique(),
           "n_ties": int((g["stable_vote_frac"] == 0.5).sum()),
           "split": S.rate_ci(int((~split["consensus_correct"].astype(bool)).sum()), len(split)),
           "unan": S.rate_ci(int((~unan["consensus_correct"].astype(bool)).sum()), len(unan))}
    for key, col in (("vote", "disagreement"), ("freq", "freq_std_thz")):
        score = g[col].to_numpy(float)
        out[key] = {"auc": S.auc(score, err)} | S.cluster_bootstrap_auc(
            score, err, g["system"].to_numpy(), n_boot=N_RESAMPLE, seed=SEED)
    return out


def fig_ensemble_guardrail():
    """Fig. 6. The ensemble guardrail (H3) with and without ORB-v2: the majority-vote consensus
    finite-T error rate on units where the models split on the stable/unstable call against
    units where they are unanimous (k/n with Wilson 95% intervals), and the AUC of the vote split
    and of the cross-model frequency spread as predictors of consensus error, each with a
    cluster-bootstrap 95% interval that resamples whole systems. Without ORB-v2 a 2-2 tie is
    called 'stable' (analysis.h3_ensemble_guardrail)."""
    sets = [("all five models", df), ("without ORB-v2", df[df["model"] != "orb_v2"])]
    res = [(lab, _guardrail_stats(d)) for lab, d in sets]
    if not res[0][1]["n_units"]:
        return
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    w = 0.36
    for gi, (lab, r) in enumerate(res):
        for k, (key, col) in enumerate((("split", C_B), ("unan", C_A))):
            rc = r[key]
            xx = gi + (k - 0.5) * (w + 0.04)
            ax.bar(xx, rc.p, width=w, color=col)
            ax.errorbar(xx, rc.p, yerr=[[rc.p - rc.lo], [rc.hi - rc.p]], color="k", capsize=4, lw=1)
            ax.text(xx, rc.hi + 0.02, f"{rc.k}/{rc.n}", ha="center", va="bottom", fontsize=8.5)
        v, f = r["vote"], r["freq"]
        ax.text(gi, 1.06, f"vote-split AUC {v['auc']:.3f} [{v['ci_lo']:.3f}, {v['ci_hi']:.3f}]\n"
                f"frequency-spread AUC {f['auc']:.3f} [{f['ci_lo']:.3f}, {f['ci_hi']:.3f}]",
                ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(range(len(res)))
    ax.set_xticklabels([lab for lab, _ in res])
    ax.set_xlim(-0.6, len(res) - 0.4)
    ax.set_ylim(0, 1.28)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("consensus finite-T error rate")
    ax.legend(handles=[Patch(color=C_B, label="models split"), Patch(color=C_A, label="unanimous")],
              fontsize=8, loc="center right", bbox_to_anchor=(1.0, 0.62), framealpha=0.9)
    r0 = res[0][1]
    ax.set_title(f"Ensemble vote-split guardrail: {r0['n_units']} units, {r0['n_systems']} systems",
                 fontsize=10)
    ties = [f"{r['n_ties']} tied 2\u20132 unit{'s' if r['n_ties'] != 1 else ''} called stable"
            f" ({lab})" for lab, r in res if r["n_ties"]]
    fig.text(0.02, 0.01, "Error bars: Wilson 95%. AUC intervals: cluster bootstrap over systems"
             + (".\n" + "; ".join(ties) + "." if ties else "."), fontsize=7, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, "fig_ensemble_guardrail")
    for lab, r in res:
        print(f"  {lab}: split {S.fmt_rate(r['split'])}, unanimous {S.fmt_rate(r['unan'])}, "
              f"vote AUC {r['vote']['auc']:.3f} [{r['vote']['ci_lo']:.3f}, {r['vote']['ci_hi']:.3f}], "
              f"freq AUC {r['freq']['auc']:.3f} [{r['freq']['ci_lo']:.3f}, {r['freq']['ci_hi']:.3f}]")


if __name__ == "__main__":
    _check_numbering()
    fig_tolerance_sweep()
    fig_softmode_heat()
    fig_sscha_bcc()
    fig_method_agreement()
    fig_displacive_recall()
    fig_ensemble_guardrail()
    print("FIGURES_DONE")
