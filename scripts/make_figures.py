"""Regenerate the paper figures from results/ledger.parquet. Every plotted number is computed
here from the ledger (through ``analysis.canonical``), independent of when or where a unit ran.

Figures, in the order the manuscript first cites them:
  Fig. 1  fig_tolerance_sweep     harmonic false-stable / false-unstable calls vs imaginary tolerance
  Fig. 2  fig_harmonic_heat       harmonic layer: minimum phonon frequency per unit, with its call
  Fig. 3  fig_sscha_bcc           SSCHA lowest free-energy-Hessian frequency vs T, bcc Ti/Zr/Hf
  Fig. 4  fig_method_agreement    SSCHA Hessian frequency vs the screen's call on the bcc metals
  Fig. 5  fig_displacive_recall   recall of the unstable cubic phase, FE perovskites, T <= 300 K
  Fig. 6  fig_ensemble_guardrail  consensus error on split-vote vs unanimous units, +/- ORB-v2

Each figure is written as a 600 dpi PNG (the copy embedded in the DOCX) and a 600 dpi LZW TIFF,
both flattened to RGB on white, plus a numbered upload copy results/figures/upload/FigN.tif.
The numbering is checked against the figure links in paper/manuscript.md before anything is
written. Run from the repository root: python scripts/make_figures.py

fig_softmode_heat (the screen's 100 K symmetric-point curvature map, formerly Fig. 2) is retired
and no longer called: that curvature is positive by construction for a single even mode, so the
map showed only numerical noise in its sign (scripts/curvature_identity_check.py). The function
is kept for reference; its last outputs results/figures/fig_softmode_heat.{png,tiff} were left in
place and are not part of the paper.
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
# The converged-recipe SSCHA grid (ESI Table S22) and its comparison with the production claims.
CONV_SUMMARY = "results/revision/sscha_converged_grid/summary.csv"
GRID_COMPARE = "results/revision/grid_compare.json"
E3_SUMMARY = "results/revision/e3_replicates/summary.csv"   # pre-registered replicates (Table S24)
os.makedirs(OUT, exist_ok=True)
from mlip_dynstab import DEFAULT_IMAG_TOL_THZ
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
FIG_NUMBER = {"fig_tolerance_sweep": 1, "fig_harmonic_heat": 2, "fig_sscha_bcc": 3,
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
    conv = None
    if os.path.exists(CONV_SUMMARY):
        conv = pd.read_csv(CONV_SUMMARY)
        conv = conv[(conv["start"] == "A") & (conv["family"] == "bcc") & (conv["status"] == "ok")
                    & (conv["converged"].astype(str) == "True")].copy()
        # A converged unit whose pre-registered replicates disagree on the call (ESI Table S24) is
        # unresolved and is not drawn as a converged value: it gets its own marker (Ti/ORB-v2/600 K).
        unres = set()
        if os.path.exists(E3_SUMMARY):
            e3 = pd.read_csv(E3_SUMMARY)
            unres = set(e3.loc[e3["verdict"] == "unresolved", "unit_tag"])
        conv["unresolved"] = conv["unit_tag"].isin(unres)
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
            if conv is not None:
                cs = conv[conv["system"] == s]
                for m in [c for c in MODELS if c in set(cs["model"])]:
                    g = cs[cs["model"] == m].sort_values("T_K")
                    gr, gu = g[~g["unresolved"]], g[g["unresolved"]]
                    ax.scatter(gr["T_K"], gr["hessian_min_thz"], marker="s", s=22,
                               facecolors="none", edgecolors=COLOR[m], linewidths=1.1, zorder=4)
                    ax.scatter(gu["T_K"], gu["hessian_min_thz"], marker="x", s=26,
                               color=COLOR[m], linewidths=1.2, zorder=4)
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
        handles.append(Line2D([], [], color="0.35", marker="o", ms=4,
                              label="production recipe, 2×2×2"))
        if conv is not None:
            handles.append(Line2D([], [], ls="", marker="s", ms=4, mfc="none", mec="0.35",
                                  label="converged recipe, 3×3×3"))
            if conv["unresolved"].any():
                handles.append(Line2D([], [], ls="", marker="x", ms=4, color="0.35",
                                      label="converged, unresolved by replicates"))
        fig.legend(handles=handles, fontsize=7.5, loc="lower center", ncol=3,
                   frameon=False, bbox_to_anchor=(0.5, 0.0))
        fig.suptitle("SSCHA on bcc Ti, Zr and Hf: free-energy Hessian at the bcc reference\n"
                     "(lines: production recipe, 2×2×2; squares: converged recipe, "
                     "3×3×3)",
                     fontsize=9.5)
        fig.tight_layout(rect=(0, 0.14, 1, 0.97))
        _save(fig, "fig_sscha_bcc")


def fig_harmonic_heat():
    """Fig. 2. The harmonic layer (§2.3, §3.1): for every system and model, the minimum phonon
    frequency over the Gamma-centred 12x12x12 mesh interpolated from 2x2x2 finite-displacement
    force constants (negative = imaginary). The minimum includes the acoustic branch at Gamma,
    so a system with no instability reads numerical zero (printed unsigned, 0.00) and no cell is
    positive beyond it; the diverging scale is centred at 0 and only its negative half is drawn on
    the colour bar.
    Boxed cells are the harmonic calls 'unstable' at the production tolerance (minimum below
    -0.1 THz). FS / FU mark calls that disagree with the reference label (false-stable /
    false-unstable). The borderline KTaO3 is shown but not scored, so it carries no FS / FU mark.
    Rows and columns keep the order and labels of the retired screen-curvature map."""
    h = df[df["method"] == "harmonic"]
    if h.empty:
        return
    if h.groupby(["system", "model"]).size().max() != 1:
        raise SystemExit("[figures] more than one canonical harmonic row per (system, model)")
    tol = abs(float(DEFAULT_IMAG_TOL_THZ))
    stable = h["pred_stable"].astype(bool)
    # The boxes are the ledger's calls; refuse to draw them if those are not the calls at the
    # production tolerance that the caption describes.
    if not (stable == (h["min_freq_thz"] >= -tol)).all():
        raise SystemExit(f"[figures] harmonic pred_stable is not min_freq >= -{tol} THz")
    h = h.assign(ps=stable.astype(float), gt=h["gt_stable"].astype(bool).astype(float))
    rows = [s for s in _SPECS if s in set(h["system"])] + sorted(set(h["system"]) - set(_SPECS))
    cols = [c for c in MODELS if c in set(h["model"])]
    freq = h.pivot_table(index="system", columns="model", values="min_freq_thz").loc[rows, cols]
    pred = h.pivot_table(index="system", columns="model", values="ps").loc[rows, cols]
    label = h.pivot_table(index="system", columns="model", values="gt").loc[rows, cols]
    borderline = A.borderline_systems()

    # Diverging scale centred at 0, linear within +/- the tolerance and logarithmic out to
    # +/-10 THz, so the calls just past the tolerance (about -0.2 to -0.3 THz) are already clearly
    # red while the acoustic zero stays white. HfO2/CHGNet (-10.6 THz) sits at the saturated end;
    # its printed value gives the true number.
    vlim = 10.0
    norm = SymLogNorm(linthresh=tol, linscale=1.0, vmin=-vlim, vmax=vlim, base=10)
    cmap = plt.get_cmap("RdBu")                  # negative (imaginary) = red, 0 = white
    fig, ax = plt.subplots(figsize=(6.5, 7.6))
    im = ax.imshow(freq.values, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([NAME[c] for c in cols], rotation=35, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([sys_label(s) + ("*" if s in borderline else "") for s in rows], fontsize=8)
    n_fs = n_fu = 0
    for i, s in enumerate(rows):
        for j in range(len(cols)):
            v = freq.values[i, j]
            if not np.isfinite(v):
                continue
            ink = "white" if abs(float(norm(v)) - 0.5) > 0.36 else "black"
            # A value that rounds to zero is the acoustic zero at Gamma (|v| < 1e-5 THz in the
            # ledger); its sign is numerical, so it prints as the unsigned 0.00 the caption quotes.
            txt = f"{v:.2f}"
            if float(txt) == 0.0:
                txt = "0.00"
            ax.text(j, i, txt.replace("-", "−"), ha="center", va="center",
                    fontsize=6.5, color=ink)
            called_stable = pred.values[i, j] == 1
            if not called_stable:
                ax.add_patch(Rectangle((j - 0.44, i - 0.42), 0.88, 0.84, fill=False,
                                       ec="black", lw=1.3, zorder=3))
            if s not in borderline and called_stable != (label.values[i, j] == 1):
                tag = "FS" if called_stable else "FU"
                n_fs += tag == "FS"
                n_fu += tag == "FU"
                ax.text(j + 0.33, i, tag, ha="center", va="center", fontsize=5.5,
                        fontweight="bold", color=ink, zorder=4)
    ax.set_title("Harmonic layer: minimum phonon frequency\n"
                 "(2×2×2 force constants, 12×12×12 mesh; §2.3)",
                 fontsize=10.5)
    cb = fig.colorbar(im, ax=ax, extend="min", fraction=0.05, pad=0.03)
    cb.ax.set_ylim(-vlim, 0.0)                   # the minimum cannot sit above the acoustic zero
    cb.set_ticks([-10, -1, -tol, 0])
    cb.set_ticklabels(["−10", "−1", f"−{tol:g}", "0"])
    cb.ax.axhline(-tol, color="black", lw=0.9, ls="--")
    cb.set_label("minimum harmonic frequency (THz)\n"
                 f"negative = imaginary (red); dashed: −{tol:g} THz tolerance")
    note = (f"Boxed: harmonic call unstable (minimum below −{tol:g} THz). "
            "FS / FU: the call disagrees with the reference label\n"
            "(false-stable / false-unstable).")
    if any(s in borderline for s in rows):
        note += " * borderline label, not scored (no FS / FU mark)."
    fig.text(0.02, 0.01, note, fontsize=7.5, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, "fig_harmonic_heat")
    print(f"  harmonic heat map: {len(rows)} systems x {len(cols)} models, "
          f"{int((pred.values == 0).sum())} boxed, {n_fs} FS, {n_fu} FU at tol {tol} THz")


def fig_softmode_heat():
    """RETIRED 2026-09-27, not called and not in the paper (kept for reference; it writes no
    numbered upload copy because it is no longer in FIG_NUMBER). The symmetric-point curvature it
    maps equals the trial stiffness M*Omega^2 for a single even mode, so it is positive by
    construction and every negative cell was numerical (scripts/curvature_identity_check.py).

    Formerly Fig. 2. The soft-mode screen at the lowest ladder temperature (100 K): for every system
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
    """Fig. 4. Screen vs SSCHA on the bcc metals at the temperatures both ran: the production
    SSCHA lowest free-energy-Hessian frequency (2x2x2) against the screen's stability CALL, both on
    the same MLIP energies. The x-axis is the call, not the screen's symmetric-point curvature:
    for a single even mode that curvature is positive by construction and its negative values are
    numerical (scripts/curvature_identity_check.py), so it is not plotted against SSCHA (audit
    2026-10-09, M7). Points in the screen-stable column above the SSCHA tolerance, and in the
    screen-unstable column below it, agree on the call. Reported on the panel: stability-call
    agreement as k/n with a Wilson 95% interval. Open markers are pairs with no harmonic
    instability on that model's PES, where agreement is trivial."""
    dfb = df[df["system"].str.contains("bcc")]
    m = A.method_agreement(dfb)
    if m.empty:
        return
    calls = dfb[dfb["method"] == "softmode"][KEY + ["pred_stable"]].merge(
        dfb[dfb["method"] == "sscha"][KEY + ["pred_stable"]], on=KEY, suffixes=("_scr", "_ss"))
    m = m.merge(calls, on=KEY)
    call_agree = m["pred_stable_scr"].astype(bool) == m["pred_stable_ss"].astype(bool)
    call = S.rate_ci(int(call_agree.sum()), len(m))
    trivial = _trivial_bcc_pairs()
    temps = sorted(m["temperature_K"].unique())
    y = m["min_eff_freq_thz_sscha"]
    mods = [c for c in MODELS if c in set(m["model"])]
    ylo, yhi = min(-0.4, float(y.min()) - 0.3), max(2.5, float(y.max()) + 0.3)
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.axhline(0, color="grey", lw=0.8, ls=":")
    ax.axhline(DEFAULT_IMAG_TOL_THZ, color="grey", lw=0.8, ls="--")   # -0.1 THz
    for i, mod in enumerate(mods):
        g = m[m["model"] == mod].reset_index(drop=True)
        for col, stable in ((0, False), (1, True)):
            gg = g[g["pred_stable_scr"].astype(bool) == stable]
            if gg.empty:
                continue
            # one sub-column per model, points spread across it in (system, T) order
            base = col + (i - (len(mods) - 1) / 2) * 0.15
            xs = base + np.linspace(-0.045, 0.045, len(gg)) if len(gg) > 1 else np.array([base])
            triv = np.array([(s, mod) in trivial for s in gg["system"]])
            for sel, face in ((~triv, COLOR[mod]), (triv, "white")):
                ax.scatter(xs[sel], gg["min_eff_freq_thz_sscha"].to_numpy()[sel], s=36, marker="o",
                           facecolors=face, edgecolors=COLOR[mod], linewidths=1.1, alpha=0.9,
                           zorder=3)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(ylo, yhi)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["screen calls unstable", "screen calls stable"])
    ax.text(0.98, 0.03, f"stability-call agreement {S.fmt_rate(call)}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7"))
    ax.set_xlabel("soft-mode screen: stability call (free-energy comparison)")
    ax.set_ylabel("SSCHA (production recipe, 2×2×2):\nlowest free-energy Hessian frequency (THz)")
    ax.set_title("bcc Ti, Zr, Hf at " + ", ".join(str(int(t)) for t in temps)
                 + f" K: screen call vs SSCHA (n = {len(m)})")
    handles = [Line2D([], [], ls="", marker="o", ms=6, color=COLOR[c], label=NAME[c]) for c in mods]
    handles.append(Line2D([], [], ls="", marker="o", ms=6, mfc="white", mec="0.35",
                          label="open: no harmonic instability"))
    handles.append(Line2D([], [], color="grey", lw=0.8, ls="--", label="SSCHA tolerance (−0.1 THz)"))
    ax.legend(handles=handles, fontsize=7.5, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    _save(fig, "fig_method_agreement")
    print(f"  bcc: call {S.fmt_rate(call)}")


def fig_displacive_recall():
    """Fig. 5. Recall of the unstable cubic phase on the ferroelectric oxide perovskites
    (BaTiO3, KNbO3, PbTiO3) at T <= 300 K, below every transition temperature: the fraction of
    (system, model, T) units each method calls unstable, as k/n with a Wilson 95% interval.
    SSCHA units that blew up numerically (|f| > 50 THz) or returned no result are left out of its
    denominator and counted under the bar. Both methods see the same units, so the comparison is
    a paired one; its system-clustered test is in the text, and two marginal intervals are not
    that test. The first two bars are the screen and converged SSCHA on the units where converged
    SSCHA returned a value (ESI Table S22); the hatched bar is the production recipe."""
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
    prod = {row["method"]: row for _, row in r.iterrows()}
    # Bars: the screen and converged SSCHA on the same units (the converged-recipe grid,
    # grid_compare.json, converged_only / converged_matched), then production SSCHA for reference.
    bars = []
    conv_note = ""
    if os.path.exists(GRID_COMPARE) and os.path.exists(CONV_SUMMARY):
        import json
        gc = json.load(open(GRID_COMPARE, encoding="utf-8"))
        cii = gc["variants"]["converged_only"]["converged_matched"]["ii"]["all_models"]
        cs = pd.read_csv(CONV_SUMMARY)
        cs = cs[(cs["start"] == "A") & cs["system"].isin(fe) & (cs["T_K"] <= 300.0)]
        n_cf = int((cs["status"] == "failed").sum())
        n_cu = int(((cs["status"] == "ok") & (cs["converged"].astype(str) != "True")).sum())
        # The pre-registered replicates (results/revision/e3_replicates/): a unit whose call
        # differs between replicates is unresolved and is left out, not resolved by majority.
        unres = set()
        if os.path.exists(E3_SUMMARY):
            e3 = pd.read_csv(E3_SUMMARY)
            unres = set(e3.loc[e3["verdict"] == "unresolved", "unit_tag"])
        u = cs[(cs["status"] == "ok") & (cs["converged"].astype(str) == "True")
               & ~cs["gt_stable"].astype(bool) & ~cs["unit_tag"].isin(unres)]
        n_ur = int(cs["unit_tag"].isin(unres).sum())
        k_scr = int((~u["screen_stable_call"].astype(bool)).sum())
        k_ss = int((u["stable_call"].astype(str) != "True").sum())
        check_a = (int((~cs[(cs["status"] == "ok") & (cs["converged"].astype(str) == "True")]
                        ["stable_call"].astype(str).eq("True")).sum()), cii["sscha"]["recall"]["k"])
        assert check_a[0] == check_a[1], f"Fig. 5: start-A recall {check_a} disagrees with grid_compare"
        bars.append(("soft-mode screen\n(same units)", k_scr, len(u), C_A, None))
        bars.append(("SSCHA, converged\n(default criterion)", k_ss, len(u), C_B, None))
        conv_note = (f"Converged SSCHA: {len(cs)} units, {n_cf} failed, {n_cu} did not converge and "
                     f"{n_ur} are unresolved between\n"
                     "pre-registered replicates; the screen is scored "
                     f"on the same {len(u)} (start A alone: {cii['softmode']['recall']['k']}/"
                     f"{cii['softmode']['recall']['n']} against {cii['sscha']['recall']['k']}/"
                     f"{cii['sscha']['recall']['n']}).")
    ps = prod["sscha"]
    bars.append(("SSCHA, production\n(not converged)", int(ps["correct_unstable"]),
                 int(ps["n_valid"]), "0.75", "//"))
    parts = []
    if int(ps["n_numerical_blowup"]):
        parts.append(f"{int(ps['n_numerical_blowup'])} numerical blow-up (|f| > 50 THz)")
    if n_failed:
        parts.append(f"{n_failed} failed run{'s' if n_failed != 1 else ''}")
    note = conv_note
    if parts:
        note += ("\n" if note else "") + "Production SSCHA excludes " + " and ".join(parts) + "."
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    for i, (lab, k, n, col, hatch) in enumerate(bars):
        rc = S.rate_ci(int(k), int(n))
        ax.bar(i, rc.p, width=0.55, color=col, hatch=hatch, edgecolor="0.3" if hatch else col)
        ax.errorbar(i, rc.p, yerr=[[rc.p - rc.lo], [rc.hi - rc.p]], color="k", capsize=5, lw=1)
        ax.text(i, rc.hi + 0.025, f"{rc.k}/{rc.n}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b[0] for b in bars], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("recall: cubic phase called unstable")
    ax.set_title("FE oxide perovskites, T ≤ 300 K\n(below every T$_\\mathrm{c}$; "
                 "Wilson 95% intervals)", fontsize=10)
    if note:
        fig.text(0.02, 0.01, note, fontsize=6.5, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.07 if note else 0, 1, 1))
    _save(fig, "fig_displacive_recall")
    print(f"  bars {[(b[0].replace(chr(10), ' '), b[1], b[2]) for b in bars]}, "
          f"production SSCHA failed runs {n_failed}")


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
    fig_harmonic_heat()
    fig_sscha_bcc()
    fig_method_agreement()
    fig_displacive_recall()
    fig_ensemble_guardrail()
    print("FIGURES_DONE")
