"""Generate the table-of-contents graphic for the RSC Advances submission.

RSC limit: at most 8 cm wide x 4 cm high. The graphic is exactly 8 cm x 4 cm. No number is typed
in here: both panels read results/ledger.parquet through mlip_dynstab.analysis.

  (a) The harmonic benchmark does not certify finite-temperature behaviour. On the matched set
      (15 non-bcc, non-borderline systems x 5 models), the units the harmonic layer gets right
      that the single-mode screen mis-calls, at each ladder temperature
      (analysis.h2_paired_summary, 'harm_ok_ft_bad'). The panel says "screen's" call because
      most of these sit in three systems the screen mis-calls for >= 4 of 5 models, so they
      measure the screen as much as the models; the existence claim is all the panel carries.
  (b) Neither does a default SSCHA cross-check, and the second revision's direct test shows why
      the two finite-temperature methods disagree (Section 3.3, ESI Table S28). Along the screen's
      own M-point mode of cubic BaTiO3 / MACE-MP-0 at 100 K: the static energy (gate.json), the
      screen's single-mode free energy (the registered values in
      tasks/preregistration-centroid-2026-10-21.md, per 2x1x2 cell, doubled) and the multi-mode
      SSCHA free energy at held centroids (summary.json, converged arms only). The screen's
      minimum is gone once every mode responds.

Writes toc_graphic.png and toc_graphic.tiff next to this file, both 600 dpi and flattened to RGB
(the TIFF LZW-compressed). Run from the repository root:
    python paper/toc_entry/make_toc_graphic.py
"""
import os
import sys
from io import BytesIO

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from mlip_dynstab import analysis as A

DPI = 600
CM = 1 / 2.54
C_BAR, C_GLOBAL = "#0072B2", "#D55E00"
KEY = ["system", "model", "temperature_K"]

df = A.canonical(pd.read_parquet(os.path.join(REPO, "results", "ledger.parquet")))

# ---- panel (a) data: harmonic-right, finite-T-wrong units per ladder temperature
temps = sorted(df[df["method"] == "softmode"]["temperature_K"].unique())
h2 = {t: A.h2_paired_summary(df, t) for t in temps}
mis = [h2[t]["concordance"]["harm_ok_ft_bad"] for t in temps]
n_harm_ok = {h2[t]["concordance"]["harm_ok_ft_ok"] + h2[t]["concordance"]["harm_ok_ft_bad"]
             for t in temps}
assert len(n_harm_ok) == 1, "harmonic-correct count should not depend on T"
n_harm_ok = n_harm_ok.pop()

# ---- panel (b) data: the R2 centroid scan along the screen's M mode (meV per 40 atoms)
import json
import re
R2 = os.path.join(REPO, "results", "revision", "r2_centroid")
gate = json.load(open(os.path.join(R2, "gate", "gate.json"), encoding="utf-8"))
cen = json.load(open(os.path.join(R2, "summary.json"), encoding="utf-8"))
st = [(0.0, 0.0)] + [(r["Q_ang"], r["dE_meV_per_40"]) for r in gate["m_pattern"]["static_energy_mapped_order"]]
mm = [(r["q0_ang"], r["dF_meV_per_40"]) for r in cen["m_scan_table"] if r["converged"]]
pre = open(os.path.join(REPO, "tasks", "preregistration-centroid-2026-10-21.md"), encoding="utf-8").read()
m_ = re.search(r"\+([0-9.]+) meV at Q₀ = 0\.025 Å and −([0-9.]+) meV at 0\.075 Å", pre)
sc = [(0.0, 0.0), (0.025, 2 * float(m_.group(1))), (0.075, -2 * float(m_.group(2)))]
print("(b) static", [(q, round(e, 1)) for q, e in st], "screen", sc, "multi-mode", [(q, round(f, 1)) for q, f in mm])

plt.rcParams.update({"font.size": 6, "axes.linewidth": 0.5,
                     "xtick.major.width": 0.5, "xtick.major.size": 2})
fig = plt.figure(figsize=(8 * CM, 4 * CM))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.42,
                      left=0.04, right=0.98, top=0.78, bottom=0.17)

# ---- (a)
ax0 = fig.add_subplot(gs[0])
x = np.arange(len(temps))
ax0.bar(x, mis, width=0.62, color=C_BAR)
for xi, k in zip(x, mis):
    ax0.text(xi, k + max(mis) * 0.03, str(k), ha="center", va="bottom", fontsize=6)
ax0.set_xticks(x)
ax0.set_xticklabels([f"{int(t)}" for t in temps], fontsize=5.5)
ax0.set_xlabel("temperature (K)", fontsize=5.5, labelpad=1)
ax0.set_ylim(0, max(mis) * 1.22)
ax0.set_yticks([])
for sp in ("top", "right", "left"):
    ax0.spines[sp].set_visible(False)
ax0.set_title("Harmonic check right, screen's\nfinite-T call wrong (of %d units)" % n_harm_ok,
              fontsize=6.2, pad=2, loc="center")

# ---- (b): free energy along the screen's M mode, three treatments of the same surface
ax1 = fig.add_subplot(gs[1])
C_ST, C_SC, C_MM = "0.55", C_BAR, C_GLOBAL
for pts, c, mk, lab in ((st, C_ST, "s", "static energy"), (sc, C_SC, "o", "screen (one mode)"),
                        (mm, C_MM, "D", "SSCHA (all modes)")):
    qx, ey = zip(*sorted(pts))
    ax1.plot(qx, ey, "-", marker=mk, color=c, lw=0.9, ms=2.2, label=lab)
ax1.axhline(0, color="k", lw=0.4)
ax1.set_xlim(-0.004, 0.106)
ax1.set_xticks([0, 0.05, 0.1])
ax1.set_xticklabels(["0", "0.05", "0.10"], fontsize=5.3)
ax1.tick_params(axis="y", labelsize=5.3, width=0.5, length=2, pad=1)
ax1.set_yticks([-100, 0, 100])
for sp in ("top", "right"):
    ax1.spines[sp].set_visible(False)
ax1.set_xlabel("displacement along M (Å)", fontsize=5.5, labelpad=1)
ax1.set_ylabel("meV / 40 atoms", fontsize=5.3, labelpad=0)
ax1.legend(loc="upper left", fontsize=4.8, frameon=False, handlelength=1.2, borderaxespad=0.1,
           labelspacing=0.2)
ax1.text(0.104, 6, "BaTiO₃, 100 K", ha="right", va="bottom", fontsize=5.0)
ax1.set_title("Screen's minimum vanishes\nwhen all modes respond", fontsize=6.2, pad=2)

buf = BytesIO()
fig.savefig(buf, format="png", dpi=DPI, facecolor="white")
plt.close(fig)
buf.seek(0)
with Image.open(buf) as im:
    im.load()
    rgb = Image.new("RGB", im.size, (255, 255, 255))
    rgb.paste(im, mask=im.getchannel("A") if "A" in im.getbands() else None)
rgb.save(os.path.join(HERE, "toc_graphic.png"), dpi=(DPI, DPI), optimize=True)
rgb.save(os.path.join(HERE, "toc_graphic.tiff"), dpi=(DPI, DPI), compression="tiff_lzw")
w_cm, h_cm = rgb.size[0] / DPI * 2.54, rgb.size[1] / DPI * 2.54
print(f"wrote toc_graphic.png/.tiff: {rgb.size[0]}x{rgb.size[1]} px at {DPI} dpi = "
      f"{w_cm:.2f} x {h_cm:.2f} cm, RGB")
