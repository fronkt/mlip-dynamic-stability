"""Generate the table-of-contents graphic for the RSC Advances submission.

RSC limit: at most 8 cm wide x 4 cm high. The graphic is exactly 8 cm x 4 cm. No number is typed
in here: both panels read results/ledger.parquet through mlip_dynstab.analysis.

  (a) The harmonic benchmark does not certify finite-temperature behaviour. On the matched set
      (15 non-bcc, non-borderline systems x 5 models), the units the harmonic layer gets right
      that the single-mode screen mis-calls, at each ladder temperature
      (analysis.h2_paired_summary, 'harm_ok_ft_bad'). The panel says "screen's" call because
      most of these sit in three systems the screen mis-calls for >= 4 of 5 models, so they
      measure the screen as much as the models; the existence claim is all the panel carries.
  (b) Neither does a default SSCHA cross-check. Its free-energy Hessian is evaluated at the
      symmetric reference, a local test: a double well that keeps a local minimum at Q = 0 but
      has deeper displaced minima reads as 'stable'. The curve is a schematic (no data). The
      script prints the measured count behind it for the record: the non-bcc units SSCHA calls
      stable against an unstable label on which the screen's own curvature at Q = 0 is also
      positive on the same MLIP energies (as in stats_hardening.criterion_blindness).

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
C_BAR, C_LOCAL, C_GLOBAL = "#0072B2", "#0072B2", "#D55E00"
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

# ---- panel (b) data: SSCHA false-stables whose screen curvature at Q = 0 is also positive
sm = df[df["method"] == "softmode"][KEY + ["min_eff_freq_thz", "gt_stable"]]
ss = df[df["method"] == "sscha"][KEY + ["pred_stable"]]
m = sm.merge(ss, on=KEY)
m = m[~m["system"].str.contains("bcc")]
fs = m[m["pred_stable"].astype(bool) & ~m["gt_stable"].astype(bool)]
n_fs, n_curv_pos = len(fs), int((fs["min_eff_freq_thz"] > 0).sum())
print(f"(a) harmonic-correct units mis-called at finite T: "
      + ", ".join(f"{int(t)} K {k}/{n_harm_ok}" for t, k in zip(temps, mis)))
print(f"(b) SSCHA false-stables with positive screen curvature at Q = 0: {n_curv_pos}/{n_fs}")

plt.rcParams.update({"font.size": 6, "axes.linewidth": 0.5,
                     "xtick.major.width": 0.5, "xtick.major.size": 2})
fig = plt.figure(figsize=(8 * CM, 4 * CM))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.28,
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

# ---- (b): illustrative double well F(Q) = Q^2 - 2.5 Q^4 + Q^6
ax1 = fig.add_subplot(gs[1])
q = np.linspace(-1.45, 1.45, 400)
F = q ** 2 - 2.5 * q ** 4 + q ** 6
ax1.plot(q, F, color="k", lw=0.9)
qq = np.linspace(-0.32, 0.32, 50)
ax1.plot(qq, qq ** 2, color=C_LOCAL, lw=1.6)             # curvature at Q = 0
qmin = np.sqrt((5 + np.sqrt(13)) / 6)
Fmin = qmin ** 2 - 2.5 * qmin ** 4 + qmin ** 6
ax1.plot([qmin, -qmin], [Fmin, Fmin], "o", color=C_GLOBAL, ms=2.6)
ax1.annotate("positive Hessian at Q = 0:\nSSCHA reads 'stable'", xy=(0, 0.02), xytext=(0, 0.62),
             ha="center", va="bottom", fontsize=5.3, color=C_LOCAL,
             arrowprops=dict(arrowstyle="-", color=C_LOCAL, lw=0.5))
ax1.text(0, Fmin + 0.02, "lower minima:\nphase condenses", ha="center", va="bottom",
         fontsize=5.3, color=C_GLOBAL, linespacing=0.95)
ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(Fmin - 0.25, 1.25)
ax1.set_xticks([])
ax1.set_yticks([])
for sp in ("top", "right", "left"):
    ax1.spines[sp].set_visible(False)
ax1.set_xlabel("order parameter Q", fontsize=5.5, labelpad=1)
ax1.set_title("Default SSCHA check\nis a local test", fontsize=6.2, pad=2)

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
