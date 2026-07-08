"""Generate the RSC Digital Discovery graphical-abstract/TOC entry.
Two panels: (a) harmonic accuracy does not imply finite-T accuracy -- MatterSim
and SevenNet-0 lead at 0K but are not the finite-T leaders, (b) the cheap
soft-mode screen recovers the displacive instability that multi-mode SSCHA
false-stabilises. Numbers from results/ledger.parquet via
scripts.mlip_dynstab.analysis.displacive_recall. Sized to RSC's spec
(<=8cm x 4cm), exported at 600dpi TIFF plus a PNG preview.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CM = 1 / 2.54
fig = plt.figure(figsize=(8 * CM, 4 * CM))
gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.5,
                       left=0.04, right=0.98, top=0.80, bottom=0.20)

# --- Panel (a): harmonic pass, finite-T fail schematic ---
ax0 = fig.add_subplot(gs[0])
ax0.axis("off")
ax0.set_xlim(0, 1); ax0.set_ylim(0, 1)
ax0.text(0.5, 0.98, "harmonic (0K)\naccuracy is not sufficient",
          ha="center", va="top", fontsize=5.2, weight="bold")

models = ["MatterSim", "SevenNet-0"]
y0 = 0.66
for i, m in enumerate(models):
    y = y0 - i * 0.30
    ax0.text(0.03, y, m, fontsize=5.0, va="center")
    ax0.text(0.62, y, "0K: #1", fontsize=5.0, color="#1b9e77", va="center")
    ax0.text(0.85, y, "finite-T:\nnot #1", fontsize=4.4, color="#d95f02", va="center",
             ha="left", linespacing=1.0)

ax0.plot([0, 1], [0.04, 0.04], lw=0.5, color="0.6", transform=ax0.transAxes)
ax0.text(0.5, -0.10, "harmonic leaders $\\neq$ finite-T leaders",
          ha="center", va="top", fontsize=4.6, style="italic",
          transform=ax0.transAxes)

# --- Panel (b): displacive recall bars (softmode vs SSCHA) ---
ax1 = fig.add_subplot(gs[1])
labels = ["soft-mode\nscreen", "multi-mode\nSSCHA"]
vals = [0.767, 0.233]
colors = ["#1b9e77", "#d95f02"]
bars = ax1.bar(range(2), vals, color=colors, width=0.55)
for b, v in zip(bars, vals):
    ax1.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}",
              ha="center", va="bottom", fontsize=5.4)
ax1.set_xticks(range(2)); ax1.set_xticklabels(labels, fontsize=4.8)
ax1.set_ylim(0, 1.05)
ax1.set_yticks([])
for spine in ("top", "right", "left"):
    ax1.spines[spine].set_visible(False)
ax1.set_title("recall: cubic perovskite\ncorrectly called unstable", fontsize=5.0, pad=2)

for out, dpi in [("toc_graphic.png", 300), ("toc_graphic.tiff", 600)]:
    kwargs = {"dpi": dpi}
    if out.endswith(".tiff"):
        kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
    fig.savefig(out, **kwargs)
plt.close(fig)
print("done")
