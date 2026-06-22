"""Regenerate the paper figures from results/ledger.parquet. Reproducible: all numbers come
from the ledger, independent of when/where units were computed.

Figures:
  fig_sscha_zr.png      - SSCHA multi-mode dynamic-stabilization curve for bcc-Zr, 5 MLIPs.
  fig_softmode_heat.png - per-system softmode min effective frequency at the lowest T, 5 MLIPs.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LEDGER = os.environ.get("LEDGER", "results/ledger.parquet")
OUT = os.environ.get("FIGDIR", "results/figures")
os.makedirs(OUT, exist_ok=True)
df = pd.read_parquet(LEDGER)
MODELS = ["mattersim", "sevennet0", "mace_mp0", "chgnet", "orb_v2"]


def fig_sscha_zr():
    d = df[(df["method"] == "sscha") & (df["system"] == "zr_bcc")]
    if d.empty:
        return
    piv = d.pivot_table(index="temperature_K", columns="model", values="min_eff_freq_thz")
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for m in [c for c in MODELS if c in piv.columns]:
        ax.plot(piv.index, piv[m], "o-", label=m)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("min free-energy Hessian freq (THz)")
    ax.set_title("bcc-Zr multi-mode SSCHA dynamic stabilization")
    ax.legend(fontsize=8, title="MLIP")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_sscha_zr.png", dpi=160); plt.close(fig)
    print(f"wrote {OUT}/fig_sscha_zr.png")


def fig_softmode_heat():
    sm = df[df["method"] == "softmode"]
    if sm.empty:
        return
    d = sm[sm["temperature_K"] == sm["temperature_K"].min()]
    piv = d.pivot_table(index="system", columns="model", values="min_eff_freq_thz")
    piv = piv[[c for c in MODELS if c in piv.columns]]
    fig, ax = plt.subplots(figsize=(6.5, 7))
    # clip the colour scale so a few extreme (float32/direct-model) outliers don't wash out the
    # structure; the printed cell values still report the true numbers.
    vmax = 10.0
    im = ax.imshow(np.clip(piv.values, -vmax, vmax), aspect="auto", cmap="RdBu",
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6.5)
    ax.set_title(f"softmode min eff. freq (THz) @ {int(d['temperature_K'].iloc[0])} K")
    fig.colorbar(im, ax=ax, label="THz (blue<0 unstable)")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_softmode_heat.png", dpi=160); plt.close(fig)
    print(f"wrote {OUT}/fig_softmode_heat.png")


if __name__ == "__main__":
    fig_sscha_zr()
    fig_softmode_heat()
    print("FIGURES_DONE")
