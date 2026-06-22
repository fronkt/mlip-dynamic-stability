"""Regenerate the paper figures from results/ledger.parquet. Reproducible: all numbers come
from the ledger, independent of when/where units were computed.

Figures:
  fig_sscha_zr.png      - SSCHA multi-mode dynamic-stabilization curve for bcc-Zr, 5 MLIPs.
  fig_softmode_heat.png - per-system softmode min effective frequency at the lowest T, 5 MLIPs.
"""
from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


def fig_sscha_bcc():
    """Multi-mode SSCHA dynamic-stabilization curves for the three bcc metals, one panel each."""
    metals = [("ti_bcc", "bcc-Ti"), ("zr_bcc", "bcc-Zr"), ("hf_bcc", "bcc-Hf")]
    have = [(s, t) for s, t in metals if not df[(df["method"] == "sscha") & (df["system"] == s)].empty]
    if not have:
        return
    fig, axes = plt.subplots(1, len(have), figsize=(4.0 * len(have), 4.2), sharey=True)
    if len(have) == 1:
        axes = [axes]
    for ax, (s, title) in zip(axes, have):
        piv = (df[(df["method"] == "sscha") & (df["system"] == s)]
               .pivot_table(index="temperature_K", columns="model", values="min_eff_freq_thz"))
        for m in [c for c in MODELS if c in piv.columns]:
            ax.plot(piv.index, piv[m], "o-", label=m)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("Temperature (K)"); ax.set_title(title)
    axes[0].set_ylabel("min free-energy Hessian freq (THz)")
    axes[-1].legend(fontsize=8, title="MLIP")
    fig.suptitle("Multi-mode SSCHA dynamic stabilization of bcc Ti/Zr/Hf")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_sscha_bcc.png", dpi=160); plt.close(fig)
    print(f"wrote {OUT}/fig_sscha_bcc.png")


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


def fig_method_agreement():
    """VALIDATION figure: on the bcc metals -- where SSCHA is numerically clean -- the cheap
    single-mode softmode minimum frequency tracks the gold-standard multi-mode SSCHA free-energy
    Hessian minimum (points along y=x). This is the positive cross-validation that justifies
    softmode as the screening method. (Perovskite SSCHA is excluded here because it is unreliable
    on deep displacive instabilities -- see fig_displacive_recall.)"""
    from mlip_dynstab.analysis import method_agreement, method_agreement_summary
    dfb = df[df["system"].str.contains("bcc")]
    m = method_agreement(dfb)
    if m.empty:
        return
    summ = method_agreement_summary(dfb)
    fig, ax = plt.subplots(figsize=(6.0, 5.8))
    for mod in [c for c in MODELS if c in set(m["model"])]:
        g = m[m["model"] == mod]
        ax.scatter(g["min_eff_freq_thz_softmode"], g["min_eff_freq_thz_sscha"], s=42,
                   alpha=0.85, label=mod)
    # readable window: most points sit in [-4,3]; orb_v2's float32 over-softening drives Ti/Hf
    # softmode to ~-35 THz, annotated off-scale rather than allowed to squash the cluster.
    lo, hi = -4.0, 3.5
    ax.plot([lo, hi], [lo, hi], "k-", lw=0.7, alpha=0.5, zorder=0)
    ax.axhline(0, color="grey", lw=0.7, ls="--"); ax.axvline(0, color="grey", lw=0.7, ls="--")
    ax.set_xlim(lo, hi); ax.set_ylim(-0.8, 3.5)
    n_off = int((m["min_eff_freq_thz_softmode"] < lo).sum())
    if n_off:
        ax.annotate(f"<- {n_off} orb_v2 Ti/Hf pts: softmode ~ -35 THz (float32)",
                    xy=(lo + 0.1, 1.3), fontsize=7.0, color="purple")
    ax.set_xlabel("softmode min eff. freq (THz)")
    ax.set_ylabel("SSCHA min free-energy Hessian freq (THz)")
    ax.set_title(f"bcc: softmode vs SSCHA  (ρ={summ.get('spearman_freq','?')}, "
                 f"sign agree {summ.get('sign_agreement','?')}, n={summ.get('n_paired',0)})")
    ax.legend(fontsize=8, title="MLIP", loc="lower right")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_method_agreement.png", dpi=160); plt.close(fig)
    print(f"wrote {OUT}/fig_method_agreement.png  (bcc {summ})")


def fig_displacive_recall():
    """CAUTIONARY figure: on the ferroelectric perovskites at T<=300 K (cubic phase definitively
    unstable, far below every Tc), the fraction of model units each method correctly calls
    unstable. Softmode catches the displacive instability; multi-mode SSCHA (v4=False) collapses
    to the cubic minimum and reports false-stable. This is why softmode, not SSCHA, is the
    perovskite headline method."""
    from mlip_dynstab.analysis import displacive_recall
    r = displacive_recall(df)
    if r.empty:
        return
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    bars = ax.bar(r["method"], r["recall_unstable"], color=["#2c7fb8", "#d95f0e"], width=0.6)
    for b, (_, row) in zip(bars, r.iterrows()):
        ax.text(b.get_x() + b.get_width() / 2, row["recall_unstable"] + 0.02,
                f"{row['correct_unstable']}/{row['n_valid']}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.05); ax.set_ylabel("recall: cubic correctly called unstable")
    ax.set_title("FE perovskites, T≤300 K\n(cubic definitively unstable)")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_displacive_recall.png", dpi=160); plt.close(fig)
    print(f"wrote {OUT}/fig_displacive_recall.png  ({r.to_dict('records')})")


def fig_ensemble_guardrail():
    """H3: cross-model disagreement as a guardrail. Consensus finite-T error rate on units where
    the five MLIPs split on the stable/unstable call vs unanimous units; the binary vote split is
    a useful predictor of consensus error (AUC annotated) where the continuous frequency spread
    is not."""
    from mlip_dynstab.analysis import h3_guardrail_summary
    s = h3_guardrail_summary(df)
    if not s:
        return
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    labels = [f"split vote\n(n={s['n_split']})", f"unanimous\n(n={s['n_unanimous']})"]
    vals = [s["split_vote_error_rate"], s["unanimous_error_rate"]]
    bars = ax.bar(labels, vals, color=["#d95f0e", "#2c7fb8"], width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=10)
    ax.set_ylim(0, max(vals) * 1.25 + 0.05); ax.set_ylabel("consensus finite-T error rate")
    ax.set_title(f"Ensemble disagreement guardrail (H3)\nvote-split AUC {s['auc_vote_disagreement']}, "
                 f"freq-std AUC {s['auc_freq_std']}")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_ensemble_guardrail.png", dpi=160); plt.close(fig)
    print(f"wrote {OUT}/fig_ensemble_guardrail.png  ({s})")


if __name__ == "__main__":
    fig_sscha_bcc()
    fig_softmode_heat()
    fig_method_agreement()
    fig_displacive_recall()
    fig_ensemble_guardrail()
    print("FIGURES_DONE")
