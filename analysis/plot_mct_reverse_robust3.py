"""Summarize the three-parameter off-grid development gate."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent / "mct_feedback_audit" / "reverse_robust3"


def main() -> None:
    scored = json.loads((ROOT / "twin_score.json").read_text(encoding="utf-8"))
    fits = json.loads((ROOT / "inverse_diagnostics.json").read_text(encoding="utf-8"))
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    ax = axs[0, 0]
    for mode, marker, color in (("clean_low", "x", "#dd8452"),
                                ("clean_lowmid", "o", "#2b7a78")):
        rows = [r for r in fits if r["mode"] == mode]
        ax.scatter([r["true_mct_gain"] for r in rows],
                   [r["mct_gain"] for r in rows], marker=marker, color=color,
                   s=33, label=mode.replace("clean_", "").replace("lowmid", "low + mid"))
    ax.plot([.15, 1.35], [.15, 1.35], "k--", lw=1)
    ax.set(xlabel="True MCT gain", ylabel="Inferred MCT gain", title="Clean calibration")
    ax.grid(alpha=.2)
    ax.legend(frameon=False)

    names = ("Low twin", "Low+mid ridge", "Low+mid RBF", "Low+mid twin")
    keys = ("native_clean_low_rep0", "lowmid_ridge_clean", "lowmid_RBF_clean",
            "native_clean_lowmid_rep0")
    colors = ("#d17a66", "#8c9aa5", "#3995a5", "#2b7a78")
    for ax, field, ylabel in ((axs[0, 1], "APD_MAE_ms", "High-load APD MAE (ms)"),
                              (axs[1, 0], "Ca_peak_MAE_uM", "High-load Ca peak MAE (µM)")):
        vals = [scored["scores"][k][field] for k in keys]
        ax.barh(names[::-1], vals[::-1], color=colors[::-1])
        ax.set_xlabel(ylabel)
        ax.grid(axis="x", alpha=.2)

    ax = axs[1, 1]
    reps = np.arange(3)
    for label, prefix, color, marker in (("Native twin", "native_noisy_lowmid_rep", "#2b7a78", "o"),
                                         ("Augmented direct KRR", "lowmid_KRR_aug_noise", "#d17a66", "s")):
        apd = [scored["scores"][f"{prefix}{i}"]["APD_MAE_ms"] for i in reps]
        ca = [scored["scores"][f"{prefix}{i}"]["Ca_peak_MAE_uM"] for i in reps]
        ax.plot(reps, apd, color=color, marker=marker, label=f"{label}: APD")
        ax.plot(reps, np.array(ca)*300, color=color, marker=marker, ls="--",
                label=f"{label}: Ca ×300")
    ax.set_xticks(reps)
    ax.set(xlabel="Synthetic noise draw", ylabel="APD MAE (ms); Ca MAE ×300",
           title="Noisy calibration: variable winner")
    ax.grid(alpha=.2)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.savefig(ROOT / "robust3_summary.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
