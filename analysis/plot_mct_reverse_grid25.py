"""Compact figure for the exploratory rabbit reverse-challenge benchmark."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent / "mct_feedback_audit"


def main() -> None:
    grid = json.loads((ROOT / "reverse_grid25_score.json").read_text(encoding="utf-8"))
    twin = json.loads((ROOT / "reverse_twin25_score.json").read_text(encoding="utf-8"))
    inverse = json.loads((ROOT / "reverse_grid25_inverse_diagnostics.json").read_text(encoding="utf-8"))
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

    ax = axs[0, 0]
    for mode, color, marker in (("low", "#e15759", "x"), ("lowmid", "#2b7a78", "o")):
        pts = [x for x in inverse if x["mode"] == mode]
        ax.scatter([x["true_mct_gain"] for x in pts],
                   [x["inferred_mct_gain"] for x in pts],
                   color=color, marker=marker, s=28, alpha=.75, label=mode)
    ax.plot([.2, 1.3], [.2, 1.3], "k--", lw=1)
    ax.set(xlabel="True MCT feedback gain", ylabel="Inferred gain",
           title="Individual parameter recovery")
    ax.legend(frameon=False)
    ax.grid(alpha=.2)

    names = ("Low ridge", "Low+mid ridge", "Low+mid RBF", "Linear extrap.",
             "Low twin", "Low+mid twin")
    keys = ("direct_low_ridge", "direct_low_mid_ridge", "direct_low_mid_RBF_postscore",
            "linear_extrapolation", "native_ODE_twin_low", "native_ODE_twin_lowmid")
    colors = ("#8f9da8", "#5c7f93", "#3295a2", "#b0a07b", "#d47d70", "#2b7a78")
    for ax, field, ylabel in ((axs[0, 1], "APD_above_minus67_mV_ms", "High-load APD MAE (ms)"),
                              (axs[1, 0], "Ca_peak_uM", "High-load Ca peak MAE (µM)")):
        vals = [twin["scores"][k]["MAE"][field] for k in keys]
        ax.barh(names[::-1], vals[::-1], color=colors[::-1])
        ax.set_xlabel(ylabel)
        ax.grid(axis="x", alpha=.2)

    rows = {r["id"]: r for r in grid["rows"]}
    ax = axs[1, 1]
    for ident, color in (("g025_p110", "#e15759"), ("g125_p100", "#2b7a78")):
        r = rows[ident]
        x = [0.02, 0.06, 0.10]
        y = [r[f"l{round(100*v):02d}"]["APD_above_minus67_mV_ms"] for v in x]
        ax.plot(x, y, marker="o", color=color, label=ident)
    ax.set(xlabel="Gel crosslinker concentration CL", ylabel="APD above −67 mV (ms)",
           title="Illustrative low-load-matched pair")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=.2)
    fig.savefig(ROOT / "reverse_grid25_summary.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
