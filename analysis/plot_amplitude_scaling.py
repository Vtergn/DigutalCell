"""Diagnostic figure for held-out peak scaling and cell-paired errors."""

from __future__ import annotations

import csv
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from dynamic_input_screen import HERE


def main() -> None:
    with (HERE / "amplitude_scaling_scores.csv").open(newline="", encoding="utf-8") as handle:
        scores = list(csv.DictReader(handle))
    with (HERE / "amplitude_scaling_fits.csv").open(newline="", encoding="utf-8") as handle:
        fits = list(csv.DictReader(handle))
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    for col, (channel, title, color) in enumerate((("mp1", "Piezo1", "#2563eb"),
                                                    ("mp2", "Piezo2", "#ea580c"))):
        baseline = [r for r in scores if r["channel"] == channel and r["model"] == "A0_cal_peak"]
        x = np.array([float(r["force_ratio"]) for r in baseline])
        y = np.array([float(r["actual_peak_pA"])/float(r["cal_peak_pA"]) for r in baseline])
        events = np.array([int(r["actual_has_event"]) > 0 for r in baseline])
        ax = axes[0, col]
        ax.scatter(x[events], y[events], s=26, alpha=.75, color=color, edgecolor="none", label="future responder")
        if np.any(~events):
            ax.scatter(x[~events], y[~events], s=38, facecolor="none", edgecolor=color,
                       linewidth=1.4, label="no detected event")
        xmax = max(4.5, float(np.max(x))*1.05)
        xx = np.geomspace(.4, xmax, 300)
        gamma = np.median([float(r["gamma_force"]) for r in fits if r["channel"] == channel])
        ax.plot(xx, xx, color="#64748b", lw=1.5, label="linear force")
        ax.plot(xx, xx**gamma, color="#0f172a", lw=1.8, label=f"force power (median exponent {gamma:.2f})")
        ax.set(xscale="log", yscale="log", xlim=(.4, xmax),
               ylim=(.05, max(50, np.max(y)*1.1)), title=title,
               xlabel="future / calibration force above threshold",
               ylabel="future / calibration current peak")
        ax.grid(alpha=.2, which="both")
        ticks = [v for v in (.5, 1, 2, 4, 8) if v <= xmax]
        ax.set_xticks(ticks, [str(v) for v in ticks])
        ax.legend(fontsize=8, loc="upper left")
        by_cell = defaultdict(dict)
        for row in scores:
            if row["channel"] == channel and row["model"] in {"A1_force_linear", "A2_force_power"}:
                by_cell[row["cell_id"]].setdefault(row["model"], []).append(float(row["abs_error_pA"]))
        paired = sorted((np.mean(v["A2_force_power"])-np.mean(v["A1_force_linear"]), k)
                        for k, v in by_cell.items())
        differences = np.array([value for value, _ in paired])
        ax = axes[1, col]
        ax.bar(np.arange(len(differences)), differences,
               color=["#16a34a" if d < 0 else "#dc2626" for d in differences])
        ax.axhline(0, color="#0f172a", lw=1)
        ax.set(xlabel="held-out cell (sorted within channel)",
               ylabel="A2 − A1 mean absolute peak error (pA)",
               title=f"{np.sum(differences < 0)}/{len(differences)} cells improved")
        ax.set_xticks([])
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("One-shot peak-current scaling: whole-date held-out", fontsize=14)
    fig.savefig(HERE / "amplitude_scaling_diagnostics.png", dpi=180)
    fig.savefig(HERE / "amplitude_scaling_diagnostics.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
