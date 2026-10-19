"""Compact diagnostic figure for the frozen WT shape analysis."""

from __future__ import annotations

import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dynamic_input_screen import HERE


def rows(name: str) -> list[dict]:
    with (HERE / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    summary = rows("mechanical_waveform_null_summary.csv")
    repeats = rows("mechanical_waveform_null_repeats.csv")
    projection = rows("frozen_wt_projection_scores.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), constrained_layout=True)
    colors = {"mp1": "#4773b7", "mp2": "#dc784b"}
    for index, channel in enumerate(("mp1", "mp2")):
        values = np.array([float(row["pc1"]) for row in repeats if row["channel"] == channel])
        target = next(float(row["observed"]) for row in summary
                      if row["channel"] == channel and row["metric"] == "pc1")
        axes[0].hist(values, bins=22, alpha=.55, density=True, color=colors[channel],
                     label=f"{'Piezo1' if channel == 'mp1' else 'Piezo2'} force null")
        axes[0].axvline(target, color=colors[channel], linewidth=2.4, linestyle="--",
                        label=f"{'Piezo1' if channel == 'mp1' else 'Piezo2'} observed")
    axes[0].set_xlabel("PC1 variance fraction")
    axes[0].set_ylabel("Noise-null density")
    axes[0].set_title("Low rank also arises from force + noise")
    axes[0].legend(fontsize=8, loc="upper left")
    groups = ["WT_mp1", "chimA", "WT_mp2", "Piezo2_DMSO", "Piezo2_CytoD"]
    labels = ["WT Piezo1", "chimA", "WT Piezo2", "P2 DMSO", "P2 CytoD"]
    palette = ["#4773b7", "#8f60ab", "#dc784b", "#e2a553", "#55a3a0"]
    rng = np.random.default_rng(7)
    for position, (group, color) in enumerate(zip(groups, palette), start=1):
        per_cell = defaultdict(list)
        for row in projection:
            if row["group"] == group:
                per_cell[row["cell_id"]].append(float(row["p1_to_p2_axis"]))
        values = np.array([np.mean(item) for item in per_cell.values()])
        jitter = rng.uniform(-.13, .13, len(values))
        axes[1].scatter(np.full(len(values), position)+jitter, values,
                        color=color, alpha=.7, s=25, edgecolors="none")
        axes[1].plot([position-.22, position+.22], [np.median(values)]*2,
                     color="black", linewidth=2)
    axes[1].axhline(0, color="#4773b7", linestyle=":", alpha=.7)
    axes[1].axhline(1, color="#dc784b", linestyle=":", alpha=.7)
    axes[1].set_xticks(range(1, len(groups)+1), labels, rotation=30, ha="right")
    axes[1].set_ylabel("Frozen WT Piezo1 → Piezo2 axis")
    axes[1].set_title("External waveforms, one dot per cell")
    axes[1].text(.03, .03, "chimA vs WT: different dates", transform=axes[1].transAxes,
                 fontsize=8, color="#555555")
    fig.savefig(HERE / "signed_null_projection.png", dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    main()
