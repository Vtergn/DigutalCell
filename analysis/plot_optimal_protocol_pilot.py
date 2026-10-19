"""Plot the held-cell medians from the synthetic protocol-design pilot."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent


def main() -> None:
    data = json.loads((HERE / "optimal_protocol_pilot_results.json").read_text(
        encoding="utf-8"))["summary_median"]
    methods = ("duke_like", "random", "information_optimal")
    labels = ("Fixed speed / 5 s", "Random diverse", "Information guided")
    colors = ("#9b9b9b", "#349a92", "#4c68a9")
    metrics = (("tau_abs_log_error", "Recovery parameter |log error|"),
               ("challenge_peak_abs_error", "Future challenge peak |error|"))
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.1), layout="constrained")
    for row, scenario in enumerate(("in_model", "misspecified")):
        for col, (metric, title) in enumerate(metrics):
            values = [data[scenario][method][metric] for method in methods]
            ax = axes[row, col]
            ax.bar(np.arange(3), values, color=colors, width=.67)
            ax.set_xticks(np.arange(3), labels, rotation=14, ha="right")
            ax.set_ylabel(title)
            ax.set_title(("Correct model" if row == 0 else "Model mismatch")+
                         (" · recovery" if col == 0 else " · future prediction"))
            ax.grid(axis="y", alpha=.2)
            for i, value in enumerate(values):
                ax.text(i, value+max(values)*.025, f"{value:.3f}",
                        ha="center", va="bottom", fontsize=9)
            ax.set_ylim(0, max(values)*1.21)
    fig.suptitle("Six synthetic stimuli per cell; medians across 50 cells")
    fig.savefig(HERE / "optimal_protocol_pilot.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
