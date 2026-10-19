"""Cell-paired held-out diagnostics for the tension-gating simulator."""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent


def main():
    with (HERE / "mechanistic_tension_personalization_scores.csv").open(encoding="utf-8") as handle:
        scores = list(csv.DictReader(handle))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, channel in zip(axes, ("mp1", "mp2")):
        grouped = defaultdict(lambda: defaultdict(list))
        for row in scores:
            if row["channel"] == channel:
                grouped[row["cell_id"]][row["model"]].append(float(row["mae_pA"]))
        cells = sorted(grouped)
        x = np.arange(len(cells))
        for offset, model, label, color in (
            (-.17, "shared_T0_with_fixed_lag", "shared offset", "#4477aa"),
            (.17, "one_shot_shape_T0_with_fixed_lag", "one-shot offset", "#cc6677"),
        ):
            delta = [np.mean(grouped[c][model]) - np.mean(grouped[c]["lagged_threshold_force"])
                     for c in cells]
            axis.scatter(x + offset, delta, label=label, color=color, s=34)
        axis.axhline(0, color="black", linewidth=.8)
        axis.set(title=f"{'Piezo1' if channel == 'mp1' else 'Piezo2 (exploratory)'}: whole-date held out",
                 xlabel="Independent test cell", ylabel="350 ms MAE change vs lagged force (pA)")
        axis.set_xticks(x, [str(i+1) for i in x])
        axis.legend(fontsize=8)
        axis.grid(axis="y", alpha=.2)
    fig.savefig(HERE / "mechanistic_tension_paired.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
