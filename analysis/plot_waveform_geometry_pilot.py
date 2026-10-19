"""Cell-level PCA diagnostics for the force-peak-aligned geometry pilot."""

from __future__ import annotations

import csv
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from dynamic_input_screen import HERE


def main(suffix: str = "") -> None:
    with (HERE / f"waveform_geometry_pilot_scores{suffix}.csv").open(newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["alignment"] == "afm_force_peak"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.6), constrained_layout=True)
    for column, (channel, name) in enumerate((("mp1", "Piezo1"), ("mp2", "Piezo2"))):
        for row_index, (representation, feature, label) in enumerate((
            ("raw_pA", "peak_current_pA", "log10 peak current (pA)"),
            ("observed_peak_normalized", "postpeak_20ms_retention", "current retention at +20 ms"),
        )):
            by_cell = defaultdict(list)
            for record in rows:
                if record["channel"] == channel and record["representation"] == representation:
                    by_cell[record["cell_id"]].append(record)
            cell_rows = list(by_cell.values())
            x = np.array([np.mean([float(r["pc1"]) for r in group]) for group in cell_rows])
            y = np.array([np.mean([float(r["pc2"]) for r in group]) for group in cell_rows])
            z = np.array([np.mean([float(r[feature]) for r in group]) for group in cell_rows])
            if feature == "peak_current_pA":
                z = np.log10(z)
            ax = axes[row_index, column]
            scatter = ax.scatter(x, y, c=z, cmap="viridis", s=55, edgecolor="#111827", linewidth=.5)
            fig.colorbar(scatter, ax=ax, label=label)
            ax.set(title=f"{name} | {'raw amplitude' if row_index == 0 else 'observed-peak-normalized shape'}",
                   xlabel="PC1", ylabel="PC2")
            ax.grid(alpha=.2)
    fig.suptitle("AFM force-peak-aligned current waveforms (one marker per cell)" +
                 (" | peak SNR ≥ 5" if suffix else " | detected events"))
    fig.savefig(HERE / f"waveform_geometry_pilot{suffix}.png", dpi=180)
    fig.savefig(HERE / f"waveform_geometry_pilot{suffix}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
    main("_snr5")
