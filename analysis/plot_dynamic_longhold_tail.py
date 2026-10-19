"""Plot held-out saturating-SAC long-hold voltage for audited cells."""
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "human_bidirectional_output"
with h5py.File(OUT / "dynamic_discrepancy_cohort.mat") as f:
    truth = f["waveforms"][:]
with h5py.File(OUT / "dynamic_model_candidates.mat") as f:
    candidate = f["candidate_target"][:]
with h5py.File(OUT / "dynamic_inverse_calibration.mat") as f:
    tab = f["tab"][:].T
    mapping = f["map"][:].ravel().astype(int) - 1

cells = [7, 17, 42, 56]
fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
t = np.arange(1001)
for ax, cell in zip(axes.ravel(), cells):
    row, = np.flatnonzero((tab[:, 0] == cell + 1) &
                           (tab[:, 1] == 2) & (tab[:, 2] == 1))
    u = mapping[row]
    actual = truth[cell, 1, 3, 0]
    predicted = candidate[u, 1, 0, 0]
    ax.axvspan(300, 402, color="#e5e5e5", alpha=.55)
    ax.plot(t, actual, color="#176b9a", lw=1.5, label="True saturating SAC")
    ax.plot(t, predicted, color="#d05b32", lw=1.4,
            label="Inferred state + correct SAC")
    ax.set_xlim(290, 610)
    ax.set_title(f"Virtual cell {cell + 1}")
    ax.grid(alpha=.15)
for ax in axes[:, 0]:
    ax.set_ylabel("Voltage (mV)")
for ax in axes[1]:
    ax.set_xlabel("Time (ms)")
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center",
           ncol=2, frameon=False)
fig.tight_layout(rect=(0, 0, 1, .94))
fig.savefig(OUT / "dynamic_longhold_tail_voltage.png", dpi=180)
