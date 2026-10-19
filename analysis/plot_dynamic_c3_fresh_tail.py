"""New-seed sealed long-hold voltage in the three largest-risk cells."""
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "human_bidirectional_output"
score = json.loads((OUT / "dynamic_c3_fresh_scores.json").read_text())
arm = score["selected_arm"]
with h5py.File(OUT / "dynamic_c3_fresh.mat") as f:
    truth = f["waveforms"][:]
with h5py.File(OUT / "dynamic_c3_fresh_targets.mat") as f:
    pred = f["pred"][:]
    tab = f["tab"][:].T

fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), sharey=True)
t = np.arange(1001)
for ax, cell in zip(axes, [18, 37, 53]):
    ax.axvspan(300, 402, color="#e5e5e5", alpha=.6)
    ax.plot(t, truth[cell-1, 4, 0], color="#176b9a", lw=1.5,
            label="True long hold")
    for a, color, label in [(0, "#a56a37", "B+C1+C2"),
                            (arm, "#b73256", "B+C1+C2+C3")]:
        row, = np.flatnonzero((tab[:, 0] == cell) & (tab[:, 1] == a))
        ax.plot(t, pred[row, 1, 0], color=color, lw=1.25, label=label)
    ax.set_xlim(300, 610)
    ax.set_title(f"Virtual cell {cell}")
    ax.set_xlabel("Time (ms)")
    ax.grid(alpha=.15)
axes[0].set_ylabel("Voltage (mV)")
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
           ncol=3, frameon=False)
fig.tight_layout(rect=(0, 0, 1, .88))
fig.savefig(OUT / "dynamic_c3_fresh_tail_voltage.png", dpi=180)
