"""Plot exploratory fixed-state scans and independent re-pacing checks."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent / "human_bidirectional_output"


def read(name):
    with (OUT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


scan = read("dynamic_event_boundary_refine.csv")
repace = read("dynamic_event_boundary_repace.csv")
repace_refined = read("dynamic_event_boundary_repace_refined.csv")
names = {8: "INaL scale", 43: "IKr scale", 57: "effective stretch gain"}
fig, axes = plt.subplots(3, 2, figsize=(10.6, 9.0), constrained_layout=True)
for row_idx, (cell, label) in enumerate(names.items()):
    rows = sorted((r for r in scan if int(r["cell"]) == cell),
                  key=lambda r: float(r["parameter"]))
    x = np.array([float(r["parameter"]) for r in rows])
    duration = np.array([float(r["duration_above_minus50_ms"]) for r in rows])
    peak = np.array([float(r["peak_v_mV"]) for r in rows])
    ax, bx = axes[row_idx]
    ax.plot(x, duration, "o-", color="#2d6e9d", markersize=4,
            label="fixed-state scan")
    points = [r for r in repace if int(r["cell"]) == cell and int(r["beats"]) == 60]
    ax.scatter([float(r["parameter"]) for r in points],
               [float(r["duration_above_minus50_ms"]) for r in points],
               marker="s", s=55, facecolors="none", edgecolors="#db6c2b",
               linewidths=1.8, label="60-beat re-paced")
    fine_points = [r for r in repace_refined
                   if int(r["cell"]) == cell and int(r["beats"]) == 60]
    ax.scatter([float(r["parameter"]) for r in fine_points],
               [float(r["duration_above_minus50_ms"]) for r in fine_points],
               marker="^", s=65, color="#d99728", label="60-beat fine flanks")
    ax.set_ylabel("Time above -50 mV (ms)")
    ax.set_title(f"Cell {cell}: {label}")
    ax.grid(alpha=.2)
    bx.plot(x, peak, "o-", color="#8e4b8b", markersize=4)
    bx.axhline(0, linestyle="--", color="#777", linewidth=.8)
    bx.set_ylabel("Peak V during hold (mV)")
    bx.set_title("Second-voltage-response peak")
    bx.grid(alpha=.2)
    for c in (ax, bx):
        c.set_xlabel(label)
    if row_idx == 0:
        ax.legend(frameon=False, fontsize=8, loc="best")
fig.suptitle("Within-model long-hold sensitivity: exploratory tail cells", fontsize=14)
fig.savefig(OUT / "dynamic_event_boundary_pilot.png", dpi=180)
print(OUT / "dynamic_event_boundary_pilot.png")

fig2, axes2 = plt.subplots(3, 2, figsize=(10.6, 8.6), constrained_layout=True)
for row_idx, (cell, label) in enumerate(names.items()):
    pair = [np.loadtxt(OUT / f"dynamic_event_waveform_cell{cell}_side{k}.csv",
                       delimiter=",") for k in (1, 2)]
    for k, data in enumerate(pair):
        keep = (data[:, 0] >= 300) & (data[:, 0] <= 602)
        linestyle = "-" if k == 0 else "--"
        legend = "lower path fraction" if k == 0 else "higher path fraction"
        axes2[row_idx, 0].plot(data[keep, 0], data[keep, 1],
                               linestyle=linestyle, label=legend)
        axes2[row_idx, 1].plot(data[keep, 0], data[keep, 2],
                               linestyle=linestyle, label=legend)
    axes2[row_idx, 0].axhline(-50, color="#777", linewidth=.8, linestyle=":")
    axes2[row_idx, 0].set_ylabel(f"Cell {cell}: V (mV)")
    axes2[row_idx, 1].set_ylabel(f"Cell {cell}: Ca (µM)")
    for ax in axes2[row_idx]:
        ax.axvspan(300, 402, color="#dddddd", alpha=.25)
        ax.set_xlabel("Time (ms)")
        ax.grid(alpha=.2)
axes2[0, 0].legend(frameon=False, fontsize=8)
fig2.suptitle("Near-neighbour parameter values, same initial state and stretch")
fig2.savefig(OUT / "dynamic_event_boundary_waveforms.png", dpi=180)
print(OUT / "dynamic_event_boundary_waveforms.png")
