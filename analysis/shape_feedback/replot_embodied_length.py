"""Replot the saved closed-loop trajectories without re-solving the ODE."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

here = Path(__file__).resolve().parent
saved = np.load(here / "embodied_length_pilot.npz")
fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True, layout="constrained")
series = (("free_feedback", "#666666", "No external load"),
          ("load_feedback", "#ba2d3b", "External load, SAC on"),
          ("load_no_sac", "#2455a4", "External load, SAC off"))
for name, color, label in series:
    time = saved[f"{name}_time"]
    wave = saved[f"{name}_wave"]
    mechanical_time = saved[f"{name}_stress_time"]
    axes[0].plot(mechanical_time, saved[f"{name}_lam"], color=color, label=label)
    axes[1].plot(time, wave[:, 0], color=color)
    axes[2].plot(time, wave[:, 1], color=color)
    axes[3].plot(mechanical_time, saved[f"{name}_active"], color=color)
for axis, title in zip(axes, ("Solved sarcomere stretch λ", "Voltage (mV)",
                              "Cytosolic Ca (µM)", "Active tension (kPa)")):
    axis.set_ylabel(title)
    axis.axvspan(250, 350, color="#aaaaaa", alpha=.12)
    axis.grid(alpha=.18)
axes[0].legend(frameon=False, ncol=2, fontsize=8)
axes[-1].set_xlabel("Time (ms)")
fig.savefig(here / "embodied_length_pilot.png", dpi=180)
