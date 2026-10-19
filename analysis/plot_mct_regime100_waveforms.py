"""Overlay late AP/Ca waveforms to audit the period-2 feature label."""
from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent / "mct_feedback_audit" / "reverse_robust3" / "regime100"


def load(path: Path) -> dict:
    with h5py.File(path) as f:
        return {key: value[()].ravel() for key, value in f["a"].items()}


def main() -> None:
    id_ = "train_036"
    conditions = [
        ("CL 0.06: stable 1:1", ROOT / "load_check" / f"{id_}_l06_100.mat"),
        ("CL 0.10: period-2 candidate", ROOT / "native" / f"{id_}_l10_100.mat"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, constrained_layout=True)
    grid = np.arange(0, 1000, 1)
    for column, (title, path) in enumerate(conditions):
        case = load(path)
        for beat, color in [(99, "#0072b2"), (100, "#d55e00")]:
            start = (beat - 1) * 1000
            end = beat * 1000
            t = case["t_ms"]
            chosen = (t >= start) & (t < end)
            rel_t = t[chosen] - start
            voltage = np.interp(grid, rel_t, case["Vm_mV"][chosen])
            calcium = np.interp(grid, rel_t, case["Cai_uM"][chosen])
            axes[0, column].plot(grid, voltage, color=color, lw=1.4, label=f"Beat {beat}")
            axes[1, column].plot(grid, calcium, color=color, lw=1.4)
        axes[0, column].axhline(-67, color="k", ls="--", lw=.8, alpha=.6)
        axes[0, column].set_title(title)
        axes[1, column].set_xlabel("Time within beat (ms)")
        axes[0, column].legend(frameon=False)
        for ax in axes[:, column]:
            ax.grid(alpha=.2)
    axes[0, 0].set_ylabel("Membrane voltage (mV)")
    axes[1, 0].set_ylabel("Intracellular Ca (µM)")
    fig.savefig(ROOT / "train036_waveform_overlay.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
