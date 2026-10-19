"""Score within-onset acute stretch effects in the human SAC reference model."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent / "human_sac_reference_pilot"
GRID = np.arange(0, 1000)


def load(onset: int, lam: float) -> dict:
    path = OUT / f"lambda_{round(100*lam):03d}_onset_{onset:03d}_trace.csv"
    raw = np.genfromtxt(path, delimiter=",", names=True)
    return {"Vm": np.interp(GRID, raw["time_ms"], raw["Vm_mV"]),
            "Ca": np.interp(GRID, raw["time_ms"], raw["Cai_uM"])}


def main() -> None:
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex="col", constrained_layout=True)
    for j, onset in enumerate([150, 300, 600]):
        control = load(onset, 1.00)
        for lam, color in [(1.00, "#333333"), (1.05, "#0072b2"), (1.10, "#d55e00")]:
            case = load(onset, lam)
            dv = case["Vm"] - control["Vm"]
            dc = case["Ca"] - control["Ca"]
            post = GRID >= onset
            upward_zero = np.where((case["Vm"][:-1] < 0) &
                                   (case["Vm"][1:] >= 0) &
                                   (GRID[1:] >= onset+10) & (GRID[1:] < 900))[0]
            new_upstroke = any(np.min(case["Vm"][max(0, k-30):k+1]) < -60
                               for k in upward_zero)
            rows.append({
                "onset_ms": onset, "lambda": lam,
                "APD_above_minus67_ms": int(np.sum(case["Vm"] > -67)),
                "delta_APD_ms": int(np.sum(case["Vm"] > -67)-np.sum(control["Vm"] > -67)),
                "Ca_peak_uM": float(np.max(case["Ca"])),
                "delta_Ca_peak_uM": float(np.max(case["Ca"])-np.max(control["Ca"])),
                "max_positive_delta_Vm_mV": float(np.max(dv[post])),
                "max_negative_delta_Vm_mV": float(np.min(dv[post])),
                "max_positive_delta_Ca_uM": float(np.max(dc[post])),
                "new_upstroke_after_stretch": bool(new_upstroke),
            })
            axes[0, j].plot(GRID, case["Vm"], color=color, label=f"λ {lam:.2f}")
            axes[1, j].plot(GRID, case["Ca"], color=color)
        axes[0, j].set_title(f"Stretch at {onset} ms for 10 ms")
        axes[1, j].set_xlabel("Time within beat (ms)")
        for ax in axes[:, j]:
            ax.axvspan(onset, onset+10, color="#bdbdbd", alpha=.35)
            ax.grid(alpha=.2)
            ax.set_xlim(max(0,onset-100), min(1000,onset+350))
    axes[0, 0].set_ylabel("Membrane voltage (mV)")
    axes[1, 0].set_ylabel("Intracellular Ca (µM)")
    axes[0, 0].legend(frameon=False)
    fig.savefig(OUT / "stretch_timing_grid.png", dpi=180)
    plt.close(fig)
    (OUT / "timing_grid_score.json").write_text(json.dumps({
        "protocol": "analysis/human_sac_timing_grid_protocol.md", "scores": rows}, indent=2),
        encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
