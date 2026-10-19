"""Compare acute stretch effects after 120 and 500 beats of pre-pacing."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent / "human_sac_reference_pilot"
GRID = np.arange(1000)


def trace(beats: int, lam: float, onset: int) -> tuple[np.ndarray, np.ndarray]:
    if beats == 500:
        name = f"prepace500_lambda_{round(100 * lam):03d}_onset_{onset:03d}_trace.csv"
    else:
        name = f"lambda_{round(100 * lam):03d}_onset_{onset:03d}_trace.csv"
    raw = np.genfromtxt(OUT / name, delimiter=",", names=True)
    return (np.interp(GRID, raw["time_ms"], raw["Vm_mV"]),
            np.interp(GRID, raw["time_ms"], raw["Cai_uM"]))


def main() -> None:
    metrics = np.loadtxt(OUT / "prepace500_metrics.csv", delimiter=",")
    rows = []
    for beats in [120, 500]:
        for onset in [150, 600]:
            vm0, ca0 = trace(beats, 1.0, onset)
            vm1, ca1 = trace(beats, 1.1, onset)
            post = GRID >= onset
            rows.append({
                "prepace_beats": beats, "stretch_onset_ms": onset,
                "control_APD_above_minus67_ms": int(np.count_nonzero(vm0 > -67)),
                "delta_APD_ms": int(np.count_nonzero(vm1 > -67) -
                                    np.count_nonzero(vm0 > -67)),
                "control_Ca_peak_uM": float(ca0.max()),
                "delta_Ca_peak_uM": float(ca1.max() - ca0.max()),
                "max_positive_delta_Vm_mV": float((vm1-vm0)[post].max()),
                "max_positive_delta_Ca_uM": float((ca1-ca0)[post].max()),
            })
    result = {
        "last_10_beats_APD_range_ms": float(np.ptp(metrics[-10:, 1])),
        "last_10_beats_Ca_peak_relative_range": float(
            np.ptp(metrics[-10:, 2]) / np.mean(metrics[-10:, 2])),
        "beat_120_to_500_APD_change_ms": float(metrics[499, 1]-metrics[119, 1]),
        "beat_120_to_500_Ca_peak_change_uM": float(metrics[499, 2]-metrics[119, 2]),
        "scores": rows,
    }
    target = OUT / "prepace500_audit.json"
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex="col", constrained_layout=True)
    for col, onset in enumerate([150, 600]):
        vm0, ca0 = trace(500, 1.0, onset)
        vm1, ca1 = trace(500, 1.1, onset)
        post = (GRID >= onset) & (GRID <= onset + 120)
        axes[0, col].plot(GRID, vm0, color="#555555", label="λ=1.00 control")
        axes[0, col].plot(GRID, vm1, color="#d55e00", label="λ=1.10 stretch")
        axes[1, col].plot(GRID[post], (vm1-vm0)[post], color="#d55e00",
                          label="ΔVm")
        axes[1, col].set_xlabel("Time within beat (ms)")
        axes[0, col].set_title(f"10 ms stretch at {onset} ms; 500 beat prepace")
        axes[0, col].set_xlim(max(0, onset-80), min(999, onset+200))
        axes[1, col].set_xlim(onset, onset+120)
        for row in range(2):
            axes[row, col].axvspan(onset, onset+10, color="#aaaaaa", alpha=.3)
            axes[row, col].grid(alpha=.2)
    axes[0, 0].set_ylabel("Membrane voltage (mV)")
    axes[1, 0].set_ylabel("Stretch − control Vm (mV)")
    axes[0, 0].legend(frameon=False)
    fig.savefig(OUT / "prepace500_stretch_effect.png", dpi=180)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
