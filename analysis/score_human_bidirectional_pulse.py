"""Score finite-ramp human electromechanical feedback pulse ablations."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent / "human_bidirectional_output"
GRID = np.arange(1001)


def load(name: str) -> dict[str, np.ndarray]:
    raw = np.genfromtxt(ROOT / name, delimiter=",", names=True)
    return {key: np.interp(GRID, raw["time_ms"], raw[key])
            for key in ["Vm_mV", "Cai_uM", "Ta_kPa", "Ttotal_kPa"]}


def main() -> None:
    rows = []
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex="col", constrained_layout=True)
    colors = {(0, 0): "#555555", (1, 0): "#0072b2",
              (0, 1): "#d55e00", (1, 1): "#009e73"}
    for col, onset in enumerate([150, 600]):
        for trpn in [0, 1]:
            for sac in [0, 1]:
                pulse = load(f"pulse_onset{onset:03d}_trpn{trpn}_sac{sac}.csv")
                control = load(f"trpn{trpn}_sac{sac}_lambda100.csv")
                post = GRID >= onset
                dv = pulse["Vm_mV"] - control["Vm_mV"]
                dc = pulse["Cai_uM"] - control["Cai_uM"]
                rows.append({
                    "onset_ms": onset, "trpn_feedback": bool(trpn),
                    "SAC_feedback": bool(sac),
                    "delta_APD_ms": int(np.count_nonzero(pulse["Vm_mV"][:-1] > -67) -
                                        np.count_nonzero(control["Vm_mV"][:-1] > -67)),
                    "max_abs_post_pulse_delta_Vm_mV": float(np.abs(dv[post]).max()),
                    "max_abs_post_pulse_delta_Ca_uM": float(np.abs(dc[post]).max()),
                    "delta_Ca_peak_uM": float(pulse["Cai_uM"].max() -
                                              control["Cai_uM"].max()),
                    "max_active_tension_kPa": float(pulse["Ta_kPa"].max()),
                })
                for row, key in enumerate(["Vm_mV", "Cai_uM", "Ta_kPa"]):
                    axes[row, col].plot(GRID, pulse[key], color=colors[trpn, sac],
                                        label=f"TnC {trpn}, SAC {sac}", lw=1.2)
                    axes[row, col].axvspan(onset, onset+20, color="#cccccc", alpha=.25)
                    axes[row, col].grid(alpha=.2)
        axes[0, col].set_title(f"5 ms ramp + 10 ms hold + 5 ms return at {onset} ms")
        axes[0, col].set_xlim(max(0, onset-100), min(1000, onset+400))
        axes[2, col].set_xlabel("Time within beat (ms)")
    for row, label in enumerate(["Membrane voltage (mV)", "Intracellular Ca (µM)",
                                  "Active tension (kPa)"]):
        axes[row, 0].set_ylabel(label)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.savefig(ROOT / "finite_ramp_pulse_ablation.png", dpi=160)
    plt.close(fig)
    assert all(np.isfinite(value) for row in rows for value in row.values()
               if isinstance(value, float))
    (ROOT / "pulse_audit.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
