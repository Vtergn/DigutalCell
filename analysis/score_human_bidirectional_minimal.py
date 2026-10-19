"""Audit pathway ablations of the first human ToR-ORd/SAC/Land closed loop."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent / "human_bidirectional_output"
GRID = np.arange(1001)


def load(trpn: int, sac: int, lam: float) -> dict[str, np.ndarray]:
    name = f"trpn{trpn}_sac{sac}_lambda{round(100*lam):03d}.csv"
    raw = np.genfromtxt(ROOT / name, delimiter=",", names=True)
    return {key: np.interp(GRID, raw["time_ms"], raw[key])
            for key in ["Vm_mV", "Cai_uM", "Ta_kPa", "Tp_kPa", "Ttotal_kPa"]}


def main() -> None:
    rows: list[dict] = []
    traces = {}
    for trpn in [0, 1]:
        for sac in [0, 1]:
            control = load(trpn, sac, 1.0)
            for lam in [1.05, 1.1]:
                challenge = load(trpn, sac, lam)
                traces[trpn, sac, lam] = challenge
                dv = challenge["Vm_mV"] - control["Vm_mV"]
                dc = challenge["Cai_uM"] - control["Cai_uM"]
                rows.append({
                    "trpn_feedback": bool(trpn), "SAC_feedback": bool(sac),
                    "lambda": lam,
                    "delta_APD_above_minus67_ms": int(
                        np.count_nonzero(challenge["Vm_mV"][:-1] > -67) -
                        np.count_nonzero(control["Vm_mV"][:-1] > -67)),
                    "max_abs_delta_Vm_mV": float(np.max(np.abs(dv))),
                    "max_abs_delta_Ca_uM": float(np.max(np.abs(dc))),
                    "delta_Ca_peak_uM": float(np.max(challenge["Cai_uM"]) -
                                              np.max(control["Cai_uM"])),
                    "delta_active_tension_peak_kPa": float(
                        np.max(challenge["Ta_kPa"]) -
                        np.max(control["Ta_kPa"])),
                    "peak_total_tension_kPa": float(np.max(challenge["Ttotal_kPa"])),
                })
    # The mechanical state still changes the adaptive solver's step sequence.
    # Require the nominally disconnected electrical effect to be negligible.
    assert max(r["max_abs_delta_Vm_mV"] for r in rows
               if not r["trpn_feedback"] and not r["SAC_feedback"]) < 1e-3
    assert max(r["max_abs_delta_Ca_uM"] for r in rows
               if not r["trpn_feedback"] and not r["SAC_feedback"]) < 1e-5
    for trace in traces.values():
        for array in trace.values():
            assert np.all(np.isfinite(array))
        assert np.min(trace["Cai_uM"]) > 0
    metrics = np.loadtxt(ROOT / "prepace_metrics.csv", delimiter=",")
    audit = {
        "prepace_beats": int(metrics[-1, 0]),
        "last_10_APD_range_ms": float(np.ptp(metrics[-10:, 1])),
        "last_10_Ca_peak_relative_range": float(np.ptp(metrics[-10:, 2]) /
                                                 np.mean(metrics[-10:, 2])),
        "conditions": rows,
    }
    (ROOT / "ablation_audit.json").write_text(json.dumps(audit, indent=2),
                                                encoding="utf-8")
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex="col", constrained_layout=True)
    colors = {(0, 0): "#555555", (1, 0): "#0072b2",
              (0, 1): "#d55e00", (1, 1): "#009e73"}
    for j, lam in enumerate([1.0, 1.1]):
        for trpn in [0, 1]:
            for sac in [0, 1]:
                x = load(trpn, sac, lam)
                label = f"TnC {'on' if trpn else 'off'}, SAC {'on' if sac else 'off'}"
                for i, col in enumerate(["Vm_mV", "Cai_uM", "Ta_kPa"]):
                    axes[i, j].plot(GRID, x[col], color=colors[trpn, sac],
                                    label=label, linewidth=1.2)
        axes[0, j].set_title(f"Constant stretch λ={lam:.2f}; first beat after calibration")
        axes[2, j].set_xlabel("Time within beat (ms)")
        axes[0, j].set_xlim(0, 600)
    for ax, title in zip(axes[:, 0], ["Membrane voltage (mV)",
                                         "Intracellular Ca (µM)",
                                         "Active tension (kPa)"]):
        ax.set_ylabel(title)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.savefig(ROOT / "bidirectional_ablation.png", dpi=160)
    plt.close(fig)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
