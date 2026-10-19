"""Audit acute stretch signatures from the published human ToR-ORd SAC code."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent / "human_sac_reference_pilot"


def main() -> None:
    metadata = json.loads((OUT / "metadata.json").read_text(encoding="utf-8"))
    pre = np.loadtxt(OUT / "prepace_metrics.csv", delimiter=",")
    grid = np.arange(0, 1000, 1)
    cases = {}
    for lam in [1.00, 1.05, 1.10]:
        raw = np.genfromtxt(OUT / f"lambda_{round(100*lam):03d}_trace.csv",
                            delimiter=",", names=True)
        cases[lam] = {
            "Vm_mV": np.interp(grid, raw["time_ms"], raw["Vm_mV"]),
            "Cai_uM": np.interp(grid, raw["time_ms"], raw["Cai_uM"]),
        }
    control = cases[1.00]
    scores = []
    for lam, case in cases.items():
        vm, ca = case["Vm_mV"], case["Cai_uM"]
        post = (grid >= 600) & (grid <= 900)
        scores.append({
            "lambda": lam,
            "APD_above_minus67_ms": int(np.sum(vm > -67)),
            "Ca_peak_uM": float(ca.max()),
            "poststretch_max_Vm_mV": float(vm[post].max()),
            "poststretch_max_delta_Vm_vs_control_mV": float((vm-control["Vm_mV"])[post].max()),
            "poststretch_max_delta_Ca_vs_control_uM": float((ca-control["Cai_uM"])[post].max()),
            "poststretch_Vm_crossed_zero": bool(np.any(vm[post] > 0)),
        })
    result = {"metadata": metadata, "prepace_last10": pre[-10:].tolist(), "scores": scores}
    (OUT / "score.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    colors = {1.00: "#333333", 1.05: "#0072b2", 1.10: "#d55e00"}
    for lam, case in cases.items():
        for row, key in enumerate(["Vm_mV", "Cai_uM"]):
            axes[row, 0].plot(grid, case[key], color=colors[lam], label=f"λ {lam:.2f}")
            axes[row, 1].plot(grid, case[key], color=colors[lam])
    for ax in axes[:, 0]:
        ax.set_xlim(0, 1000)
    for ax in axes[:, 1]:
        ax.set_xlim(550, 850)
    for ax in axes.ravel():
        ax.axvspan(600, 610, color="#bdbdbd", alpha=.35)
        ax.grid(alpha=.2)
        ax.set_xlabel("Time in challenged beat (ms)")
    axes[0, 0].set_ylabel("Membrane voltage (mV)")
    axes[1, 0].set_ylabel("Intracellular Ca (µM)")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].set_title("Full beat")
    axes[0, 1].set_title("Acute stretch window")
    fig.savefig(OUT / "acute_stretch_AP_Ca.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"prepace_beats": metadata["prepace_beats"],
                      "converged": metadata["convergence_criterion_met"],
                      "scores": scores}, indent=2))


if __name__ == "__main__":
    main()
