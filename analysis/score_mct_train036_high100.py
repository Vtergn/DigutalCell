"""Describe, without mechanistic attribution, the long high-load beat sequence."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from score_mct2024_reference import beats


ROOT = Path(__file__).resolve().parent / "mct_feedback_audit/reverse_robust3"
PATH = ROOT / "train036_high100.mat"


def main() -> None:
    with h5py.File(PATH) as f:
        g = f["a"]
        case = {k: g[k][()].ravel() for k in
                ("t_ms", "Vm_mV", "Cai_uM", "t_mech_ms", "Lneg", "force", "gelForce", "pCaCoeff")}
    rows = beats(case, final_beat=100, window=64)
    last10 = rows[-10:]
    apd = np.array([x["APD_above_minus67_mV_ms"] for x in last10])
    ca = np.array([x["Ca_peak_uM"] for x in last10])
    result = {
        "case": "train_036 at CL 0.10 for 100 beats, 1000ms pacing",
        "last10_APD_ms": apd.tolist(), "last10_Ca_peak_uM": ca.tolist(),
        "last10_APD_range_ms": float(np.ptp(apd)),
        "last10_Ca_peak_range_fraction": float(np.ptp(ca)/ca[-1]),
        "even_minus_odd_last10_APD_ms": float(np.mean(apd[1::2])-np.mean(apd[::2])),
        "even_minus_odd_last10_Ca_uM": float(np.mean(ca[1::2])-np.mean(ca[::2])),
        "interpretation": "Exploratory model sequence. Alternating pattern alone does not establish biological arrhythmia or numerical convergence.",
    }
    (ROOT / "train036_high100_score.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    fig, axs = plt.subplots(2, 1, figsize=(8, 5), sharex=True, constrained_layout=True)
    for ax, key, label in ((axs[0], "APD_above_minus67_mV_ms", "APD above −67 mV (ms)"),
                           (axs[1], "Ca_peak_uM", "Ca peak (µM)")):
        ax.plot([r["beat"] for r in rows], [r[key] for r in rows], "o-", ms=2.5)
        ax.axvline(40, color="gray", ls="--", lw=1)
        ax.axvline(60, color="gray", ls=":", lw=1)
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
    axs[-1].set_xlabel("Beat number")
    fig.savefig(ROOT / "train036_high100_beats.png", dpi=180)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
