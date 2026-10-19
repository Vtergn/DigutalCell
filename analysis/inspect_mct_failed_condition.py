"""Inspect beat-wise drift/alternation in the 60-beat high-load outlier."""
from pathlib import Path

import h5py
import matplotlib.pyplot as plt

from score_mct2024_reference import beats


ROOT = Path(__file__).resolve().parent / "mct_feedback_audit/reverse_robust3"
PATH = ROOT / "failed_60beat/train_036_l10_60.mat"
with h5py.File(PATH) as f:
    g = f["a"]
    case = {k: g[k][()].ravel() for k in
            ("t_ms", "Vm_mV", "Cai_uM", "t_mech_ms", "Lneg", "force", "gelForce", "pCaCoeff")}
last = beats(case, final_beat=60, window=24)
for row in last[-12:]:
    print(row["beat"], row["APD_above_minus67_mV_ms"], round(row["Ca_peak_uM"], 4))
fig, axs = plt.subplots(2, 1, figsize=(8, 5), sharex=True, constrained_layout=True)
for ax, key, label in ((axs[0], "APD_above_minus67_mV_ms", "APD above −67 mV (ms)"),
                       (axs[1], "Ca_peak_uM", "Ca peak (µM)")):
    ax.plot([r["beat"] for r in last], [r[key] for r in last], "o-", ms=3)
    ax.axvline(40, color="gray", ls="--", lw=1)
    ax.set_ylabel(label)
    ax.grid(alpha=.2)
axs[-1].set_xlabel("Beat number")
fig.savefig(ROOT / "train036_highload_beat_dynamics.png", dpi=180)
plt.close(fig)
