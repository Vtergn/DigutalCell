"""Linearized mechanics/gating cascade: an exact dynamic ambiguity.

Synthetic 1-ms protocol. This does not assign measured Piezo time constants;
it checks which data types can label two stages in a minimal linear cascade.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DT_MS = 1.0
T = np.arange(0, 350, DT_MS)
F = np.zeros(T.shape)
F[(T >= 5) & (T < 85)] = 1.0
TAU_FAST_MS, TAU_SLOW_MS = 2.0, 10.0
G = 100.0
SD_M = .05


def filter_first_order(x, tau_ms):
    decay = np.exp(-DT_MS/tau_ms)
    y = np.zeros_like(x, float)
    previous = 0.
    for j, value in enumerate(x):
        previous = decay*previous + (1-decay)*value
        y[j] = previous
    return y


def main():
    mechanics_fast = filter_first_order(F, TAU_FAST_MS)
    mechanics_slow = filter_first_order(F, TAU_SLOW_MS)
    current_fast_mech = G*filter_first_order(mechanics_fast, TAU_SLOW_MS)
    current_slow_mech = G*filter_first_order(mechanics_slow, TAU_FAST_MS)
    dense = np.arange(len(T))
    sparse = np.arange(0, len(T), 40)
    diff = mechanics_fast - mechanics_slow
    phase_chisq = []
    for onset in range(40):
        shifted_force = np.zeros_like(F)
        shifted_force[(T >= onset) & (T < onset+80)] = 1.0
        shifted_diff = (filter_first_order(shifted_force, TAU_FAST_MS)
                        - filter_first_order(shifted_force, TAU_SLOW_MS))
        phase_chisq.append(float(np.sum((shifted_diff[sparse]/SD_M)**2)))
    result = {
        "synthetic": True,
        "model": "tau_m dm/dt=F-m; tau_g dq/dt=m-q; I=Gq",
        "time_constants_ms": [TAU_FAST_MS, TAU_SLOW_MS],
        "force_protocol": "unit pulse from 5 to 85 ms, total window 350 ms, 1 ms grid",
        "max_abs_current_difference_after_swapping_ms_pA": float(np.max(np.abs(
            current_fast_mech-current_slow_mech))),
        "max_abs_intermediate_mechanics_difference": float(np.max(np.abs(diff))),
        "expected_intermediate_delta_chisq_at_sd_0p05": {
            "1ms_sampling": float(np.sum((diff[dense]/SD_M)**2)),
            "40ms_sampling": float(np.sum((diff[sparse]/SD_M)**2)),
        },
        "40ms_sample_phase_sensitivity_for_80ms_pulse": {
            "min_median_max_delta_chisq": list(map(float,
                [np.min(phase_chisq), np.median(phase_chisq), np.max(phase_chisq)])),
            "fraction_below_unit_expected_noise_separation": float(np.mean(np.asarray(phase_chisq) < 1.0)),
        },
        "qualifier": "Sampling contrast depends on pulse phase and assumed noise; exact current equivalence follows from linear filter commutation for any input.",
    }
    (HERE / "dynamic_label_swap_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True, constrained_layout=True)
    axes[0].plot(T, F, color="black")
    axes[0].set(ylabel="Input F (scaled)")
    axes[1].plot(T, current_fast_mech, label="tau_m=2, tau_g=10 ms")
    axes[1].plot(T, current_slow_mech, ls="--", label="tau_m=10, tau_g=2 ms")
    axes[1].set(ylabel="Output I (pA)")
    axes[1].legend(fontsize=8)
    axes[2].plot(T, mechanics_fast, label="mechanics tau=2 ms")
    axes[2].plot(T, mechanics_slow, label="mechanics tau=10 ms")
    axes[2].scatter(T[sparse], mechanics_fast[sparse], color="black", s=12,
                    label="40 ms sample positions")
    axes[2].set(xlabel="Time (ms)", ylabel="Intermediate m")
    axes[2].legend(fontsize=8)
    fig.savefig(HERE / "dynamic_label_swap_audit.png", dpi=180)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
