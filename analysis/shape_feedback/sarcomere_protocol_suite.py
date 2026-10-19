"""Action-conditioned protocol audit on the upstream-sarcomere closed loop.

The original single-pulse trace is a regression reference. Long and compound
mechanical inputs are new interventions. Linear pulse superposition uses only
the baseline and original short-pulse traces, making its predictions target
blind. Physical closed-loop trajectories are same-model oracle references.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from blind_closed_loop_benchmark import reconstruct_step, shift_step
from sarcomere_closed_loop import ClosedLoop, initial_state, run

HERE = Path(__file__).resolve().parent
OUT = HERE / "sarcomere_protocol_suite"
ORIGINAL = HERE / "sarcomere_closed_loop.npz"
PULSES = {
    "long_hold": [(-5.0, 250.0, 400.0)],
    "compound": [(-5.0, 250.0, 300.0), (-5.0, 350.0, 400.0)],
}


def modalities(state, lam, tension):
    return np.column_stack([state[:, 0], 1000*state[:, 5], tension, lam])


def superpose(step, pulses, length):
    answer = np.zeros((length, step.shape[1]), dtype=float)
    for amplitude, onset, offset in pulses:
        answer += amplitude * (shift_step(step, round(onset), length)
                               - shift_step(step, round(offset), length))
    return answer


def main():
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    with np.load(ORIGINAL) as old:
        baseline = modalities(old["no_external_pulse_state"],
                              old["no_external_pulse_lambda"],
                              old["no_external_pulse_tension_kpa"])
        calibration = modalities(old["closed_pulse_state"],
                                 old["closed_pulse_lambda"],
                                 old["closed_pulse_tension_kpa"])
    y0 = initial_state(20)
    same = run(ClosedLoop(n=20), y0, pulse=(-5., 250., 350.))
    same_wave = modalities(same["state"], same["lambda"], same["tension_kpa"])
    regression_max = np.max(np.abs(same_wave-calibration), axis=0)
    if not np.all(regression_max < np.array([1e-7, 1e-9, 1e-7, 1e-9])):
        raise AssertionError(f"Original single-pulse result changed: {regression_max}")
    cal_delta = calibration-baseline
    step = reconstruct_step(cal_delta, -5., 250, 350)
    results = {}
    fig, axes = plt.subplots(4, 2, figsize=(12, 9), sharex=True,
                             layout="constrained")
    for col, (label, pulses) in enumerate(PULSES.items()):
        solution = run(ClosedLoop(n=20), y0, pulses=pulses)
        truth = modalities(solution["state"], solution["lambda"],
                           solution["tension_kpa"])-baseline
        pred = superpose(step, pulses, len(truth))
        absolute = np.abs(truth-pred)
        results[label] = {
            "pulses": pulses,
            "MAE_V_mV_Ca_uM_T_kPa_lambda": absolute.mean(axis=0).tolist(),
            "max_abs_error_V_mV_Ca_uM_T_kPa_lambda": absolute.max(axis=0).tolist(),
            "max_abs_true_response_V_mV_Ca_uM_T_kPa_lambda": np.max(np.abs(truth), axis=0).tolist(),
            "runtime_s": solution["runtime_s"],
            "rhs_calls": solution["rhs_calls"],
        }
        np.savez_compressed(OUT/f"{label}.npz", truth=truth, linear=pred,
                            state=solution["state"],
                            local_lambda=solution["local_lambda"])
        for ch, ax in enumerate(axes[:, col]):
            ax.plot(np.arange(len(truth)), truth[:, ch], color="#242b38",
                    lw=1.5, label="native closed loop")
            ax.plot(np.arange(len(pred)), pred[:, ch], color="#d07a32",
                    lw=1.1, label="calibration-only linear")
            ax.set_ylabel(("ΔV (mV)", "ΔCa (µM)", "ΔT (kPa)", "Δλ")[ch])
            ax.grid(alpha=.2)
        axes[0, col].set_title(label.replace("_", " "))
        axes[-1, col].set_xlabel("Time (ms)")
    axes[0, 1].legend(fontsize=8)
    fig.suptitle("Unseen dynamic interventions in the 90-state hybrid model")
    fig.savefig(OUT/"protocol_comparison.png", dpi=180)
    plt.close(fig)
    report = {
        "calibration": "Original -5 abstract prestrain, 250--350 ms",
        "prediction_input": "Observed no-load and calibration V/Ca/T/lambda; target pulse schedule",
        "target_access": "None for linear prediction; native model used only as same-equation reference",
        "single_pulse_regression_max_V_mV_Ca_uM_T_kPa_lambda": regression_max.tolist(),
        "results": results,
        "limits": "One nominal virtual cell; prestrain units abstract; same-model target; no paired real AP/Ca/force/shape validation",
    }
    (OUT/"results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
