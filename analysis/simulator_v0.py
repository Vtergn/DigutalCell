"""Combine frozen one-shot amplitude and causal shape modules under voltage clamp.

No future current is used to predict a sweep. A1/A2 amplitudes come from the
whole-date-held-out amplitude_scaling.py fits; adaptation parameters come from
the earlier whole-date-held-out adaptation_nested.py fits. The two modules are
not refit jointly here.
"""

from __future__ import annotations

import csv

import numpy as np

from adaptation_nested import basis, drive, force_threshold, postpeak_score
from dynamic_input_screen import HERE, load_traces, score
from one_shot_work import first_calibration, read_metadata


def normalized_shape(stimulus: np.ndarray, tau_ms: float | None = None, beta: float = 0.0) -> np.ndarray:
    shape = basis(stimulus, tau_ms, beta)
    peak = float(np.max(shape))
    return shape/peak if peak > 0 else np.zeros_like(shape)


def main() -> None:
    with (HERE / "amplitude_scaling_scores.csv").open(newline="", encoding="utf-8") as handle:
        amplitudes = list(csv.DictReader(handle))
    with (HERE / "adaptation_nested_fits.csv").open(newline="", encoding="utf-8") as handle:
        adaptation_fits = {(r["channel"], r["heldout_date"]): r for r in csv.DictReader(handle)}
    amplitudes_by_sweep = {}
    for row in amplitudes:
        amplitudes_by_sweep.setdefault((row["cell_id"], int(row["test_sweep"])), {})[row["model"]] = row
    by_cell = read_metadata()
    calibration = {cell: first_calibration(rows) for cell, rows in by_cell.items()}
    traces, _ = load_traces(by_cell, calibration)
    results = []
    for (cell, sweep), models in amplitudes_by_sweep.items():
        cal_row = calibration[cell]
        cal_trace = traces[(cell, int(cal_row["sweep"]))]
        trace = traces[(cell, sweep)]
        threshold = force_threshold(cal_row, cal_trace)
        cal_force_peak = float(np.max(cal_trace["F"][:150]))
        stimulus = drive(trace["F"], threshold, cal_force_peak)
        q_force = normalized_shape(stimulus)
        selected = adaptation_fits[(cal_row["channel"], cal_row["date"])]
        beta = float(selected["beta"])
        tau_ms = float(selected["tau_ms"]) if beta else None
        q_adapt = normalized_shape(stimulus, tau_ms, beta)
        amplitude_linear = float(models["A1_force_linear"]["predicted_peak_pA"])
        amplitude_power = float(models["A2_force_power"]["predicted_peak_pA"])
        predicted = {
            "S0_zero": np.zeros_like(trace["current"]),
            "S1_linear_force_shape": amplitude_linear*q_force,
            "S2_power_force_shape": amplitude_power*q_force,
            "S3_linear_adapt_shape": amplitude_linear*q_adapt,
            "S4_power_adapt_shape": amplitude_power*q_adapt,
        }
        event_threshold = max(10.0, 5.0*float(cal_row["stdev_pA"]))
        for model, waveform in predicted.items():
            results.append({
                "cell_id": cell, "channel": cal_row["channel"], "date": cal_row["date"],
                "test_sweep": sweep, "model": model,
                "amplitude_linear_pA": amplitude_linear,
                "amplitude_power_pA": amplitude_power,
                "force_exponent": float(models["A2_force_power"]["gamma_force"]),
                "adapt_tau_ms": tau_ms if tau_ms is not None else np.nan,
                "adapt_beta": beta,
                **score(trace["current"], waveform, event_threshold, int(trace["contact_ms"])),
                **postpeak_score(trace["current"], waveform, event_threshold, int(trace["contact_ms"])),
            })
    with (HERE / "simulator_v0_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print("Wrote", len(results), "model scores for", len(amplitudes_by_sweep), "held-out future sweeps")


if __name__ == "__main__":
    main()
