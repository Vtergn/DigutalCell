"""Separate persistent cell response scale from short-term residual prediction.

All coefficient fits leave the target cell out. A target cell's first eligible
sweep sets its transfer curve; later observed currents enter only as sequential
updates, never as information about the current test sweep.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from one_shot_work import read_metadata, first_calibration, load_approaches, gain_from_sweep, waveform


HERE = Path(__file__).resolve().parent


def predict_ridge(train: list[dict], test: dict, features: tuple[str, ...]) -> float:
    design = np.asarray([[r[key] for key in features] for r in train])
    x = np.asarray([test[key] for key in features])
    center, scale = design.mean(axis=0), np.maximum(design.std(axis=0), 1e-6)
    matrix = np.column_stack([np.ones(len(train)), (design - center) / scale])
    target = np.asarray([r["residual"] for r in train])
    penalty = np.diag([0.0] + [5.0] * len(features))
    coefficients = np.linalg.solve(matrix.T @ matrix + penalty, matrix.T @ target)
    return float(np.r_[1, (x - center) / scale] @ coefficients)


def main(holdout_date: bool = False) -> None:
    by_cell = read_metadata()
    traces = load_approaches(by_cell)
    calibrations = {cell: first_calibration(rows) for cell, rows in by_cell.items()}
    observations = []
    for cell, rows in by_cell.items():
        calibration = calibrations[cell]
        if calibration is None:
            continue
        channel = calibration["channel"]
        train = [x for other, x in calibrations.items() if other != cell and x is not None and x["channel"] == channel]
        if len(train) < 4:
            continue
        alpha = float(np.median([float(x["work_alpha"]) for x in train]))
        threshold = float(calibration["wthresh_fJ"])
        cal_sweep = int(calibration["sweep"])
        trace = traces[(cell, cal_sweep)]
        gain = gain_from_sweep(trace["work"], trace["current"], threshold, alpha)
        residuals = []
        work_history = []
        cal_voltage = float(np.median(trace["voltage"]))
        for row in rows:
            sweep = int(row["sweep"])
            if sweep < cal_sweep:
                continue
            current_trace = traces[(cell, sweep)]
            if abs(float(np.median(current_trace["voltage"])) - cal_voltage) > 5:
                break
            observed = max(float(row["current_peak_minus_offset_pA"]), 0.0)
            predicted = float(waveform(np.asarray([float(row["peak_work_fJ"])]), gain, alpha, threshold)[0])
            log_predicted = float(np.log1p(predicted))
            residual = float(np.log1p(observed) - log_predicted)
            if sweep > cal_sweep and len(residuals) >= 1:
                observations.append({
                    "cell_id": cell, "channel": channel, "date": row["date"], "sweep": sweep,
                    "future_ordinal": len(residuals),
                    "observed_peak_pA": observed, "static_peak_pA": predicted,
                    "log_static": log_predicted, "residual": residual,
                    "previous_residual": residuals[-1],
                    "running_mean_residual": float(np.mean(residuals)),
                    "older_mean_residual": float(np.mean(residuals[:-1])) if len(residuals) >= 2 else float("nan"),
                    "recent_innovation": residuals[-1] - float(np.mean(residuals[:-1])) if len(residuals) >= 2 else float("nan"),
                    "delta_log_work": float(np.log(float(row["peak_work_fJ"]) / float(calibration["peak_work_fJ"]))),
                    "previous_log_work": float(np.log(work_history[-1] / float(calibration["peak_work_fJ"]))),
                })
            residuals.append(residual)
            work_history.append(float(row["peak_work_fJ"]))
    output = []
    for test in observations:
        train = [r for r in observations if r["channel"] == test["channel"] and r["cell_id"] != test["cell_id"]
                 and (not holdout_date or r["date"] != test["date"])]
        if len(train) < 4:
            continue
        candidates = {
            "static": 0.0,
            "pooled_bias": predict_ridge(train, test, ("delta_log_work", "previous_log_work")),
            "previous_residual": predict_ridge(train, test, ("delta_log_work", "previous_log_work", "previous_residual")),
            "running_mean": predict_ridge(train, test, ("delta_log_work", "previous_log_work", "running_mean_residual")),
        }
        if test["future_ordinal"] >= 3:
            older = [r for r in train if r["future_ordinal"] >= 3]
            if len(older) >= 4:
                candidates["older_mean"] = predict_ridge(older, test, ("delta_log_work", "previous_log_work", "older_mean_residual"))
                candidates["older_mean_plus_innovation"] = predict_ridge(
                    older, test, ("delta_log_work", "previous_log_work", "older_mean_residual", "recent_innovation")
                )
        for model, correction in candidates.items():
            log_prediction = test["log_static"] + correction
            output.append({
                **{k: test[k] for k in ("cell_id", "channel", "date", "sweep", "future_ordinal", "observed_peak_pA", "static_peak_pA", "residual", "previous_residual", "running_mean_residual")},
                "model": model, "predicted_peak_pA": float(max(np.expm1(log_prediction), 0)),
                "absolute_log_error": abs(float(np.log1p(test["observed_peak_pA"])) - log_prediction),
            })
    filename = "previous_residual_predictions_date_heldout.csv" if holdout_date else "previous_residual_predictions.csv"
    with (HERE / filename).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    for channel in sorted({r["channel"] for r in output}):
        for model in ("static", "pooled_bias", "previous_residual", "running_mean", "older_mean", "older_mean_plus_innovation"):
            subset = [r for r in output if r["channel"] == channel and r["model"] == model]
            if not subset:
                continue
            by_cell_error = defaultdict(list)
            for row in subset:
                by_cell_error[row["cell_id"]].append(row["absolute_log_error"])
            print("date_heldout" if holdout_date else "cell_heldout", channel, model, "cells", len(by_cell_error), "sweeps", len(subset),
                  "median_cell_mean_absolute_log_error", round(float(np.median([np.mean(x) for x in by_cell_error.values()])), 3))


if __name__ == "__main__":
    main()
    main(holdout_date=True)
