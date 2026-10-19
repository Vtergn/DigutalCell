"""Whole-date-held-out nested force threshold and effective adaptation models.

Uses the frozen 1 kHz mechanical/contact and precontact-current preprocessing
from dynamic_input_screen.py. The adaptive state is an effective within-sweep
description; no intersweep recovery constant is fit or inferred.
"""

from __future__ import annotations

import csv
from collections import defaultdict

import numpy as np

from dynamic_input_screen import HERE, load_traces, score, onset_ms
from one_shot_work import read_metadata, first_calibration


TAU_MS = (2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0)
BETA = (0.0, 0.25, 0.5, 0.75, 1.0)


def force_threshold(calibration: dict, trace: dict) -> float:
    onset = float(calibration["threshold_time_ms"]) - (float(calibration["compression_end_ms"])-150)
    return float(trace["F"][int(np.clip(round(onset), 0, 149))])


def drive(force: np.ndarray, threshold: float, calibration_peak: float) -> np.ndarray:
    span = calibration_peak - threshold
    if span <= 0:
        raise ValueError("Calibration force must exceed its response threshold")
    return np.maximum((force-threshold)/span, 0.0)


def basis(stimulus: np.ndarray, tau_ms: float | None = None, beta: float = 0.0) -> np.ndarray:
    if tau_ms is None or beta == 0:
        return stimulus.copy()
    state = 0.0
    adaptation_fraction = -np.expm1(-1.0/tau_ms)
    output = np.empty_like(stimulus)
    for index, value in enumerate(stimulus):
        state += adaptation_fraction*(float(value)-state)
        output[index] = max(float(value)-beta*state, 0.0)
    return output


def fit_gain(shape: np.ndarray, current: np.ndarray, method: str) -> float:
    positive_current = np.maximum(current, 0.0)
    if method == "peak":
        return float(np.max(positive_current)/max(float(np.max(shape)), 1e-12))
    if method != "least_squares":
        raise ValueError(f"Unknown calibration method: {method}")
    return float(max(np.dot(shape, positive_current)/max(np.dot(shape, shape), 1e-12), 0.0))


def training_loss(cells: list[str], calibrated: dict, tau_ms: float | None, beta: float, method: str) -> float:
    losses = []
    for cell in cells:
        item = calibrated[cell]
        predicted_shape = basis(item["cal_drive"], tau_ms, beta)
        gain = fit_gain(predicted_shape, item["cal_current"], method)
        loss = np.mean(np.abs(item["cal_current"]-gain*predicted_shape))/item["cal_peak_current"]
        losses.append(float(loss))
    return float(np.mean(losses))


def postpeak_score(actual: np.ndarray, predicted: np.ndarray, threshold: float, contact: int) -> dict:
    if not np.isfinite(onset_ms(actual, threshold, contact-5)):
        return {"postpeak_0_50ms_mae_pA": float("nan"),
                "postpeak_0_50ms_signed_bias_pA": float("nan"),
                "postpeak_0_50ms_charge_abs_error_fC": float("nan"),
                "oracle_peak_scaled_postpeak_shape_mae": float("nan"),
                "peak_amplitude_ratio_actual_predicted": float("nan")}
    start = max(contact-5, 0)
    peak = start+int(np.argmax(actual[start:]))
    end = min(peak+50, len(actual))
    delta = predicted[peak:end]-actual[peak:end]
    actual_peak = max(float(np.max(actual[start:])), 1e-12)
    predicted_peak = float(np.max(predicted[start:]))
    shape_error = (
        float(np.mean(np.abs(actual[peak:end]/actual_peak-predicted[peak:end]/predicted_peak)))
        if predicted_peak > 1e-9 else float("nan")
    )
    return {"postpeak_0_50ms_mae_pA": float(np.mean(np.abs(delta))),
            "postpeak_0_50ms_signed_bias_pA": float(np.mean(delta)),
            "postpeak_0_50ms_charge_abs_error_fC": float(abs(np.sum(delta))),
            # Diagnostic only: this uses the test peak and is not a forecast.
            "oracle_peak_scaled_postpeak_shape_mae": shape_error,
            "peak_amplitude_ratio_actual_predicted": actual_peak/predicted_peak if predicted_peak > 1e-9 else float("nan")}


def main(gain_method: str = "least_squares") -> None:
    by_cell = read_metadata()
    calibration = {cell: first_calibration(rows) for cell, rows in by_cell.items()}
    traces, _ = load_traces(by_cell, calibration)
    calibrated = {}
    for cell, row in calibration.items():
        if row is None or row["channel"] not in {"mp1", "mp2"}:
            continue
        cal_sweep = int(row["sweep"])
        trace = traces[(cell, cal_sweep)]
        threshold = force_threshold(row, trace)
        peak_force = float(np.max(trace["F"]))
        calibrated[cell] = {
            "cell_id": cell, "channel": row["channel"], "date": row["date"],
            "cal_sweep": cal_sweep, "cal_voltage": trace["voltage_mV"],
            "force_threshold_nN": threshold, "cal_peak_force_nN": peak_force,
            "cal_current": trace["current"],
            "cal_peak_current": max(float(np.max(trace["current"])), 10.0),
            "event_threshold_pA": max(10.0, 5.0*float(row["stdev_pA"])),
            "cal_drive": drive(trace["F"], threshold, peak_force),
        }
    fit_rows, score_rows = [], []
    for channel in ("mp1", "mp2"):
        test_dates = sorted({item["date"] for item in calibrated.values() if item["channel"] == channel})
        for date in test_dates:
            targets = [cell for cell, item in calibrated.items() if item["channel"] == channel and item["date"] == date
                       and any(int(row["sweep"]) > item["cal_sweep"] and
                               abs(traces[(cell, int(row["sweep"]))]["voltage_mV"]-item["cal_voltage"]) <= 5
                               for row in by_cell[cell])]
            if not targets:
                continue
            train = [cell for cell, item in calibrated.items() if item["channel"] == channel and item["date"] != date]
            if len(train) < 4:
                continue
            loss_m1 = training_loss(train, calibrated, None, 0.0, gain_method)
            candidates = [(training_loss(train, calibrated, tau, beta, gain_method), tau, beta) for tau in TAU_MS for beta in BETA]
            loss_m2, tau, beta = min(candidates)
            fit_rows.append({"channel": channel, "heldout_date": date, "train_cells": len(train),
                             "test_cells": len(targets), "tau_ms": tau if beta > 0 else float("nan"),
                             "beta": beta, "selected_adaptation": int(beta > 0),
                             "amplitude_method": gain_method,
                             "train_normalized_mae_m1": loss_m1, "train_normalized_mae_m2": loss_m2})
            for cell in targets:
                item = calibrated[cell]
                m1_shape_cal = basis(item["cal_drive"])
                m2_shape_cal = basis(item["cal_drive"], tau, beta)
                gain_m1 = fit_gain(m1_shape_cal, item["cal_current"], gain_method)
                gain_m2 = fit_gain(m2_shape_cal, item["cal_current"], gain_method)
                for row in by_cell[cell]:
                    sweep = int(row["sweep"])
                    if sweep <= item["cal_sweep"]:
                        continue
                    test = traces[(cell, sweep)]
                    if abs(test["voltage_mV"]-item["cal_voltage"]) > 5:
                        continue
                    stimulus = drive(test["F"], item["force_threshold_nN"], item["cal_peak_force_nN"])
                    original_gate = test["F"] >= item["force_threshold_nN"]
                    models = {
                        "M0_zero": np.zeros_like(test["current"]),
                        "M1_threshold_force": gain_m1*basis(stimulus),
                        "M2_effective_adaptation": gain_m2*basis(stimulus, tau, beta),
                        "previous_force_linear": item["cal_peak_current"]*test["F"]/item["cal_peak_force_nN"]*original_gate,
                    }
                    for model, predicted in models.items():
                        score_rows.append({
                            "cell_id": cell, "channel": channel, "date": date, "test_sweep": sweep,
                            "model": model, "amplitude_method": gain_method,
                            "gain_from_calibration_pA": gain_m1 if model == "M1_threshold_force" else gain_m2 if model == "M2_effective_adaptation" else float("nan"),
                            "shared_tau_ms": tau if beta > 0 else float("nan"), "shared_beta": beta,
                            **score(test["current"], predicted, item["event_threshold_pA"], test["contact_ms"]),
                            **postpeak_score(test["current"], predicted, item["event_threshold_pA"], test["contact_ms"]),
                        })
    suffix = "_peak_calibration" if gain_method == "peak" else ""
    for filename, rows in ((f"adaptation_nested_fits{suffix}.csv", fit_rows), (f"adaptation_nested_scores{suffix}.csv", score_rows)):
        with (HERE / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for channel in ("mp1", "mp2"):
        for model in ("M0_zero", "M1_threshold_force", "M2_effective_adaptation", "previous_force_linear"):
            rows = [r for r in score_rows if r["channel"] == channel and r["model"] == model]
            grouped = defaultdict(list)
            for row in rows:
                grouped[row["cell_id"]].append(row)
            print(gain_method, channel, model, "cells", len(grouped), "sweeps", len(rows),
                  "median_cell_mean_mae_pA", round(float(np.median([np.mean([r["mae_pA"] for r in values]) for values in grouped.values()])), 2),
                  "median_cell_mean_postpeak_mae_pA", round(float(np.median([np.nanmean([r["postpeak_0_50ms_mae_pA"] for r in values]) for values in grouped.values()])), 2))


if __name__ == "__main__":
    main()
    main(gain_method="peak")
