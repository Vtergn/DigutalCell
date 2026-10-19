"""Prospective peak-current scaling after one responsive calibration sweep.

Uses the frozen traces/cohort from adaptation_nested.py. Exponents are selected
using future sweeps of other-date cells only; each held-out date is untouched.
"""

from __future__ import annotations

import csv
from collections import defaultdict

import numpy as np

from adaptation_nested import force_threshold
from dynamic_input_screen import HERE, load_traces, onset_ms
from one_shot_work import first_calibration, read_metadata


EXPONENTS = np.arange(0.05, 4.001, 0.05)
MODELS = ("A0_cal_peak", "A1_force_linear", "A2_force_power", "A3_work_power")


def mechanical_ratio(peak: float, calibration_peak: float, threshold: float) -> float:
    span = calibration_peak - threshold
    if not np.isfinite(span) or span <= 0:
        raise ValueError("Calibration peak must exceed its threshold")
    return max(peak-threshold, 0.0)/span


def select_exponent(train_cells: list[str], records: list[dict], key: str) -> tuple[float, float, int]:
    """Minimize cell-balanced mean absolute log1p peak error on training dates."""
    by_cell = defaultdict(list)
    for row in records:
        if row["cell_id"] in train_cells and row[key] > 0:
            by_cell[row["cell_id"]].append(row)
    if not by_cell:
        raise ValueError("No training cells with valid positive mechanical ratios")
    best = None
    for exponent in EXPONENTS:
        errors = []
        for observations in by_cell.values():
            sweep_errors = [abs(np.log1p(row["cal_peak_pA"]*row[key]**exponent)
                                - np.log1p(row["actual_peak_pA"])) for row in observations]
            errors.append(float(np.mean(sweep_errors)))
        candidate = (float(np.mean(errors)), float(exponent))
        if best is None or candidate < best:
            best = candidate
    return best[1], best[0], len(by_cell)


def main() -> None:
    by_cell = read_metadata()
    calibration = {cell: first_calibration(rows) for cell, rows in by_cell.items()}
    traces, _ = load_traces(by_cell, calibration)
    records = []
    for cell, row in calibration.items():
        if row is None or row["channel"] not in {"mp1", "mp2"}:
            continue
        cal_sweep = int(row["sweep"])
        cal_trace = traces[(cell, cal_sweep)]
        cal_voltage = cal_trace["voltage_mV"]
        contact = max(int(cal_trace["contact_ms"])-5, 0)
        cal_peak_pA = max(float(np.max(cal_trace["current"][contact:])), 0.0)
        force_threshold_nN = force_threshold(row, cal_trace)
        cal_force_peak = float(np.max(cal_trace["F"][:150]))
        work_threshold_fJ = float(row["wthresh_fJ"])
        cal_work_peak = float(np.max(cal_trace["W"][:150]))
        for future in by_cell[cell]:
            sweep = int(future["sweep"])
            if sweep <= cal_sweep:
                continue
            trace = traces[(cell, sweep)]
            if abs(trace["voltage_mV"]-cal_voltage) > 5:
                continue
            future_contact = max(int(trace["contact_ms"])-5, 0)
            actual_peak = max(float(np.max(trace["current"][future_contact:])), 0.0)
            force_peak = float(np.max(trace["F"][:150]))
            work_peak = float(np.max(trace["W"][:150]))
            records.append({
                "cell_id": cell, "channel": row["channel"], "date": row["date"],
                "test_sweep": sweep, "cal_sweep": cal_sweep,
                "cal_peak_pA": cal_peak_pA, "actual_peak_pA": actual_peak,
                "cal_force_peak_nN": cal_force_peak, "test_force_peak_nN": force_peak,
                "force_threshold_nN": force_threshold_nN,
                "cal_work_peak_fJ": cal_work_peak, "test_work_peak_fJ": work_peak,
                "work_threshold_fJ": work_threshold_fJ,
                "force_ratio": mechanical_ratio(force_peak, cal_force_peak, force_threshold_nN),
                "work_ratio": mechanical_ratio(work_peak, cal_work_peak, work_threshold_fJ),
                "actual_has_event": int(np.isfinite(onset_ms(
                    trace["current"], max(10.0, 5.0*float(row["stdev_pA"])),
                    trace["contact_ms"]-5))),
            })
    fits, scores = [], []
    for channel in ("mp1", "mp2"):
        channel_records = [r for r in records if r["channel"] == channel]
        for date in sorted({r["date"] for r in channel_records}):
            test = [r for r in channel_records if r["date"] == date]
            train = sorted({r["cell_id"] for r in channel_records if r["date"] != date})
            gamma, force_loss, force_train_cells = select_exponent(train, channel_records, "force_ratio")
            alpha, work_loss, work_train_cells = select_exponent(train, channel_records, "work_ratio")
            fits.append({"channel": channel, "heldout_date": date, "train_cells": len(train),
                         "test_cells": len({r["cell_id"] for r in test}), "gamma_force": gamma,
                         "alpha_work": alpha, "force_train_cells": force_train_cells,
                         "work_train_cells": work_train_cells,
                         "train_log1p_loss_force": force_loss, "train_log1p_loss_work": work_loss})
            for row in test:
                predictions = {
                    "A0_cal_peak": row["cal_peak_pA"],
                    "A1_force_linear": row["cal_peak_pA"]*row["force_ratio"],
                    "A2_force_power": row["cal_peak_pA"]*row["force_ratio"]**gamma,
                    "A3_work_power": row["cal_peak_pA"]*row["work_ratio"]**alpha,
                }
                for model, predicted in predictions.items():
                    actual = row["actual_peak_pA"]
                    scores.append({**row, "model": model, "predicted_peak_pA": predicted,
                                   "abs_error_pA": abs(actual-predicted),
                                   "signed_error_pA": predicted-actual,
                                   "log1p_error": abs(np.log1p(predicted)-np.log1p(actual)),
                                   "log_ratio_error_responders": abs(np.log(predicted/actual))
                                   if row["actual_has_event"] and predicted > 0 and actual > 0 else np.nan,
                                   "actual_over_predicted": actual/predicted if predicted > 0 else np.nan,
                                   "gamma_force": gamma, "alpha_work": alpha})
    for filename, rows in (("amplitude_scaling_fits.csv", fits), ("amplitude_scaling_scores.csv", scores)):
        with (HERE / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print("Wrote", len(scores), "scores for", len(records), "future sweeps in", len(fits), "date folds")


if __name__ == "__main__":
    main()
