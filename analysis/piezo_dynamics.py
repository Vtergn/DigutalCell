"""Held-out-cell Piezo activation/inactivation waveform experiment.

Fits channel rate constants on other cells' calibration sweeps only. The cell's
first eligible sweep provides threshold and amplitude. Recovery is never fit.
Sequential availability uses the previous *observed* peak, so it is a distinct
task from one-shot prediction and is not a measured channel state.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

from one_shot_work import read_metadata, first_calibration, gain_from_sweep, waveform
from three_state import Rates, stimulate


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "Data" / "2022_pnas" / "2022_pnas" / "bychannel"
RATE_GRID = [(opening, inactivation) for opening in (30.0, 100.0, 300.0) for inactivation in (3.0, 10.0, 30.0, 100.0)]


def load_traces(by_cell: dict, calibrations: dict) -> dict[tuple[str, int], dict]:
    with (ROOT / "bychannel_summary.csv").open(newline="", encoding="utf-8-sig") as handle:
        selected = list(csv.DictReader(handle))
    paths = {path.name: path for path in ROOT.rglob("*_preprocessed.feather")}
    output = {}
    for chosen in selected:
        cell = chosen["uniqueID"]
        if calibrations[cell] is None or chosen["construct"] not in {"mp1", "mp2"}:
            continue
        path = paths[Path(chosen["path"]).name]
        with path.with_name(path.name.replace("_preprocessed.feather", "_summary.csv")).open(newline="", encoding="utf-8-sig") as handle:
            meta = {int(row["sweep"]): row for row in csv.DictReader(handle)}
        table = feather.read_table(path, columns=["sweep", "ti", "work", "i_blsub", "v"])
        data = {key: table[key].to_numpy() for key in table.column_names}
        for row in by_cell[cell]:
            sweep = int(row["sweep"])
            if sweep < int(calibrations[cell]["sweep"]):
                continue
            start = float(row["compression_end_ms"]) - 150
            mask = ((data["sweep"] == sweep) & (data["ti"] >= start-1e-6) & (data["ti"] <= start+350+1e-6))
            if np.count_nonzero(mask) != 8751:
                raise ValueError(f"Unexpected full window for {cell} sweep {sweep}")
            output[(cell, sweep)] = {
                "work": np.maximum(data["work"][mask][::25], 0),
                "current": np.abs(data["i_blsub"][mask][::25]) - float(meta[sweep]["offset"]),
                "voltage": float(np.median(data["v"][mask])),
            }
    return output


def open_shape(work: np.ndarray, threshold: float, alpha: float,
               opening: float, inactivation: float, availability: float = 1.0) -> np.ndarray:
    # The channel-level alpha shapes the work drive. W is cantilever-estimated
    # mechanical work, not directly measured membrane tension.
    relative = np.maximum((work - threshold) / max(threshold, 1.0), 0.0)
    drive = 20.0 * relative ** alpha
    rates = Rates(opening, inactivation, 0.0, 20.0)
    opened, _ = stimulate(drive, 0.0, 0.001, rates, (availability, 0.0, 1-availability))
    return opened


def fit_amplitude(shape: np.ndarray, current: np.ndarray) -> float:
    positive = np.maximum(current, 0)
    return float(max(np.dot(shape, positive) / max(np.dot(shape, shape), 1e-12), 0))


def score_1ms(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    event = actual > max(10.0, 5 * float(np.std(actual[:20])))
    difference = np.abs(actual - predicted)
    return {
        "mae_pA": float(np.mean(difference)),
        "event_mae_pA": float(np.mean(difference[event])) if np.any(event) else float("nan"),
        "peak_abs_error_pA": float(abs(np.max(actual) - np.max(predicted))),
        "charge_abs_error_fC": float(abs(np.sum(np.maximum(actual, 0)) - np.sum(predicted))),
        "false_positive_charge_fC": float(np.sum(predicted[~event])),
    }


def cal_loss(trace: dict, threshold: float, alpha: float, opening: float, inactivation: float) -> float:
    shape = open_shape(trace["work"], threshold, alpha, opening, inactivation)
    amplitude = fit_amplitude(shape, trace["current"])
    return float(np.mean(np.abs(trace["current"] - amplitude*shape)) / max(np.max(trace["current"]), 10.0))


def main() -> None:
    by_cell = read_metadata()
    calibrations = {cell: first_calibration(rows) for cell, rows in by_cell.items()}
    traces = load_traces(by_cell, calibrations)
    output = []
    fit_rows = []
    for cell, calibration in calibrations.items():
        if calibration is None or calibration["channel"] not in {"mp1", "mp2"}:
            continue
        cal_sweep = int(calibration["sweep"])
        later = [r for r in by_cell[cell] if int(r["sweep"]) > cal_sweep]
        if not later:
            continue
        channel = calibration["channel"]
        train = [(other, row) for other, row in calibrations.items()
                 if other != cell and row is not None and row["channel"] == channel]
        alpha = float(np.median([float(row["work_alpha"]) for _, row in train]))
        losses = []
        for opening, inactivation in RATE_GRID:
            losses.append(float(np.mean([
                cal_loss(traces[(other, int(row["sweep"]))], float(row["wthresh_fJ"]), alpha, opening, inactivation)
                for other, row in train
            ])))
        opening, inactivation = RATE_GRID[int(np.argmin(losses))]
        threshold = float(calibration["wthresh_fJ"])
        cal = traces[(cell, cal_sweep)]
        cal_shape = open_shape(cal["work"], threshold, alpha, opening, inactivation)
        amplitude = fit_amplitude(cal_shape, cal["current"])
        static_gain = gain_from_sweep(cal["work"], cal["current"], threshold, alpha)
        fit_rows.append({"cell_id": cell, "channel": channel, "alpha": alpha,
                         "opening_per_s": opening, "inactivation_per_s": inactivation,
                         "amplitude_pA": amplitude, "calibration_sweep": cal_sweep})
        previous_row = calibration
        for row in later:
            sweep = int(row["sweep"])
            test = traces[(cell, sweep)]
            if abs(test["voltage"] - cal["voltage"]) > 5:
                break
            # Prospective update from previous observed peak. It is bounded as
            # a state fraction; its biological identity remains unverified.
            previous_trace = traces[(cell, int(previous_row["sweep"]))]
            previous_shape = open_shape(previous_trace["work"], threshold, alpha, opening, inactivation)
            previous_predicted_peak = amplitude * float(np.max(previous_shape))
            previous_observed_peak = max(float(previous_row["current_peak_minus_offset_pA"]), 0.0)
            availability = float(np.clip(previous_observed_peak / max(previous_predicted_peak, 1.0), 0.0, 1.0))
            shape = open_shape(test["work"], threshold, alpha, opening, inactivation)
            candidates = {
                "zero": np.zeros_like(test["current"]),
                "static_power": waveform(test["work"], static_gain, alpha, threshold),
                "dynamic_q1": amplitude*shape,
                "dynamic_previous_q": amplitude*open_shape(test["work"], threshold, alpha, opening, inactivation, availability),
            }
            actual = test["current"]
            actual_peak_index = int(np.argmax(actual))
            for model, predicted in candidates.items():
                predicted_peak_index = int(np.argmax(predicted)) if np.max(predicted) > 0 else -1
                output.append({
                    "cell_id": cell, "channel": channel, "test_sweep": sweep,
                    "model": model, "previous_availability_proxy": availability,
                    "peak_timing_abs_error_ms": float(abs(predicted_peak_index-actual_peak_index)) if predicted_peak_index >= 0 else float("nan"),
                    **score_1ms(actual, predicted),
                })
            previous_row = row
    for filename, rows in (("piezo_dynamic_scores.csv", output), ("piezo_dynamic_fits.csv", fit_rows)):
        with (HERE / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for channel in ("mp1", "mp2"):
        for model in ("zero", "static_power", "dynamic_q1", "dynamic_previous_q"):
            subset = [r for r in output if r["channel"] == channel and r["model"] == model]
            by_cell_score = defaultdict(list)
            for row in subset:
                by_cell_score[row["cell_id"]].append(row)
            print(channel, model, "cells", len(by_cell_score), "sweeps", len(subset),
                  "median_cell_mean_mae", round(float(np.median([np.mean([x["mae_pA"] for x in values]) for values in by_cell_score.values()])), 2),
                  "median_cell_mean_peak_error", round(float(np.median([np.mean([x["peak_abs_error_pA"] for x in values]) for values in by_cell_score.values()])), 2))


if __name__ == "__main__":
    main()
