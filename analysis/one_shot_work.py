"""One-sweep calibration of approach-phase W-I curves, then future-sweep tests."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.feather as feather


ROOT = Path(__file__).resolve().parents[1] / "Data" / "2022_pnas" / "2022_pnas" / "bychannel"
HERE = Path(__file__).resolve().parent


def read_metadata() -> dict[str, list[dict]]:
    with (HERE / "all_sweep_power.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_cell = defaultdict(list)
    for row in rows:
        by_cell[row["cell_id"]].append(row)
    return {cell: sorted(rows, key=lambda x: int(x["sweep"])) for cell, rows in by_cell.items()}


def eligible(row: dict) -> bool:
    return (
        math.isfinite(float(row["work_alpha"]))
        and float(row["work_log_r2"]) >= 0.5
        and float(row["threshold_time_ms"]) < float(row["compression_end_ms"])
    )


def first_calibration(rows: list[dict]) -> dict | None:
    return next((row for row in rows if eligible(row)), None)


def load_approaches(by_cell: dict[str, list[dict]]) -> dict[tuple[str, int], dict[str, np.ndarray]]:
    with (ROOT / "bychannel_summary.csv").open(newline="", encoding="utf-8-sig") as f:
        selected = list(csv.DictReader(f))
    paths = {p.name: p for p in ROOT.rglob("*_preprocessed.feather")}
    approaches = {}
    for chosen in selected:
        path = paths[Path(chosen["path"]).name]
        filename = path.name.replace("_preprocessed.feather", "_summary.csv")
        with path.with_name(filename).open(newline="", encoding="utf-8-sig") as f:
            metadata = {int(x["sweep"]): x for x in csv.DictReader(f)}
        table = feather.read_table(path, columns=["sweep", "ti", "work", "i_blsub", "v"])
        data = {col: table[col].to_numpy() for col in table.column_names}
        for row in by_cell[chosen["uniqueID"]]:
            sweep = int(row["sweep"])
            start = float(row["compression_end_ms"]) - 150
            mask = (
                (data["sweep"] == sweep)
                & (data["ti"] >= start - 1e-6)
                & (data["ti"] <= start + 150 + 1e-6)
            )
            if np.count_nonzero(mask) != 3751:
                raise ValueError(f"Unexpected approach length: {chosen['uniqueID']} sweep {sweep}")
            abs_current = np.abs(data["i_blsub"][mask])
            # Per-sweep summary provides the author's offset. It is applied only to the target.
            current = abs_current - float(metadata[sweep]["offset"])
            approaches[(chosen["uniqueID"], sweep)] = {
                "work": np.maximum(data["work"][mask], 0),
                "current": current,
                "voltage": data["v"][mask],
            }
    return approaches


def gain_from_sweep(work: np.ndarray, current: np.ndarray, threshold: float, alpha: float) -> float:
    valid = (work > max(threshold, 0)) & (current > 0)
    if np.count_nonzero(valid) < 50:
        raise ValueError("Calibration sweep has too few positive points")
    return float(np.exp(np.median(np.log(current[valid]) - alpha * np.log(work[valid]))))


def waveform(work: np.ndarray, gain: float, alpha: float, threshold: float) -> np.ndarray:
    return gain * np.where(work >= threshold, np.maximum(work, 0) ** alpha, 0.0)


def graded_basis(work: np.ndarray, alpha: float, threshold: float, half_work: float) -> np.ndarray:
    positive = np.maximum(work, 0)
    powered = positive ** alpha
    return np.where(work >= threshold, powered / (half_work ** alpha + powered), 0.0)


def gain_from_basis(basis: np.ndarray, current: np.ndarray) -> float:
    valid = (basis > 0) & (current > 0)
    if np.count_nonzero(valid) < 50:
        raise ValueError("Calibration sweep has too few positive points")
    return float(np.exp(np.median(np.log(current[valid]) - np.log(basis[valid]))))


def score(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    delta = np.abs(actual - predicted)
    event = actual > max(10.0, 5 * float(np.std(actual[:500])))
    return {
        "mae_pA": float(np.mean(delta)),
        "event_mae_pA": float(np.mean(delta[event])) if np.any(event) else float("nan"),
        "charge_abs_error_fC": float(abs(np.sum(np.maximum(actual, 0)) - np.sum(predicted)) * 0.04),
        "peak_abs_error_pA": float(abs(np.max(actual) - np.max(predicted))),
        "false_positive_charge_fC": float(np.sum(predicted[~event]) * 0.04),
    }


def main() -> None:
    by_cell = read_metadata()
    traces = load_approaches(by_cell)
    calibrations = {cell: first_calibration(rows) for cell, rows in by_cell.items()}
    records = []
    for cell, rows in by_cell.items():
        calibration = calibrations[cell]
        if calibration is None:
            continue
        channel = calibration["channel"]
        cal_index = int(calibration["sweep"])
        later = [row for row in rows if int(row["sweep"]) > cal_index]
        if not later:
            continue
        train = [x for other, x in calibrations.items() if other != cell and x is not None and x["channel"] == channel]
        if len(train) < 4:
            continue
        shared_alpha = float(np.median([float(x["work_alpha"]) for x in train]))
        cell_alpha = float(calibration["work_alpha"])
        train_alphas = np.asarray([float(x["work_alpha"]) for x in train])
        squared_within_changes = [
            (float(candidate["work_alpha"]) - float(first["work_alpha"])) ** 2
            for other, first in calibrations.items()
            if other != cell and first is not None and first["channel"] == channel
            for candidate in by_cell[other]
            if int(candidate["sweep"]) > int(first["sweep"]) and eligible(candidate)
        ]
        within_variance = float(np.median(squared_within_changes) / 2) if squared_within_changes else 0.0
        between_variance = max(float(np.var(train_alphas, ddof=1)) - within_variance, 0.0)
        shrink_weight = between_variance / (between_variance + within_variance) if between_variance + within_variance else 0.0
        partial_alpha = shared_alpha + shrink_weight * (cell_alpha - shared_alpha)
        calibration_trace = traces[(cell, cal_index)]
        w0 = float(calibration["wthresh_fJ"])
        cell_gain_shared_alpha = gain_from_sweep(calibration_trace["work"], calibration_trace["current"], w0, shared_alpha)
        cell_gain_cell_alpha = gain_from_sweep(calibration_trace["work"], calibration_trace["current"], w0, cell_alpha)
        cell_gain_partial_alpha = gain_from_sweep(calibration_trace["work"], calibration_trace["current"], w0, partial_alpha)
        train_gains = []
        train_thresholds = []
        for candidate in train:
            candidate_trace = traces[(candidate["cell_id"], int(candidate["sweep"]))]
            threshold = float(candidate["wthresh_fJ"])
            train_gains.append(gain_from_sweep(candidate_trace["work"], candidate_trace["current"], threshold, shared_alpha))
            train_thresholds.append(threshold)
        population_gain = float(np.median(train_gains))
        population_threshold = float(np.median(train_thresholds))
        # A held-out-cell shared saturation scale for K2P, estimated only from
        # other cells' calibration sweeps. This remains an empirical work scale.
        graded_half_work = float(np.median([
            np.max(traces[(candidate["cell_id"], int(candidate["sweep"]))]["work"])
            for candidate in train
        ]))
        graded_gain = (
            gain_from_basis(graded_basis(calibration_trace["work"], shared_alpha, w0, graded_half_work), calibration_trace["current"])
            if channel in {"mtraak", "mtrek1"} else float("nan")
        )
        for row in later:
            test = traces[(cell, int(row["sweep"]))]
            if abs(float(np.median(test["voltage"])) - float(np.median(calibration_trace["voltage"]))) > 5:
                continue
            models = {
                "zero": np.zeros_like(test["current"]),
                "population": waveform(test["work"], population_gain, shared_alpha, population_threshold),
                "one_shot_gain": waveform(test["work"], cell_gain_shared_alpha, shared_alpha, w0),
                "one_shot_gain_alpha": waveform(test["work"], cell_gain_cell_alpha, cell_alpha, w0),
                "one_shot_partial_alpha": waveform(test["work"], cell_gain_partial_alpha, partial_alpha, w0),
            }
            if channel in {"mtraak", "mtrek1"}:
                models["one_shot_graded"] = graded_gain * graded_basis(test["work"], shared_alpha, w0, graded_half_work)
            for name, predicted in models.items():
                records.append({
                    "cell_id": cell, "channel": channel, "calibration_sweep": cal_index,
                    "test_sweep": int(row["sweep"]), "model": name,
                    "shared_alpha": shared_alpha, "calibration_alpha": cell_alpha,
                    "partial_alpha": partial_alpha, "shrink_weight": shrink_weight,
                    "calibration_threshold_fJ": w0, **score(test["current"], predicted),
                })
    if not records:
        raise RuntimeError("No eligible future sweeps")
    target = HERE / "one_shot_scores.csv"
    with target.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["cell_id"], record["channel"], record["model"])].append(record)
    cell_rows = []
    for (cell, channel, model), values in grouped.items():
        cell_rows.append({
            "cell_id": cell, "channel": channel, "model": model,
            "future_sweeps": len(values),
            **{
                f"mean_{metric}": float(np.nanmean([v[metric] for v in values]))
                for metric in ("mae_pA", "event_mae_pA", "peak_abs_error_pA", "charge_abs_error_fC", "false_positive_charge_fC")
            },
        })
    with (HERE / "one_shot_cell_scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(cell_rows[0]))
        writer.writeheader()
        writer.writerows(cell_rows)
    by_cell = defaultdict(dict)
    for row in cell_rows:
        by_cell[(row["channel"], row["cell_id"])][row["model"]] = row
    rng = np.random.default_rng(20260926)
    summary = []
    for channel in sorted({r["channel"] for r in records}):
        cells = [models for (c, _), models in by_cell.items() if c == channel]
        differences = np.asarray([
            models["one_shot_gain"]["mean_mae_pA"] - models["population"]["mean_mae_pA"]
            for models in cells
        ])
        boot = np.median(rng.choice(differences, size=(5000, len(differences)), replace=True), axis=1)
        summary.append({
            "channel": channel, "cells": len(cells),
            "future_sweeps": sum(models["zero"]["future_sweeps"] for models in cells),
            "median_cell_mean_mae_zero_pA": float(np.median([x["zero"]["mean_mae_pA"] for x in cells])),
            "median_cell_mean_mae_population_pA": float(np.median([x["population"]["mean_mae_pA"] for x in cells])),
            "median_cell_mean_mae_one_shot_pA": float(np.median([x["one_shot_gain"]["mean_mae_pA"] for x in cells])),
            "cells_one_shot_better_population": int(np.sum(differences < 0)),
            "median_paired_one_shot_minus_population_pA": float(np.median(differences)),
            "bootstrap_95_low_pA": float(np.quantile(boot, .025)),
            "bootstrap_95_high_pA": float(np.quantile(boot, .975)),
        })
        model_names = ("zero", "population", "one_shot_gain", "one_shot_gain_alpha", "one_shot_partial_alpha")
        if channel in {"mtraak", "mtrek1"}:
            model_names += ("one_shot_graded",)
        for model in model_names:
            subset = [r for r in records if r["channel"] == channel and r["model"] == model]
            print({
                "channel": channel, "model": model,
                "cells": len({r["cell_id"] for r in subset}), "future_sweeps": len(subset),
                "median_mae_pA": round(float(np.median([r["mae_pA"] for r in subset])), 2),
                "median_peak_abs_error_pA": round(float(np.median([r["peak_abs_error_pA"] for r in subset])), 2),
            })
    with (HERE / "one_shot_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
