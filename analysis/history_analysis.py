"""Assess sweep history after accounting for current mechanical work.

The 5 s interval is fixed, so this script tests predictive history features. It
does not estimate a recovery time constant or attribute a causal mechanism.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def load() -> dict[str, list[dict]]:
    with (HERE / "all_sweep_power.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["cell_id"]].append(row)
    return {cell: sorted(values, key=lambda x: int(x["sweep"])) for cell, values in grouped.items()}


def rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])


def feature_rows(grouped: dict[str, list[dict]]) -> list[dict]:
    output = []
    for cell, rows in grouped.items():
        if len(rows) < 2:
            continue
        work = np.asarray([float(row["peak_work_fJ"]) for row in rows])
        current = np.asarray([float(row["current_peak_minus_offset_pA"]) for row in rows])
        charge = np.asarray([float(row["rectified_charge_350ms_fC"]) for row in rows])
        log_work = np.log(work)
        log_current = np.log1p(current)
        for index in range(1, len(rows)):
            output.append({
                "cell_id": cell, "channel": rows[0]["channel"], "date": rows[0]["date"],
                "sweep": int(rows[index]["sweep"]), "ordinal": index + 1,
                "current_work_fJ": work[index], "current_peak_pA": current[index],
                "previous_work_fJ": work[index - 1], "previous_peak_pA": current[index - 1],
                "previous_charge_fC": charge[index - 1],
                "cumulative_prior_work_fJ": float(np.sum(work[:index])),
                "cumulative_prior_charge_fC": float(np.sum(charge[:index])),
                "delta_log_current": log_current[index] - log_current[0],
                "delta_log_work": log_work[index] - log_work[0],
                "prior_delta_log_work": log_work[index - 1] - log_work[0],
                "prior_delta_log_current": log_current[index - 1] - log_current[0],
                "prior_delta_log_charge": float(np.log1p(charge[index - 1]) - np.log1p(charge[0])),
                "log_cumulative_prior_work": float(np.log(np.sum(work[:index]))),
                "log_cumulative_prior_charge": float(np.log1p(np.sum(charge[:index]))),
                "sweep_index": index,
                "first_log_current": log_current[0],
            })
    return output


FEATURES = {
    "work": ("delta_log_work",),
    "work_index": ("delta_log_work", "sweep_index"),
    "work_previous_work": ("delta_log_work", "prior_delta_log_work"),
    "work_previous_current": ("delta_log_work", "prior_delta_log_current"),
    "work_previous_charge": ("delta_log_work", "prior_delta_log_charge"),
    "work_cumulative_work": ("delta_log_work", "log_cumulative_prior_work"),
    "work_cumulative_charge": ("delta_log_work", "log_cumulative_prior_charge"),
}


def fit_predict(train: list[dict], test: dict, features: tuple[str, ...]) -> float:
    x_train = np.asarray([[row[name] for name in features] for row in train])
    x_test = np.asarray([test[name] for name in features])
    mean = x_train.mean(axis=0)
    scale = np.maximum(x_train.std(axis=0), 1e-6)
    transformed = (x_train - mean) / scale
    design = np.column_stack([np.ones(len(train)), transformed])
    target = np.asarray([row["delta_log_current"] for row in train])
    penalty = np.diag([0] + [1.0] * len(features))
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    return float(np.r_[1.0, (x_test - mean) / scale] @ coefficients)


def main() -> None:
    grouped = load()
    rows = feature_rows(grouped)
    out = []
    for test in rows:
        train = [row for row in rows if row["channel"] == test["channel"] and row["cell_id"] != test["cell_id"]]
        for model, features in FEATURES.items():
            pred_delta = fit_predict(train, test, features)
            actual_delta = test["delta_log_current"]
            out.append({
                "cell_id": test["cell_id"], "channel": test["channel"], "date": test["date"],
                "sweep": test["sweep"], "ordinal": test["ordinal"], "model": model,
                "actual_delta_log_current": actual_delta,
                "predicted_delta_log_current": pred_delta,
                "absolute_log_error": abs(actual_delta - pred_delta),
                "absolute_current_error_pA": abs(
                    float(np.expm1(test["first_log_current"] + actual_delta))
                    - float(np.expm1(test["first_log_current"] + pred_delta))
                ),
            })
    with (HERE / "history_prediction.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)

    matched = []
    for cell, values in grouped.items():
        for previous, current in zip(values, values[1:]):
            w0, w1 = float(previous["peak_work_fJ"]), float(current["peak_work_fJ"])
            i0, i1 = float(previous["current_peak_minus_offset_pA"]), float(current["current_peak_minus_offset_pA"])
            if abs(np.log(w1 / w0)) <= np.log(1.1):
                matched.append({
                    "cell_id": cell, "channel": current["channel"],
                    "work_ratio": w1 / w0, "current_ratio": i1 / i0,
                    "delta_log_current": float(np.log(i1 / i0)),
                })
    with (HERE / "near_equal_work_pairs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(matched[0]))
        writer.writeheader()
        writer.writerows(matched)
    for channel in sorted({r["channel"] for r in rows}):
        channel_rows = [r for r in rows if r["channel"] == channel]
        print({
            "channel": channel, "cells": len({r["cell_id"] for r in channel_rows}),
            "later_sweeps": len(channel_rows),
            "median_within_cell_rank_corr_sweep_work": round(float(np.median([
                rank_corr(np.arange(len(v)), np.asarray([float(x["peak_work_fJ"]) for x in v]))
                for v in grouped.values() if len(v) >= 3 and v[0]["channel"] == channel
            ])), 3),
        })
        for model in FEATURES:
            model_rows = [r for r in out if r["channel"] == channel and r["model"] == model]
            cell_errors = defaultdict(list)
            for row in model_rows:
                cell_errors[row["cell_id"]].append(row["absolute_log_error"])
            print("  ", model, "median_cell_mean_absolute_log_error", round(float(np.median([
                np.mean(v) for v in cell_errors.values()
            ])), 3))
        pairs = [r for r in matched if r["channel"] == channel]
        print("   near_equal_work_pairs", len(pairs), "cells", len({r["cell_id"] for r in pairs}),
              "median_current_ratio", round(float(np.median([r["current_ratio"] for r in pairs])), 3),
              "declines", sum(r["current_ratio"] < 1 for r in pairs))


if __name__ == "__main__":
    main()
