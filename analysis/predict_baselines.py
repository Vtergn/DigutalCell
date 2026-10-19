"""Cell-held-out, channel-specific baselines for the Duke approach traces.

Inputs are read-only. Outputs are small CSV files next to this script.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "Data" / "2022_pnas" / "2022_pnas" / "bychannel"


def build_cell(meta: dict, force: np.ndarray, work: np.ndarray, current: np.ndarray) -> dict:
    force = np.maximum(np.asarray(force, dtype=float), 0)
    work = np.maximum(np.asarray(work, dtype=float), 0)
    current = np.asarray(current, dtype=float)
    baseline = float(np.median(current[:100]))
    onset = int(np.flatnonzero(force >= 0.05 * np.max(force))[0])
    alpha = float(np.exp(-0.04 / 5.0))  # 25 kHz samples; fixed 5 ms adaptation timescale
    smoothed = np.empty_like(force)
    smoothed[0] = force[0]
    for index in range(1, len(force)):
        smoothed[index] = alpha * smoothed[index - 1] + (1 - alpha) * force[index]
    force_rate = np.maximum(force - smoothed, 0)
    lag10 = np.r_[np.zeros(250), force_rate[:-250]]
    lag20 = np.r_[np.zeros(500), force_rate[:-500]]
    return {
        "id": meta["uniqueID"], "channel": meta["construct"], "date": meta["date"],
        "vstep_mV": float(meta["vstep"]), "responds": float(meta["responds"]),
        "force": force, "work": work, "force_rate": force_rate,
        "force_rate_lag10": lag10, "force_rate_lag20": lag20,
        "y": current - baseline,
        "baseline_pA": baseline, "onset": onset,
    }


def load() -> list[dict]:
    summary = {}
    with (DATA / "bychannel_summary.csv").open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            summary[(row["uniqueID"], row["sweep"])] = row
    traces = defaultdict(lambda: defaultdict(list))
    with (DATA / "bychannel_approaches.csv").open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            key = (row["uniqueID"], row["sweep"])
            for name, source in (("force", "force"), ("work", "work"), ("current", "absi_blsub_mech")):
                traces[key][name].append(float(row[source]))
    cells = []
    for key, values in traces.items():
        meta = summary[key]
        cells.append(build_cell(meta, values["force"], values["work"], values["current"]))
    return cells


def fit_slope(train: list[dict], feature: str, scale: float) -> float:
    numerator = denominator = 0.0
    for cell in train:
        x = cell[feature] / scale
        numerator += float(x @ cell["y"])
        denominator += float(x @ x)
    return max(0.0, numerator / denominator) if denominator else 0.0


def predict(train: list[dict], test: dict, model: str) -> np.ndarray:
    if model == "zero":
        return np.zeros_like(test["y"])
    if model == "force":
        return fit_slope(train, "force", 100) * test["force"] / 100
    if model == "work":
        return fit_slope(train, "work", 100) * test["work"] / 100
    if model == "force_rate_5ms":
        return fit_slope(train, "force_rate", 100) * test["force_rate"] / 100
    if model == "force_rate_lag10":
        return fit_slope(train, "force_rate_lag10", 100) * test["force_rate_lag10"] / 100
    if model == "force_rate_lag20":
        return fit_slope(train, "force_rate_lag20", 100) * test["force_rate_lag20"] / 100
    if model == "saturating_force":
        # K is estimated from training cells only. A single global gain is fitted.
        half_force = float(np.median([np.max(c["force"]) for c in train])) / 2
        train_mapped = [{**c, "mapped": c["force"] / (c["force"] + half_force)} for c in train]
        gain = fit_slope(train_mapped, "mapped", 1)
        return gain * test["force"] / (test["force"] + half_force)
    raise ValueError(model)


def evaluate(cell: dict, predicted: np.ndarray) -> dict:
    observed = cell["y"]
    onset = cell["onset"]
    actual = observed[onset:]
    estimate = predicted[onset:]
    scale = float(np.quantile(np.abs(actual), 0.95))
    mae = float(np.mean(np.abs(actual - estimate)))
    high = actual >= np.quantile(actual, 0.80)
    return {
        "mae_pA": mae,
        "event_mae_pA": float(np.mean(np.abs(actual[high] - estimate[high]))),
        "peak_q95_error_pA": abs(float(np.quantile(actual, 0.95) - np.quantile(estimate, 0.95))),
        "normalized_mae": mae / scale if scale > 1e-9 else float("nan"),
        "signal_q95_pA": scale,
        "peak_force_nN": float(np.max(cell["force"])),
        "precontact_mae_pA": float(np.mean(np.abs(observed[:onset] - predicted[:onset]))),
    }


def run(cells: list[dict], prefix: str) -> None:
    results = []
    for test in cells:
        voltage_bin = round(test["vstep_mV"] / 80) * 80
        for scope in ("channel", "channel_voltage", "channel_date"):
            train = [c for c in cells if c["channel"] == test["channel"] and c["id"] != test["id"]]
            if scope == "channel_voltage":
                train = [c for c in train if round(c["vstep_mV"] / 80) * 80 == voltage_bin]
                if len(train) < 4:
                    continue
            elif scope == "channel_date":
                train = [c for c in train if c["date"] != test["date"]]
            for model in ("zero", "force", "work", "force_rate_5ms", "force_rate_lag10", "force_rate_lag20", "saturating_force"):
                score = evaluate(test, predict(train, test, model))
                results.append({
                    "cell_id": test["id"], "channel": test["channel"], "date": test["date"],
                    "vstep_mV": test["vstep_mV"], "vstep_bin_mV": voltage_bin,
                    "v_span_mV": test.get("v_span_mV", ""),
                    "responds": test["responds"], "scope": scope,
                    "model": model, "onset_index": test["onset"],
                    "baseline_pA": test["baseline_pA"], **score,
                })

    target = HERE / f"{prefix}_cell_scores.csv"
    with target.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    aggregate = []
    for channel in sorted({r["channel"] for r in results}):
        for scope in ("channel", "channel_voltage", "channel_date"):
            for model in ("zero", "force", "work", "force_rate_5ms", "force_rate_lag10", "force_rate_lag20", "saturating_force"):
                subset = [r for r in results if r["channel"] == channel and r["scope"] == scope and r["model"] == model]
                if not subset:
                    continue
                aggregate.append({
                    "channel": channel, "scope": scope, "model": model, "cells": len(subset),
                    "median_mae_pA": round(float(np.median([r["mae_pA"] for r in subset])), 3),
                    "median_event_mae_pA": round(float(np.median([r["event_mae_pA"] for r in subset])), 3),
                    "median_peak_q95_error_pA": round(float(np.median([r["peak_q95_error_pA"] for r in subset])), 3),
                    "median_normalized_mae": round(float(np.median([r["normalized_mae"] for r in subset])), 3),
                    "mean_normalized_mae": round(float(np.mean([r["normalized_mae"] for r in subset])), 3),
                })
    target = HERE / f"{prefix}_summary.csv"
    with target.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    for row in aggregate:
        print(row)


if __name__ == "__main__":
    run(load(), "baseline")
