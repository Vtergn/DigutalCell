"""Cell-held-out metadata-conditioned zero-shot versus one-shot prediction."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from one_shot_work import read_metadata, first_calibration, load_approaches, gain_from_sweep, waveform, score


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "Data" / "2022_pnas" / "2022_pnas" / "bychannel"


def metadata() -> dict[tuple[str, int], dict]:
    with (ROOT / "bychannel_summary.csv").open(newline="", encoding="utf-8-sig") as handle:
        selected = list(csv.DictReader(handle))
    paths = {path.name: path for path in ROOT.rglob("*_preprocessed.feather")}
    result = {}
    for row in selected:
        path = paths[Path(row["path"]).name]
        summary = path.with_name(path.name.replace("_preprocessed.feather", "_summary.csv"))
        with summary.open(newline="", encoding="utf-8-sig") as handle:
            for sweep in csv.DictReader(handle):
                result[(sweep["uniqueID"], int(sweep["sweep"]))] = sweep
    return result


def descriptors(meta: dict, voltage: float) -> dict[str, float]:
    cm = float(meta["Cm"])
    leak = abs(float(meta["leak"]))
    if cm <= 0:
        raise ValueError("Nonpositive capacitance")
    return {
        "log_cm": float(np.log(cm)),
        "voltage_mV": voltage,
        # Baseline current per capacitance is an empirical recording proxy,
        # not an independently measured channel density.
        "log_leak_density": float(np.log1p(leak / cm)),
    }


def regress(train: list[dict], test: dict, target: str, features: tuple[str, ...]) -> float:
    if not features:
        return float(np.median([r[target] for r in train]))
    x = np.asarray([[r[key] for key in features] for r in train])
    query = np.asarray([test[key] for key in features])
    center, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-6)
    design = np.column_stack([np.ones(len(train)), (x-center)/scale])
    y = np.asarray([r[target] for r in train])
    penalty = np.diag([0.0] + [10.0] * len(features))
    coefficients = np.linalg.solve(design.T@design + penalty, design.T@y)
    return float(np.r_[1, (query-center)/scale] @ coefficients)


def main(holdout_date: bool = False) -> None:
    by_cell = read_metadata()
    traces = load_approaches(by_cell)
    meta = metadata()
    calibration = {cell: first_calibration(rows) for cell, rows in by_cell.items()}
    entries = {}
    for cell, row in calibration.items():
        if row is None:
            continue
        sweep = int(row["sweep"])
        trace = traces[(cell, sweep)]
        entries[cell] = {
            "cell_id": cell, "channel": row["channel"], "row": row,
            "trace": trace,
            **descriptors(meta[(cell, sweep)], float(np.median(trace["voltage"]))),
        }
    predictions = []
    parameters = []
    for cell, entry in entries.items():
        row = entry["row"]
        channel = entry["channel"]
        cal_sweep = int(row["sweep"])
        later = [x for x in by_cell[cell] if int(x["sweep"]) > cal_sweep]
        if not later:
            continue
        train = [x for other, x in entries.items() if other != cell and x["channel"] == channel
                 and (not holdout_date or x["row"]["date"] != row["date"])]
        if len(train) < 4:
            continue
        alpha = float(np.median([float(x["row"]["work_alpha"]) for x in train]))
        training = []
        for item in train:
            threshold = float(item["row"]["wthresh_fJ"])
            gain = gain_from_sweep(item["trace"]["work"], item["trace"]["current"], threshold, alpha)
            training.append({**item, "log_threshold": float(np.log(threshold)), "log_gain": float(np.log(gain))})
        w0 = float(row["wthresh_fJ"])
        one_gain = gain_from_sweep(entry["trace"]["work"], entry["trace"]["current"], w0, alpha)
        feature_sets = {
            "population": (),
            "cm_voltage": ("log_cm", "voltage_mV"),
            "cm_voltage_leak": ("log_cm", "voltage_mV", "log_leak_density"),
        }
        fitted = {}
        for model, features in feature_sets.items():
            fitted[model] = (
                float(np.exp(regress(training, entry, "log_threshold", features))),
                float(np.exp(regress(training, entry, "log_gain", features))),
            )
        fitted["one_shot"] = (w0, one_gain)
        for model, (threshold, gain) in fitted.items():
            parameters.append({
                "cell_id": cell, "channel": channel, "model": model,
                "observed_cal_threshold_fJ": w0, "estimated_threshold_fJ": threshold,
                "observed_cal_gain": one_gain, "estimated_gain": gain,
                "cm_pF": float(np.exp(entry["log_cm"])),
                "voltage_mV": entry["voltage_mV"],
                "baseline_current_per_cm": float(np.expm1(entry["log_leak_density"])),
            })
        for test_row in later:
            test_sweep = int(test_row["sweep"])
            test = traces[(cell, test_sweep)]
            if abs(float(np.median(test["voltage"])) - entry["voltage_mV"]) > 5:
                continue
            for model, (threshold, gain) in fitted.items():
                predicted = waveform(test["work"], gain, alpha, threshold)
                predictions.append({
                    "cell_id": cell, "channel": channel, "test_sweep": test_sweep,
                    "model": model, **score(test["current"], predicted),
                })
    suffix = "_date_heldout" if holdout_date else ""
    for filename, values in ((f"metadata_zero_shot_scores{suffix}.csv", predictions), (f"metadata_zero_shot_parameters{suffix}.csv", parameters)):
        with (HERE / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    for channel in sorted({x["channel"] for x in predictions}):
        for model in ("population", "cm_voltage", "cm_voltage_leak", "one_shot"):
            subset = [x for x in predictions if x["channel"] == channel and x["model"] == model]
            grouped = defaultdict(list)
            for item in subset:
                grouped[item["cell_id"]].append(item["mae_pA"])
            print("date_heldout" if holdout_date else "cell_heldout", channel, model, "cells", len(grouped), "median_cell_mean_mae_pA",
                  round(float(np.median([np.mean(v) for v in grouped.values()])), 2))


if __name__ == "__main__":
    main()
    main(holdout_date=True)
