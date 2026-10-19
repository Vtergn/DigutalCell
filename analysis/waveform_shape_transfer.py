"""Whole-date-held-out PCA reconstruction of observed-peak-normalized shapes.

This is an oracle-normalized shape diagnostic, not a current forecast.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict

import numpy as np

from dynamic_input_screen import HERE, load_traces
from one_shot_work import first_calibration, read_metadata


LEFT_MS, RIGHT_MS = 60, 140
COMPONENTS = (1, 2, 3)


def fit_pca(matrix: np.ndarray, cells: list[str]) -> tuple[np.ndarray, np.ndarray]:
    counts = Counter(cells)
    weights = np.asarray([1.0/counts[cell] for cell in cells])
    weights /= np.sum(weights)
    center = np.average(matrix, axis=0, weights=weights)
    _, _, vectors = np.linalg.svd((matrix-center)*np.sqrt(weights[:, None]), full_matrices=False)
    return center, vectors


def main() -> None:
    with (HERE / "waveform_geometry_pilot_scores_snr5.csv").open(newline="", encoding="utf-8") as handle:
        selected = {(r["cell_id"], int(r["sweep"])) for r in csv.DictReader(handle)
                    if r["alignment"] == "afm_force_peak" and r["representation"] == "observed_peak_normalized"}
    by_cell = read_metadata()
    calibration = {cell: first_calibration(rows) for cell, rows in by_cell.items()}
    traces, _ = load_traces(by_cell, calibration)
    observations = []
    for cell, sweep in sorted(selected):
        trace = traces[(cell, sweep)]
        force_peak = int(np.argmax(trace["F"][:150]))
        contact = max(int(trace["contact_ms"])-5, 0)
        actual_peak = float(np.max(trace["current"][contact:]))
        shape = trace["current"][force_peak-LEFT_MS:force_peak+RIGHT_MS+1]/actual_peak
        if len(shape) != LEFT_MS+RIGHT_MS+1:
            raise ValueError("Unexpected shape window")
        observations.append({"cell_id": cell, "channel": calibration[cell]["channel"],
                             "date": calibration[cell]["date"], "sweep": sweep, "shape": shape})
    scores = []
    for channel in ("mp1", "mp2"):
        group = [r for r in observations if r["channel"] == channel]
        for date in sorted({r["date"] for r in group}):
            train = [r for r in group if r["date"] != date]
            test = [r for r in group if r["date"] == date]
            center, vectors = fit_pca(np.stack([r["shape"] for r in train]), [r["cell_id"] for r in train])
            for row in test:
                target = row["shape"]
                centered = target-center
                baseline = float(np.mean(centered**2))
                for k in COMPONENTS:
                    estimate = center+(centered@vectors[:k].T)@vectors[:k]
                    mse = float(np.mean((target-estimate)**2))
                    scores.append({"cell_id": row["cell_id"], "channel": channel, "heldout_date": date,
                                   "sweep": row["sweep"], "train_cells": len({r["cell_id"] for r in train}),
                                   "k": k, "mse_mean_shape": baseline, "mse_pca": mse,
                                   "r2_vs_mean_shape": 1.0-mse/baseline if baseline > 0 else np.nan})
    with (HERE / "waveform_shape_transfer_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scores[0]))
        writer.writeheader()
        writer.writerows(scores)
    for channel in ("mp1", "mp2"):
        for k in COMPONENTS:
            grouped = defaultdict(list)
            for row in scores:
                if row["channel"] == channel and row["k"] == k:
                    grouped[row["cell_id"]].append(row)
            ratios = [np.mean([r["mse_pca"] for r in values])/
                      np.mean([r["mse_mean_shape"] for r in values]) for values in grouped.values()]
            print(channel, "k", k, "cells", len(grouped),
                  "cell-median R2 vs train mean", round(float(np.median(1-np.array(ratios))), 3),
                  "improved cells", sum(x < 1 for x in ratios))


if __name__ == "__main__":
    main()
