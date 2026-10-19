"""Whole-date-held-out nonlinear force amplitude baseline on the signed cohort."""

from __future__ import annotations

import csv
from collections import defaultdict

import numpy as np

from causal_cnn_benchmark import HERE, build_records, metrics


EXPONENTS = np.arange(.05, 4.001, .05)


def observed_peak(row: dict) -> float:
    start = max(row["contact"]-5, 0)
    return max(float(np.max(row["target"][start:])), 0.0)


def ratio(row: dict) -> float:
    return float(np.max(row["force_baseline"])/row["cal_peak"])


def choose_exponent(rows: list[dict]) -> float:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["cell_id"]].append(row)
    ranked = []
    for exponent in EXPONENTS:
        errors = []
        for sweeps in grouped.values():
            errors.append(np.mean([abs(np.log1p(row["cal_peak"]*ratio(row)**exponent)-
                                       np.log1p(observed_peak(row))) for row in sweeps]))
        ranked.append((float(np.mean(errors)), float(exponent)))
    return min(ranked)[1]


def main() -> None:
    records = build_records()
    scores, fits = [], []
    for date in sorted(set(row["date"] for row in records)):
        for channel in ("mp1", "mp2"):
            train = [row for row in records if row["date"] != date and row["channel"] == channel]
            test = [row for row in records if row["date"] == date and row["channel"] == channel]
            if not test:
                continue
            exponent = choose_exponent(train)
            fits.append({"heldout_date": date, "channel": channel, "train_cells": len(set(r["cell_id"] for r in train)),
                         "test_cells": len(set(r["cell_id"] for r in test)), "force_exponent": exponent})
            for row in test:
                r = ratio(row)
                waveform = row["force_baseline"]*r**(exponent-1) if r > 0 else np.zeros(350)
                scores.append({"cell_id": row["cell_id"], "channel": channel,
                               "heldout_date": date, "sweep": row["sweep"],
                               "model": "threshold_force_power", "force_exponent": exponent,
                               **metrics(row, waveform)})
    for name, rows in (("signed_power_baseline_scores.csv", scores),
                       ("signed_power_baseline_fits.csv", fits)):
        with (HERE / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print("saved", len(scores), "scores", fits)


if __name__ == "__main__":
    main()
