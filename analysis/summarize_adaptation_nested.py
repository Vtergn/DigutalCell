"""Summarize prospective nested-model scores with cells as the analysis units."""

from __future__ import annotations

import csv
from collections import defaultdict

import numpy as np

from dynamic_input_screen import HERE


METRICS = (
    "mae_pA", "postpeak_0_50ms_mae_pA", "peak_abs_error_pA",
    "peak_time_abs_error_ms", "onset_abs_error_ms",
    "false_positive_charge_fC", "oracle_peak_scaled_postpeak_shape_mae",
    "peak_amplitude_ratio_actual_predicted",
    "postpeak_0_50ms_signed_bias_pA",
)


def summarize(suffix: str) -> None:
    with (HERE / f"adaptation_nested_scores{suffix}.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with (HERE / f"adaptation_nested_fits{suffix}.csv").open(newline="", encoding="utf-8") as f:
        fits = list(csv.DictReader(f))
    print("\nCALIBRATION", "peak" if suffix else "least_squares")
    for channel in ("mp1", "mp2"):
        by_cell = defaultdict(lambda: defaultdict(list))
        for row in rows:
            if row["channel"] == channel:
                by_cell[row["cell_id"]][row["model"]].append(row)
        channel_fits = [f for f in fits if f["channel"] == channel]
        print("\n", channel, "cells", len(by_cell), "sweeps", sum(len(v["M1_threshold_force"]) for v in by_cell.values()),
              "held-out dates", len(channel_fits))
        print("selected tau,beta", [(f["tau_ms"], f["beta"]) for f in channel_fits])
        for metric in METRICS:
            values = defaultdict(dict)
            for cell, models in by_cell.items():
                for model, observations in models.items():
                    valid = [float(row[metric]) for row in observations if np.isfinite(float(row[metric]))]
                    values[model][cell] = float(np.mean(valid)) if valid else np.nan
            medians = {}
            for model in ("M0_zero", "M1_threshold_force", "M2_effective_adaptation"):
                valid_model = [v for v in values[model].values() if np.isfinite(v)]
                medians[model] = round(float(np.median(valid_model)), 2) if valid_model else np.nan
            deltas = np.array([values["M2_effective_adaptation"][cell]-values["M1_threshold_force"][cell]
                               for cell in by_cell])
            valid = deltas[np.isfinite(deltas)]
            print(metric, medians, "paired median", round(float(np.median(valid)), 2),
                  "improve/equal/worse", (int(np.sum(valid < -1e-8)), int(np.sum(abs(valid) <= 1e-8)),
                                         int(np.sum(valid > 1e-8))))


if __name__ == "__main__":
    summarize("")
    summarize("_peak_calibration")
