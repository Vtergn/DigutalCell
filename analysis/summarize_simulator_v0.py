"""Cell-paired summary of the crossed amplitude/shape simulator baselines."""

from __future__ import annotations

import csv
from collections import defaultdict

import numpy as np

from dynamic_input_screen import HERE


METRICS = ("mae_pA", "peak_abs_error_pA", "peak_time_abs_error_ms",
           "onset_abs_error_ms", "postpeak_0_50ms_mae_pA", "false_positive_charge_fC",
           "oracle_peak_scaled_postpeak_shape_mae")
MODELS = ("S0_zero", "S1_linear_force_shape", "S2_power_force_shape",
          "S3_linear_adapt_shape", "S4_power_adapt_shape")


def mean_valid(rows: list[dict], metric: str) -> float:
    values = [float(r[metric]) for r in rows if np.isfinite(float(r[metric]))]
    return float(np.mean(values)) if values else np.nan


def main() -> None:
    with (HERE / "simulator_v0_scores.csv").open(newline="", encoding="utf-8") as handle:
        scores = list(csv.DictReader(handle))
    for channel in ("mp1", "mp2"):
        by_cell = defaultdict(lambda: defaultdict(list))
        for row in scores:
            if row["channel"] == channel:
                by_cell[row["cell_id"]][row["model"]].append(row)
        print("\n", channel, "cells", len(by_cell), "sweeps",
              sum(len(v["S1_linear_force_shape"]) for v in by_cell.values()))
        for metric in METRICS:
            per_cell = {model: {cell: mean_valid(models[model], metric) for cell, models in by_cell.items()}
                        for model in MODELS}
            medians = {model: round(float(np.median([v for v in values.values() if np.isfinite(v)])), 2)
                       for model, values in per_cell.items()
                       if any(np.isfinite(v) for v in values.values())}
            print(metric, medians)
            for target, source in (("S2_power_force_shape", "S1_linear_force_shape"),
                                   ("S3_linear_adapt_shape", "S1_linear_force_shape"),
                                   ("S4_power_adapt_shape", "S2_power_force_shape"),
                                   ("S4_power_adapt_shape", "S1_linear_force_shape")):
                changes = np.array([per_cell[target][cell]-per_cell[source][cell] for cell in by_cell])
                changes = changes[np.isfinite(changes)]
                if len(changes):
                    print("  paired", target, "vs", source,
                          "median", round(float(np.median(changes)), 2),
                          "improve/equal/worse", (int(np.sum(changes < -1e-8)),
                                                   int(np.sum(abs(changes) <= 1e-8)),
                                                   int(np.sum(changes > 1e-8))))


if __name__ == "__main__":
    main()
