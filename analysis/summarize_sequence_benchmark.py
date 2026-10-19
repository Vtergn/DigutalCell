"""Cell-paired summary across fixed CNN seeds and the two-state ODE."""

from __future__ import annotations

import csv
from collections import defaultdict

import numpy as np

from dynamic_input_screen import HERE


SEEDS = (20260926, 20260927, 20260928)
METRICS = ("mae_pA", "peak_abs_error_pA", "peak_time_abs_error_ms",
           "postpeak_0_50ms_mae_pA", "false_positive_charge_fC")


def read(name: str) -> list[dict]:
    with (HERE / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    base = read(f"causal_cnn_benchmark_scores_seed{SEEDS[0]}.csv")
    hybrid = read("hybrid_state_benchmark_scores.csv")
    power = read("signed_power_baseline_scores.csv")
    summary = []
    per_cell = []
    for channel in ("mp1", "mp2"):
        cells = sorted({row["cell_id"] for row in base if row["channel"] == channel})
        for seed in SEEDS:
            cnn = read(f"causal_cnn_benchmark_scores_seed{seed}.csv")
            gated = read(f"causal_cnn_benchmark_scores_gated_seed{seed}.csv")
            scores = [row for row in base if row["model"] in {"zero", "threshold_force"}]+[
                row for row in cnn if row["model"] == "causal_cnn"]+[
                {**row, "model": "causal_cnn_gated"} for row in gated
                if row["model"] == "causal_cnn"]+hybrid+power
            for model in ("zero", "threshold_force", "threshold_force_power",
                          "causal_cnn", "causal_cnn_gated",
                          "two_state_neural_ode"):
                for metric in METRICS:
                    values = []
                    paired = []
                    paired_power = []
                    for cell in cells:
                        observed = [float(row[metric]) for row in scores
                                    if row["cell_id"] == cell and row["model"] == model]
                        reference = [float(row[metric]) for row in scores
                                     if row["cell_id"] == cell and row["model"] == "threshold_force"]
                        power_reference = [float(row[metric]) for row in scores
                                           if row["cell_id"] == cell and row["model"] == "threshold_force_power"]
                        if not observed or not reference or not power_reference:
                            raise ValueError((cell, model, metric))
                        value = float(np.mean(observed))
                        difference = value-float(np.mean(reference))
                        values.append(value)
                        paired.append(difference)
                        paired_power.append(value-float(np.mean(power_reference)))
                        per_cell.append({"channel": channel, "seed": seed, "cell_id": cell,
                                         "model": model, "metric": metric,
                                         "cell_mean": value, "minus_force_baseline": difference,
                                         "minus_power_baseline": value-float(np.mean(power_reference))})
                    summary.append({"channel": channel, "seed": seed, "model": model,
                                    "metric": metric, "cells": len(cells),
                                    "median_cell_mean": float(np.median(values)),
                                    "median_paired_minus_force": float(np.median(paired)),
                                    "median_paired_minus_power": float(np.median(paired_power)),
                                    "cells_improved": int(np.sum(np.array(paired) < 0)),
                                    "cells_worsened": int(np.sum(np.array(paired) > 0)),
                                    "cells_improved_over_power": int(np.sum(np.array(paired_power) < 0))})
    for filename, rows in (("sequence_benchmark_summary.csv", summary),
                           ("sequence_benchmark_per_cell.csv", per_cell)):
        with (HERE / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for row in summary:
        if row["metric"] == "mae_pA" and row["model"] in {"causal_cnn", "causal_cnn_gated",
                                                         "two_state_neural_ode"}:
            print(row)


if __name__ == "__main__":
    main()
