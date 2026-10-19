"""Cell-paired summaries of whole-date-held-out peak-current predictions."""

from __future__ import annotations

import csv
from collections import defaultdict

import numpy as np

from dynamic_input_screen import HERE


def main() -> None:
    with (HERE / "amplitude_scaling_scores.csv").open(newline="", encoding="utf-8") as handle:
        scores = list(csv.DictReader(handle))
    with (HERE / "amplitude_scaling_fits.csv").open(newline="", encoding="utf-8") as handle:
        fits = list(csv.DictReader(handle))
    for channel in ("mp1", "mp2"):
        rows = [r for r in scores if r["channel"] == channel]
        reference = [r for r in rows if r["model"] == "A0_cal_peak"]
        print("\n", channel, "cells", len({r["cell_id"] for r in reference}), "sweeps", len(reference),
              "events", sum(int(r["actual_has_event"]) for r in reference))
        print("force ratio median/range", np.median([float(r["force_ratio"]) for r in reference]),
              (min(float(r["force_ratio"]) for r in reference), max(float(r["force_ratio"]) for r in reference)))
        print("work ratio median/range", np.median([float(r["work_ratio"]) for r in reference]),
              (min(float(r["work_ratio"]) for r in reference), max(float(r["work_ratio"]) for r in reference)))
        selected = [f for f in fits if f["channel"] == channel]
        print("heldout exponents", [(f["heldout_date"], f["gamma_force"], f["alpha_work"]) for f in selected])
        cell_values = defaultdict(lambda: defaultdict(dict))
        for row in rows:
            cell_values[row["cell_id"]][row["model"]][int(row["test_sweep"])] = row
        for model in ("A0_cal_peak", "A1_force_linear", "A2_force_power", "A3_work_power"):
            print("MODEL", model)
            for metric in ("abs_error_pA", "log1p_error", "log_ratio_error_responders", "actual_over_predicted"):
                cell_means = []
                for models in cell_values.values():
                    valid = [float(r[metric]) for r in models[model].values() if np.isfinite(float(r[metric]))]
                    if valid:
                        cell_means.append(float(np.mean(valid)))
                print(metric, "cell-median", round(float(np.median(cell_means)), 3), "n", len(cell_means))
        for challenger in ("A1_force_linear", "A2_force_power", "A3_work_power"):
            for benchmark in ("A0_cal_peak", "A1_force_linear"):
                if challenger == benchmark:
                    continue
                for metric in ("abs_error_pA", "log1p_error"):
                    deltas = []
                    for models in cell_values.values():
                        change = [float(r[metric])-float(models[benchmark][sweep][metric])
                                  for sweep, r in models[challenger].items()]
                        deltas.append(float(np.mean(change)))
                    print("PAIRED", challenger, "vs", benchmark, metric,
                          "median delta", round(float(np.median(deltas)), 3),
                          "improve", sum(d < -1e-8 for d in deltas), "worse", sum(d > 1e-8 for d in deltas),
                          "equal", sum(abs(d) <= 1e-8 for d in deltas))
        for benchmark in ("A1_force_linear", "A3_work_power"):
            for metric in ("abs_error_pA", "log1p_error"):
                deltas = np.array([
                    np.mean([float(r[metric])-float(models[benchmark][sweep][metric])
                             for sweep, r in models["A2_force_power"].items()])
                    for models in cell_values.values()
                ])
                generator = np.random.default_rng(20260926)
                resampled = np.median(deltas[generator.integers(0, len(deltas), (20000, len(deltas)))], axis=1)
                print("A2 VS", benchmark, metric, "paired median", round(float(np.median(deltas)), 3),
                      "descriptive cell bootstrap 95%", np.round(np.quantile(resampled, [0.025, 0.975]), 3),
                      "improve", int(np.sum(deltas < 0)), "worse", int(np.sum(deltas > 0)))
        for date in sorted({r["date"] for r in reference}):
            date_cells = [cell for cell, models in cell_values.items()
                          if next(iter(models["A0_cal_peak"].values()))["date"] == date]
            changes = [np.mean([float(r["abs_error_pA"])-float(cell_values[cell]["A1_force_linear"][sweep]["abs_error_pA"])
                                for sweep, r in cell_values[cell]["A2_force_power"].items()])
                       for cell in date_cells]
            print("DATE A2-A1", date, "cells", len(date_cells), "median cell-mean pA", round(float(np.median(changes)), 2))
        responder_deltas = []
        for models in cell_values.values():
            changes = [float(r["abs_error_pA"])-float(models["A1_force_linear"][sweep]["abs_error_pA"])
                       for sweep, r in models["A2_force_power"].items() if int(r["actual_has_event"])]
            if changes:
                responder_deltas.append(float(np.mean(changes)))
        print("RESPONDER-ONLY A2-A1 paired median pA", round(float(np.median(responder_deltas)), 2),
              "improved cells", sum(d < 0 for d in responder_deltas), "of", len(responder_deltas))


if __name__ == "__main__":
    main()
