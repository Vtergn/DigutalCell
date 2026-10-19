"""Date-stratified construct contrast in the Duke chimera summary.

`chimB` is treated as an anonymous second construct: its molecular definition
was not verified. Its raw Feather records are absent locally. Only the author
cell-level summary is used; no waveform or mechanism is imputed for chimB.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "Data" / "2022_pnas" / "2022_pnas" / "chimera_experiments" / "capchims_summary.csv"
RNG = np.random.default_rng(20260928)
ENDPOINTS = {"log_work_threshold": lambda r: np.log(float(r["wthresh"])),
             "peak_delay_ms": lambda r: float(r["delay"]),
             "peak_current_density_pA_per_pF": lambda r: float(r["peaki"])/float(r["Cm"]),
             "peak_force_nN": lambda r: float(r["peakf"])}


def statistic(rows: list[dict], field: str, labels: dict[str, str] | None = None) -> float:
    per_date = []
    for date in sorted({r["date"] for r in rows}):
        groups = {}
        for name in ("chimA", "chimB"):
            values = [ENDPOINTS[field](r) for r in rows if r["date"] == date and
                      (labels[r["uniqueID"]] if labels else r["construct"]) == name]
            groups[name] = float(np.median(values))
        per_date.append(groups["chimA"]-groups["chimB"])
    return float(np.mean(per_date))


def force_adjusted_delay_coefficient(rows: list[dict],
                                     labels: dict[str, str] | None = None) -> float:
    """Sensitivity diagnostic, not a causal adjustment (force is post-construct)."""
    x = np.asarray([[1., float((labels[r["uniqueID"]] if labels else r["construct"]) == "chimA"),
                     float(r["date"] == "20201110"), np.log(float(r["peakf"])),
                     np.log(float(r["Cm"]))] for r in rows])
    y = np.asarray([float(r["delay"]) for r in rows])
    return float(np.linalg.lstsq(x, y, rcond=None)[0][1])


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8-sig") as handle:
        rows = [r for r in csv.DictReader(handle) if r["construct"] in {"chimA", "chimB"}]
    dates = sorted({r["date"] for r in rows})
    if dates != ["20201108", "20201110"]:
        raise ValueError(f"Construct dates changed: {dates}")
    counts = {(date, group): sum(r["date"] == date and r["construct"] == group for r in rows)
              for date in dates for group in ("chimA", "chimB")}
    dated = []
    for date in dates:
        for group in ("chimA", "chimB"):
            part = [r for r in rows if r["date"] == date and r["construct"] == group]
            dated.append({"date": date, "group": group, "cells": len(part),
                          "voltage_mV": float(np.median([float(r["vstep"]) for r in part])),
                          "loading_speed_um_per_s": float(np.median([float(r["velocity"]) for r in part])),
                          "cantilever_k_N_per_m": float(np.median([float(r["kcant"]) for r in part])),
                          "median_Cm_pF": float(np.median([float(r["Cm"]) for r in part])),
                          **{f"median_{field}": float(np.median([fn(r) for r in part]))
                             for field, fn in ENDPOINTS.items()}})
    with (HERE / "chimera_same_day_dates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dated[0]))
        writer.writeheader()
        writer.writerows(dated)

    observed = {field: statistic(rows, field) for field in ENDPOINTS}
    adjusted_delay = force_adjusted_delay_coefficient(rows)
    # Condition on both observed dates and each date's construct counts.
    permuted = {field: [] for field in ENDPOINTS}
    adjusted_null = []
    for _ in range(50000):
        labels = {}
        for date in dates:
            names = [r["uniqueID"] for r in rows if r["date"] == date]
            chosen = set(RNG.choice(names, counts[date, "chimA"], replace=False))
            labels.update({name: "chimA" if name in chosen else "chimB" for name in names})
        for field in ENDPOINTS:
            permuted[field].append(statistic(rows, field, labels))
        adjusted_null.append(force_adjusted_delay_coefficient(rows, labels))
    # Cell bootstrap is conditional on these two dates; it is not a CI over
    # future experimental dates or coverslips.
    bootstrapped = {field: [] for field in ENDPOINTS}
    by_group = {(date, group): [r for r in rows if r["date"] == date and r["construct"] == group]
                for date in dates for group in ("chimA", "chimB")}
    for _ in range(10000):
        sample = []
        for date in dates:
            for group in ("chimA", "chimB"):
                part = by_group[date, group]
                sample.extend(part[i] for i in RNG.integers(0, len(part), len(part)))
        for field in ENDPOINTS:
            bootstrapped[field].append(statistic(sample, field))
    results = {"source": str(SOURCE), "dates": dates,
               "construct_definition": "chimA=Piezo1 with Piezo2 cap; chimB molecular identity unverified",
               "endpoints": {field: {"observed_mean_of_date_median_A_minus_B": value,
                                     "conditional_cell_bootstrap_95pct": [float(x) for x in np.quantile(
                                         bootstrapped[field], [.025, .975])],
                                     "within_date_label_permutation_two_sided_p": float(
                                         (1+np.sum(np.abs(permuted[field]) >= abs(value))) /
                                         (len(permuted[field])+1))}
                             for field, value in observed.items()},
               "force_adjusted_delay_sensitivity": {
                   "A_minus_B_ms": adjusted_delay,
                   "covariates": "date, log selected peak force, log cell capacitance",
                   "within_date_label_permutation_two_sided_p": float(
                       (1+np.sum(np.abs(adjusted_null) >= abs(adjusted_delay))) /
                       (len(adjusted_null)+1)),
                   "A_minus_B_ms_excluding_delay_over_8ms": force_adjusted_delay_coefficient(
                       [r for r in rows if float(r["delay"]) < 8])},
               "limits": "Only two dates; cells may share coverslips; assignment was not verified randomized; chimB raw traces are absent locally; summary-selected peaks have unequal stimulus force."}
    (HERE / "chimera_same_day_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.3), layout="constrained")
    colors = {"chimA": "#72549a", "chimB": "#a5753b"}
    for ax, field, ylabel in zip(axes,
                                 ("log_work_threshold", "peak_delay_ms"),
                                 ("Log work threshold (fJ)", "Peak delay (ms)")):
        for date_index, date in enumerate(dates):
            for group_index, group in enumerate(("chimA", "chimB")):
                part = by_group[date, group]
                values = [ENDPOINTS[field](r) for r in part]
                center = 2*date_index+group_index
                ax.scatter(center+RNG.uniform(-.10, .10, len(values)), values,
                           s=30, alpha=.85, color=colors[group])
                ax.plot([center-.2, center+.2], [np.median(values)]*2,
                        color="black", linewidth=2)
        ax.set_xticks(range(4), ["Nov 8\nA", "Nov 8\nB", "Nov 10\nA", "Nov 10\nB"])
        ax.set_ylabel(ylabel)
    fig.suptitle("Same-date chimera summary; chimB molecular identity unverified")
    fig.savefig(HERE / "chimera_same_day.png", dpi=180)
    plt.close(fig)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
