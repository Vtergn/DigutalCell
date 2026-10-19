"""Piezo2 CytoD two-part sensitivity and right-censoring scenario audit.

The all-censored model assumes every nonresponder is a functional cell whose
threshold exceeds its largest *observed* work. The scenario grid alternatively
drops some nonresponders as possible expression/recording failures. These are
assumptions, not identified cell states or causal treatment effects.
"""

from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from cytod_censored import fit


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "Data" / "2022_pnas" / "2022_pnas" / "cytod" / "mp1-mp2-cytod"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    source = [r for r in read_csv(ROOT / "piezocd_summary.csv")
              if r["construct"] in {"mp2-dmso", "mp2-cytod"}]
    all_dates = sorted({r["date"] for r in source if r["construct"] == "mp2-dmso"}
                       & {r["date"] for r in source if r["construct"] == "mp2-cytod"})
    active_dates = [date for date in all_dates if any(
        r["date"] == date and r["responds"] == "1" for r in source)]
    records = [{**r, "treated": int(r["construct"] == "mp2-cytod")}
               for r in source if r["date"] in active_dates]
    censored = [r for r in records if r["responds"] != "1"]
    if len(censored) != 5:
        raise ValueError(f"Frozen cohort changed: {len(censored)} nonresponders")
    summary_paths = {p.name: p for p in ROOT.rglob("*_summary.csv")}
    censor_audit = []
    for row in censored:
        filename = Path(row["path"]).name.replace("_preprocessed.feather", "_summary.csv")
        if filename not in summary_paths:
            raise FileNotFoundError(f"Missing per-sweep work for {row['uniqueID']}")
        sweeps = read_csv(summary_paths[filename])
        tested = [float(sweep["peakw"]) for sweep in sweeps
                  if np.isfinite(float(sweep["peakw"])) and float(sweep["peakw"]) > 0]
        if not tested:
            raise ValueError(f"No positive tested work: {row['uniqueID']}")
        selected = float(row["peakw"])
        maximum = float(max(tested))
        if maximum + 1e-6 < selected:
            raise ValueError(f"Per-sweep maximum below selected sweep: {row['uniqueID']}")
        censor_audit.append({"cell_id": row["uniqueID"], "date": row["date"],
                             "selected_peak_work_fJ": selected,
                             "max_tested_work_fJ": maximum,
                             "additional_work_fJ": maximum-selected,
                             "recorded_sweeps": len(sweeps)})
        row["peakw"] = str(maximum)
    with (HERE / "cytod_joint_censor_bounds.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(censor_audit[0]))
        writer.writeheader()
        writer.writerows(censor_audit)

    all_censored = fit(records, active_dates)
    responders_only = fit([r for r in records if r["responds"] == "1"], active_dates)
    scenarios = []
    for count in range(len(censored)+1):
        for subset in combinations([r["uniqueID"] for r in censored], count):
            retained = set(subset)
            chosen = [r for r in records if r["responds"] == "1" or
                      r["uniqueID"] in retained]
            result = fit(chosen, active_dates)
            scenarios.append({"censored_nonresponders": count,
                              "retained_ids": "|".join(subset),
                              "cells_used": len(chosen),
                              "threshold_ratio_cytod_DMSO": result[
                                  "threshold_ratio_cytod_control"]})
    with (HERE / "cytod_joint_censor_scenarios.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scenarios[0]))
        writer.writeheader()
        writer.writerows(scenarios)
    group_counts = {}
    for label, dates in (("all_four_matched_dates", all_dates),
                         ("three_dates_with_any_response", active_dates)):
        group_counts[label] = {condition: {
            "responsive": sum(r["date"] in dates and r["construct"] == condition and
                              r["responds"] == "1" for r in source),
            "total": sum(r["date"] in dates and r["construct"] == condition for r in source)}
            for condition in ("mp2-dmso", "mp2-cytod")}
    by_count = {str(count): {"min_ratio": float(min(s["threshold_ratio_cytod_DMSO"]
                                                    for s in scenarios if s["censored_nonresponders"] == count)),
                             "max_ratio": float(max(s["threshold_ratio_cytod_DMSO"]
                                                    for s in scenarios if s["censored_nonresponders"] == count))}
                for count in range(6)}
    result = {"dates": {"all_matched": all_dates, "usable_for_threshold": active_dates},
              "response_probability_component": group_counts,
              "right_censoring_component": {
                  "all_five_true_high_threshold": all_censored,
                  "none_of_five_assumed_true_high_threshold": responders_only,
                  "max_tested_work_exceeds_selected_in_cells": sum(
                      item["additional_work_fJ"] > 1e-6 for item in censor_audit),
                  "scenario_ratio_range_by_number_censored": by_count},
              "limits": "A date with 0 responders in both arms cannot identify its threshold intercept. Right censoring assumes each retained nonresponder was functional and never responded at any recorded work. Other scenarios drop possible technical/expression failures. The two assumptions are not distinguishable here; no causal treatment estimate."}
    (HERE / "cytod_joint_sensitivity_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
