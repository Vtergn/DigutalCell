"""Weak, date-stratified Duke CytoD direction check for a PIEZO2 tether module.

This is a descriptive perturbation audit, not a FLNB-specific or causal model.
PIEZO1 and PIEZO2 were measured on different dates; PIEZO1's untreated label
is not an explicit DMSO match. Never pool dates for a channel interaction test.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SUMMARY = HERE.parent / "Data" / "2022_pnas" / "2022_pnas" / "cytod" / "mp1-mp2-cytod" / "piezocd_summary.csv"
CONDITIONS = {"Piezo1": ("mp1", "mp1cytod", "Piezo1_control", "Piezo1_CytoD"),
              "Piezo2": ("mp2-dmso", "mp2-cytod", "Piezo2_DMSO", "Piezo2_CytoD")}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def median_positive(rows: list[dict], field: str) -> float:
    values = [float(row[field]) for row in rows if float(row[field]) > 0]
    return float(np.median(values)) if values else float("nan")


def main() -> None:
    source = read_csv(SUMMARY)
    shape = read_csv(HERE / "frozen_wt_projection_scores.csv")
    dates = []
    for channel, (control, treated, shape_control, shape_treated) in CONDITIONS.items():
        common = sorted({r["date"] for r in source if r["construct"] == control} &
                        {r["date"] for r in source if r["construct"] == treated})
        for date in common:
            all_control = [r for r in source if r["construct"] == control and r["date"] == date]
            all_treated = [r for r in source if r["construct"] == treated and r["date"] == date]
            responsive = [[r for r in group if r["responds"] == "1"]
                          for group in (all_control, all_treated)]
            if not all(map(len, responsive)):
                continue
            medians = {field: [median_positive(group, field) for group in responsive]
                       for field in ("wthresh", "peaki", "peakf", "peakw")}
            cell_shapes = []
            for group_name in (shape_control, shape_treated):
                by_cell = defaultdict(list)
                for row in shape:
                    if row["date"] == date and row["group"] == group_name:
                        by_cell[row["cell_id"]].append(float(row["postpeak_20ms_retention"]))
                cell_shapes.append([float(np.median(v)) for v in by_cell.values()])
            output = {"channel": channel, "date": date,
                      "control_label": control, "treated_label": treated,
                      "control_cells": len(all_control), "treated_cells": len(all_treated),
                      "control_responders": len(responsive[0]),
                      "treated_responders": len(responsive[1]),
                      "shape_control_cells": len(cell_shapes[0]),
                      "shape_treated_cells": len(cell_shapes[1])}
            for field, (c, t) in medians.items():
                output[f"control_median_{field}"] = c
                output[f"treated_median_{field}"] = t
                output[f"treated_control_{field}_ratio"] = t/c if c > 0 else float("nan")
            if all(map(len, cell_shapes)):
                control_shape, treated_shape = map(np.median, cell_shapes)
                output["control_postpeak20_retention"] = float(control_shape)
                output["treated_postpeak20_retention"] = float(treated_shape)
                output["treated_minus_control_postpeak20"] = float(treated_shape-control_shape)
            else:
                output["control_postpeak20_retention"] = float("nan")
                output["treated_postpeak20_retention"] = float("nan")
                output["treated_minus_control_postpeak20"] = float("nan")
            dates.append(output)
    with (HERE / "module_p2_cytod_date_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dates[0]))
        writer.writeheader()
        writer.writerows(dates)
    result = {}
    for channel in CONDITIONS:
        group = [row for row in dates if row["channel"] == channel]
        result[channel] = {"usable_dates": len(group),
                           "date_geomean_threshold_ratio": float(np.exp(np.mean([
                               np.log(row["treated_control_wthresh_ratio"]) for row in group]))),
                           "dates_lower_threshold": sum(row["treated_control_wthresh_ratio"] < 1 for row in group),
                           "dates_higher_peak_current": sum(row["treated_control_peaki_ratio"] > 1 for row in group),
                           "dates_lower_peak_force": sum(row["treated_control_peakf_ratio"] < 1 for row in group),
                           "dates_lower_peak_work": sum(row["treated_control_peakw_ratio"] < 1 for row in group),
                           "dates_lower_postpeak20_retention": sum(
                               row["treated_minus_control_postpeak20"] < 0 for row in group),
                           "control_responders": sum(row["control_responders"] for row in group),
                           "control_total": sum(row["control_cells"] for row in group),
                           "treated_responders": sum(row["treated_responders"] for row in group),
                           "treated_total": sum(row["treated_cells"] for row in group)}
    result["limits"] = "Channel dates do not overlap; Piezo1 comparator is not explicit DMSO; responder-only amplitude and shape can be selected; doses and cells differ."
    (HERE / "module_p2_cytod_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
