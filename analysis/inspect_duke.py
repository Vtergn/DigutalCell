"""Read-only first-pass audit of the Young et al. AFM/electrophysiology tables."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1] / "Data" / "2022_pnas" / "2022_pnas"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def number(value: str | None) -> float:
    try:
        return float(value) if value not in (None, "") else math.nan
    except ValueError:
        return math.nan


def rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    ends = np.r_[starts[1:], len(values)]
    result = np.empty(len(values), dtype=float)
    for start, end in zip(starts, ends):
        result[order[start:end]] = (start + end - 1) / 2
    return result


def spearman(rows: list[dict[str, str]], x: str, y: str) -> dict[str, float | int | None]:
    pairs = np.asarray([(number(row.get(x)), number(row.get(y))) for row in rows])
    pairs = pairs[np.isfinite(pairs).all(axis=1)]
    if len(pairs) < 4 or np.unique(pairs[:, 0]).size < 2 or np.unique(pairs[:, 1]).size < 2:
        return {"n": len(pairs), "rho": None}
    return {"n": len(pairs), "rho": round(float(np.corrcoef(rank(pairs[:, 0]), rank(pairs[:, 1]))[0, 1]), 3)}


def describe(rows: list[dict[str, str]], field: str) -> dict[str, float | int | None]:
    values = np.asarray([number(row.get(field)) for row in rows])
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0, "median": None, "q25": None, "q75": None, "min": None, "max": None}
    return {
        "n": len(values),
        "median": round(float(np.median(values)), 3),
        "q25": round(float(np.quantile(values, 0.25)), 3),
        "q75": round(float(np.quantile(values, 0.75)), 3),
        "min": round(float(np.min(values)), 3),
        "max": round(float(np.max(values)), 3),
    }


def main() -> None:
    summary = read_csv(ROOT / "bychannel" / "bychannel_summary.csv")
    groups = defaultdict(list)
    for row in summary:
        groups[row["construct"]].append(row)

    output = {
        "summary_rows": len(summary),
        "unique_cell_ids": len({row["uniqueID"] for row in summary}),
        "unique_dates": len({row["date"] for row in summary}),
        "duplicate_cell_ids": {key: count for key, count in Counter(row["uniqueID"] for row in summary).items() if count > 1},
        "missing_by_field": {key: sum(row.get(key, "") == "" for row in summary) for key in summary[0] if key},
        "groups": {},
    }

    fields = ["peakf", "peakw", "peaki", "delay", "fthresh", "wthresh", "Cm", "vstep", "velocity", "offset", "stdev"]
    for name, rows in sorted(groups.items()):
        output["groups"][name] = {
            "cells": len(rows),
            "responsive": sum(number(row.get("responds")) == 1 for row in rows),
            "dates": len({row["date"] for row in rows}),
            "stats": {field: describe(rows, field) for field in fields},
            "rho_peakf_peaki": spearman(rows, "peakf", "peaki"),
            "rho_peakw_peaki": spearman(rows, "peakw", "peaki"),
            "rho_peakf_peakw": spearman(rows, "peakf", "peakw"),
            "rho_vstep_peaki": spearman(rows, "vstep", "peaki"),
        }

    approaches = ROOT / "bychannel" / "bychannel_approaches.csv"
    trace_counts = Counter()
    trace_ranges = defaultdict(lambda: [math.inf, -math.inf, math.inf, -math.inf])
    trace_current_max = defaultdict(lambda: -math.inf)
    with approaches.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            key = (row["uniqueID"], row["sweep"])
            trace_counts[key] += 1
            force, work = number(row.get("force")), number(row.get("work"))
            current = number(row.get("absi_blsub_mech"))
            if math.isfinite(current):
                trace_current_max[key] = max(trace_current_max[key], current)
            if math.isfinite(force):
                trace_ranges[key][0] = min(trace_ranges[key][0], force)
                trace_ranges[key][1] = max(trace_ranges[key][1], force)
            if math.isfinite(work):
                trace_ranges[key][2] = min(trace_ranges[key][2], work)
                trace_ranges[key][3] = max(trace_ranges[key][3], work)

    selected = {(row["uniqueID"], row["sweep"]) for row in summary}
    output["approaches"] = {
        "rows": sum(trace_counts.values()),
        "unique_cell_sweeps": len(trace_counts),
        "unique_cells": len({key[0] for key in trace_counts}),
        "summary_cell_sweeps_present": len(selected & trace_counts.keys()),
        "trace_points": {
            "median": int(np.median(list(trace_counts.values()))),
            "min": min(trace_counts.values()),
            "max": max(trace_counts.values()),
        },
        "negative_force_traces": sum(bounds[0] < 0 for bounds in trace_ranges.values()),
        "negative_work_traces": sum(bounds[2] < 0 for bounds in trace_ranges.values()),
        "mech_current_peak_by_channel": {
            name: describe(
                [{"current": trace_current_max[(row["uniqueID"], row["sweep"])]} for row in rows],
                "current",
            )
            for name, rows in sorted(groups.items())
        },
        "peak_minus_offset_vs_mech_trace_peak_median_abs_difference_pA": round(
            float(np.median([
                abs(number(row["peaki"]) - number(row["offset"]) - trace_current_max[(row["uniqueID"], row["sweep"])])
                for row in summary
            ])), 3
        ),
    }
    print(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
