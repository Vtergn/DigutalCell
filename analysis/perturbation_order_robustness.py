"""Audit acquisition-block confounding and leave-date-out perturbation contrasts.

Cell suffixes indicate recording sequence only if the dataset's naming convention
preserves acquisition order. This script treats them as ordinal labels and makes
no randomized-treatment or causal claim.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from cytod_censored import fit


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "Data" / "2022_pnas" / "2022_pnas"
DRUG = DATA / "cytod" / "mp1-mp2-cytod" / "piezocd_summary.csv"
CHIM = DATA / "chimera_experiments" / "capchims_summary.csv"


def load(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def med(rows: list[dict], field: str) -> float:
    return float(np.median([float(r[field]) for r in rows]))


def block_sequence(rows: list[dict]) -> str:
    return " ".join(f"{int(r['cell'])}:{r['construct']}" for r in
                    sorted(rows, key=lambda r: int(r["cell"])))


def robust_delay(rows: list[dict], order: bool = False) -> float:
    dates = sorted({r["date"] for r in rows})
    x = np.asarray([[1.0, float(r["construct"] == "chimA"),
                     *[float(r["date"] == d) for d in dates[1:]],
                     np.log(float(r["peakf"])), np.log(float(r["Cm"])),
                     *([float(r["cell"])] if order else [])] for r in rows])
    y = np.asarray([float(r["delay"]) for r in rows])
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    return float(least_squares(lambda b: y - x@b, beta, loss="huber",
                               f_scale=1.0).x[1])


def main() -> None:
    drug = [r for r in load(DRUG) if r["construct"] in {"mp2-dmso", "mp2-cytod"}]
    chim = [r for r in load(CHIM) if r["construct"] in {"chimA", "chimB"}]
    active = ["20200930", "20201013", "20201014"]
    all_drug_dates = ["20200311", *active]
    rows = []
    for date in all_drug_dates:
        part = [r for r in drug if r["date"] == date]
        for condition in ("mp2-cytod", "mp2-dmso"):
            group = [r for r in part if r["construct"] == condition]
            responders = [r for r in group if r["responds"] == "1"]
            rows.append({"date": date, "condition": condition,
                         "cells": len(group), "responders": len(responders),
                         "min_cell_number": min(int(r["cell"]) for r in group),
                         "max_cell_number": max(int(r["cell"]) for r in group),
                         "median_peak_force_nN": med(group, "peakf"),
                         "median_peak_work_fJ": med(group, "peakw"),
                         "median_Cm_pF": med(group, "Cm"),
                         "median_responder_work_threshold_fJ":
                             med(responders, "wthresh") if responders else ""})
    with (HERE / "perturbation_order_condition_table.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    loo = {}
    for omitted in active:
        kept = [date for date in active if date != omitted]
        subset = [{**r, "treated": int(r["construct"] == "mp2-cytod")}
                  for r in drug if r["date"] in kept]
        model = fit([r for r in subset if r["responds"] == "1"], kept)
        date_ratios = []
        for date in kept:
            control = [r for r in subset if r["date"] == date and
                       r["construct"] == "mp2-dmso" and r["responds"] == "1"]
            treated = [r for r in subset if r["date"] == date and
                       r["construct"] == "mp2-cytod" and r["responds"] == "1"]
            date_ratios.append(np.log(med(treated, "wthresh") /
                                      med(control, "wthresh")))
        loo[omitted] = {"remaining_dates": kept,
                        "date_median_geometric_threshold_ratio":
                            float(np.exp(np.mean(date_ratios))),
                        "parametric_responder_only_threshold_ratio":
                            model["threshold_ratio_cytod_control"]}

    # Chimeras are interleaved by cell number on both dates; check whether a
    # smooth acquisition-order trend removes their delay contrast.
    chim_sorted = {date: block_sequence([r for r in chim if r["date"] == date])
                   for date in sorted({r["date"] for r in chim})}
    rng = np.random.default_rng(20260928)
    chim_boot = []
    for _ in range(2000):
        sampled = []
        for date in sorted(chim_sorted):
            for condition in ("chimA", "chimB"):
                group = [r for r in chim if r["date"] == date and
                         r["construct"] == condition]
                sampled.extend(group[i] for i in rng.integers(0, len(group), len(group)))
        chim_boot.append(robust_delay(sampled, order=True))
    result = {"drug_cell_number_sequence": {
                  date: block_sequence([r for r in drug if r["date"] == date])
                  for date in all_drug_dates},
              "drug_leave_one_active_date_out": loo,
              "chimera_cell_number_sequence": chim_sorted,
              "chimera_A_minus_B_robust_delay_ms": {
                  "date_force_Cm": robust_delay(chim),
                  "date_force_Cm_linear_cell_number": robust_delay(chim, order=True),
                  "within_date_condition_cell_bootstrap_95pct_with_order":
                      [float(v) for v in np.quantile(chim_boot, [.025, .975])],
                  "single_date_with_order": {
                      date: robust_delay([r for r in chim if r["date"] == date],
                                         order=True)
                      for date in sorted(chim_sorted)}},
              "interpretation_limit": "Cell suffix is a sequence proxy, not a verified timestamp. All Piezo2 CytoD and DMSO cell-number blocks are separated within date, so any time/order or coverslip effect is confounded with treatment. Chimeras are interleaved by cell number, but molecular identity of chimB remains unknown."}
    (HERE / "perturbation_order_robustness_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
