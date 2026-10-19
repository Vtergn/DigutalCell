"""Covariate sensitivity checks for response timing and current persistence.

Selected peak force is post-stimulation and may depend on response selection.
These regressions are descriptive diagnostics, not causal adjustment.
Cross-date chimA vs WT P2 persistence remains batch-confounded.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "Data" / "2022_pnas" / "2022_pnas"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def huber_coefficient(x: np.ndarray, y: np.ndarray, scale: float) -> np.ndarray:
    initial = np.linalg.lstsq(x, y, rcond=None)[0]
    return least_squares(lambda b: y-x@b, initial, loss="huber", f_scale=scale,
                         max_nfev=1000).x


def main() -> None:
    rng = np.random.default_rng(20260928)
    # Same-date Piezo2 drug timing. Responders only because their peak timing
    # is defined. Keep the previously frozen three usable dates.
    drug = [r for r in read_csv(DATA / "cytod" / "mp1-mp2-cytod" / "piezocd_summary.csv")
            if r["construct"] in {"mp2-dmso", "mp2-cytod"} and r["responds"] == "1"
            and r["date"] in {"20200930", "20201013", "20201014"}]
    x = np.asarray([[1., float(r["construct"] == "mp2-cytod"),
                     float(r["date"] == "20201013"), float(r["date"] == "20201014"),
                     np.log(float(r["peakf"])), np.log(float(r["Cm"]))] for r in drug])
    delay = np.asarray([float(r["delay"]) for r in drug])
    drug_ols = np.linalg.lstsq(x, delay, rcond=None)[0]
    drug_huber = huber_coefficient(x, delay, scale=1.0)
    # The unusually long delay is disclosed; exclusion is a sensitivity check.
    no_long = delay < 15
    drug_huber_no_long = huber_coefficient(x[no_long], delay[no_long], scale=1.0)
    drug_boot = []
    for _ in range(2000):
        indices = []
        for date in ("20200930", "20201013", "20201014"):
            for construct in ("mp2-dmso", "mp2-cytod"):
                group = [i for i, row in enumerate(drug) if row["date"] == date
                         and row["construct"] == construct]
                indices.extend(rng.choice(group, size=len(group), replace=True))
        indices = np.asarray(indices)
        drug_boot.append(float(huber_coefficient(x[indices], delay[indices], scale=1.0)[1]))

    # Reuse the frozen signed-current/AFM-force audit; equal-cell medians.
    sweeps = read_csv(HERE / "phenotype_force_retention_sweeps.csv")
    by_cell = defaultdict(list)
    for row in sweeps:
        if row["group"] in {"WT_mp2", "chimA"}:
            by_cell[row["cell_id"]].append(row)
    wt_meta = {r["uniqueID"]: r for r in read_csv(DATA / "bychannel" / "bychannel_summary.csv")}
    chim_meta = {r["uniqueID"]: r for r in read_csv(DATA / "chimera_experiments" /
                                                      "capchims_summary.csv")}
    cells = []
    for cell, rows in sorted(by_cell.items()):
        group = rows[0]["group"]
        meta = chim_meta[cell] if group == "chimA" else wt_meta[cell]
        cells.append({"cell_id": cell, "date": rows[0]["date"], "group": group,
                      "current_retention": float(np.median([
                          float(r["current20_from_current_peak_ratio"]) for r in rows])),
                      "force_retention": float(np.median([
                          float(r["force20_from_current_peak_ratio"]) for r in rows])),
                      "peak_force_nN": float(np.median([float(r["force_peak_nN"]) for r in rows])),
                      "Cm_pF": float(meta["Cm"])})
    with (HERE / "force_adjusted_persistence_cells.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)

    def persist_coef(part: list[dict], robust: bool) -> float:
        design = np.asarray([[1., float(r["group"] == "chimA"),
                              r["force_retention"], np.log(r["peak_force_nN"]),
                              np.log(r["Cm_pF"])] for r in part])
        y = np.asarray([r["current_retention"] for r in part])
        if robust:
            return float(huber_coefficient(design, y, scale=0.12)[1])
        return float(np.linalg.lstsq(design, y, rcond=None)[0][1])

    dates = sorted({r["date"] for r in cells})
    lo_date = {date: persist_coef([r for r in cells if r["date"] != date], True)
               for date in dates}
    persist_boot = []
    for _ in range(2000):
        sampled = []
        for date in dates:
            part = [r for r in cells if r["date"] == date]
            sampled.extend(part[i] for i in rng.integers(0, len(part), len(part)))
        persist_boot.append(persist_coef(sampled, True))
    result = {"drug_timing": {
                  "cells": len(drug), "dates": 3,
                  "CytoD_coefficient_ms_OLS": float(drug_ols[1]),
                  "CytoD_coefficient_ms_Huber_1ms": float(drug_huber[1]),
                  "CytoD_coefficient_ms_Huber_excluding_delay_over_15ms": float(
                      drug_huber_no_long[1]),
                  "conditional_cell_bootstrap_95pct_Huber_ms": [float(v) for v in
                                                                  np.quantile(drug_boot, [.025, .975])],
                  "covariates": "date fixed effects, log selected peak force, log cell capacitance",
                  "maximum_delay_ms": float(np.max(delay))},
              "chimA_vs_WT_Piezo2_persistence": {
                  "cells_each": 11,
                  "chimA_coefficient_retention_OLS": persist_coef(cells, False),
                  "chimA_coefficient_retention_Huber": persist_coef(cells, True),
                  "conditional_cell_bootstrap_95pct_Huber": [float(v) for v in
                                                                np.quantile(persist_boot, [.025, .975])],
                  "leave_one_date_out_Huber_coefficients": lo_date,
                  "covariates": "global AFM force 20-ms retention, log peak force, log cell capacitance",
                  "limitation": "No shared dates between chimA and WT Piezo2; fitted difference is descriptive and cannot remove batch effects or identify channel inactivation."}}
    (HERE / "force_adjusted_phenotype_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
