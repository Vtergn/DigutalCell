"""Within-date Piezo2 DMSO/CytoD threshold and delay comparison.

Conditions on author-labeled responders. Comparisons have three independent
dates, and are descriptive because coverslip assignment was not randomized.
"""

from __future__ import annotations

import csv
import json
from math import comb
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "Data" / "2022_pnas" / "2022_pnas" / "cytod" / "mp1-mp2-cytod" / "piezocd_summary.csv"
RNG = np.random.default_rng(20260928)


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8-sig") as handle:
        all_source = [r for r in csv.DictReader(handle) if r["construct"] in
                      {"mp2-dmso", "mp2-cytod"}]
    source = [r for r in all_source if r["responds"] == "1"]
    dates = sorted({r["date"] for r in source if r["construct"] == "mp2-dmso"}
                   & {r["date"] for r in source if r["construct"] == "mp2-cytod"})
    fields = {"log_work_threshold": lambda r: np.log(float(r["wthresh"])),
              "peak_delay_ms": lambda r: float(r["delay"])}
    groups = {(date, condition): [r for r in source if r["date"] == date
                                  and r["construct"] == condition]
              for date in dates for condition in ("mp2-dmso", "mp2-cytod")}
    def difference(field: str, selected: dict | None = None) -> float:
        values = []
        for date in dates:
            samples = {(condition): (groups[date, condition] if selected is None
                                     else selected[date, condition])
                       for condition in ("mp2-dmso", "mp2-cytod")}
            values.append(np.median([fields[field](r) for r in samples["mp2-cytod"]])-
                          np.median([fields[field](r) for r in samples["mp2-dmso"]]))
        return float(np.mean(values))
    observed = {field: difference(field) for field in fields}
    null = {field: [] for field in fields}
    for _ in range(30000):
        sampled = {}
        for date in dates:
            control, treated = groups[date, "mp2-dmso"], groups[date, "mp2-cytod"]
            all_rows = control+treated
            order = RNG.permutation(len(all_rows))
            sampled[date, "mp2-dmso"] = [all_rows[i] for i in order[:len(control)]]
            sampled[date, "mp2-cytod"] = [all_rows[i] for i in order[len(control):]]
        for field in fields:
            null[field].append(difference(field, sampled))
    result = {"dates": dates,
              "responders_control": sum(len(groups[d, "mp2-dmso"]) for d in dates),
              "responders_treated": sum(len(groups[d, "mp2-cytod"]) for d in dates),
              "results": {field: {"mean_of_date_median_CytoD_minus_DMSO": value,
                                  "within_date_label_permutation_two_sided_p": float(
                                      (1+np.sum(np.abs(null[field]) >= abs(value))) /
                                      (len(null[field])+1))}
                          for field, value in observed.items()},
              "responder_audit": {},
              "limits": "Responder-only analysis; 3 dates; unequal achieved peak forces; CytoD is a broad actin perturbation and not FLNB KO."}
    zero_treated_nonresponders_probability = 1.0
    for date in dates:
        part = [r for r in all_source if r["date"] == date]
        control_total = sum(r["construct"] == "mp2-dmso" for r in part)
        nonresponders = sum(r["responds"] != "1" for r in part)
        treated_nonresponders = sum(r["construct"] == "mp2-cytod" and
                                  r["responds"] != "1" for r in part)
        if treated_nonresponders:
            raise ValueError("Expected 0 treated nonresponders in frozen cohort")
        zero_treated_nonresponders_probability *= (comb(control_total, nonresponders)/
                                                    comb(len(part), nonresponders))
    result["responder_audit"] = {
        "DMSO_responders_over_total": "14/19", "CytoD_responders_over_total": "18/18",
        "conditional_probability_all_five_nonresponders_in_DMSO":
            zero_treated_nonresponders_probability,
        "interpretation": "Exact one-sided label-exchange calculation within dates, conditional on observed totals; not a randomized causal test."}
    (HERE / "cytod_phenotype_axes_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
