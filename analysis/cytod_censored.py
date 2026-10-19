"""Exploratory date-adjusted lognormal threshold model with right censoring.

For nonresponders, the selected sweep's peak work is a conservative lower
bound on the unobserved response threshold. This assumes the response flag
correctly describes that sweep and does not establish a causal drug effect.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr

from cytod_analysis import GROUPS, ROOT, HERE


def fit(rows: list[dict], dates: list[str]) -> dict:
    n_dates = len(dates)
    index = np.asarray([dates.index(r["date"]) for r in rows])
    treated = np.asarray([r["treated"] for r in rows], dtype=float)
    event = np.asarray([r["responds"] == "1" for r in rows])
    work = np.asarray([float(r["wthresh"] if r["responds"] == "1" else r["peakw"]) for r in rows])
    if np.any(work <= 0):
        raise ValueError("All observed thresholds and censoring works must be positive")
    log_work = np.log(work)

    def objective(params: np.ndarray, fixed_beta: float | None = None) -> float:
        day_means = params[:n_dates]
        beta = params[n_dates] if fixed_beta is None else fixed_beta
        log_sigma = params[n_dates+1] if fixed_beta is None else params[n_dates]
        sigma = np.exp(log_sigma)
        z = (log_work - day_means[index] - beta*treated)/sigma
        density = -.5*z*z - log_sigma - np.log(2*np.pi)/2
        survival = log_ndtr(-z)
        return float(-np.sum(np.where(event, density, survival)))

    initial_day = [np.median(log_work[index == j]) for j in range(n_dates)]
    start = np.asarray(initial_day + [0.0, np.log(max(float(np.std(log_work)), 0.2))])
    bounds = [(None, None)]*n_dates + [(-4, 4), (-3, 3)]
    result = minimize(objective, start, method="L-BFGS-B", bounds=bounds)
    if not result.success:
        raise RuntimeError(f"Full censored fit failed: {result.message}")
    null_start = np.r_[result.x[:n_dates], result.x[n_dates+1]]
    null = minimize(lambda p: objective(p, 0.0), null_start, method="L-BFGS-B", bounds=bounds[:n_dates]+[bounds[-1]])
    if not null.success:
        raise RuntimeError(f"Null censored fit failed: {null.message}")
    return {
        "threshold_ratio_cytod_control": float(np.exp(result.x[n_dates])),
        "log_threshold_difference": float(result.x[n_dates]),
        "fitted_sigma_log_work": float(np.exp(result.x[n_dates+1])),
        "likelihood_ratio_statistic": float(2*(null.fun-result.fun)),
        "responders_control": int(np.sum(event & (treated == 0))),
        "nonresponders_control": int(np.sum(~event & (treated == 0))),
        "responders_cytod": int(np.sum(event & (treated == 1))),
        "nonresponders_cytod": int(np.sum(~event & (treated == 1))),
    }


def main() -> None:
    output = []
    for label, filename, control, treatment in GROUPS:
        with (ROOT / filename).open(newline="", encoding="utf-8-sig") as handle:
            source = list(csv.DictReader(handle))
        dates = sorted({r["date"] for r in source if r["construct"] == control}
                       & {r["date"] for r in source if r["construct"] == treatment})
        # A date with every cell censored cannot identify its day intercept.
        dates = [date for date in dates if any(r["date"] == date and r["construct"] in {control,treatment}
                                                 and r["responds"] == "1" for r in source)]
        rows = [{**r, "treated": int(r["construct"] == treatment)} for r in source
                if r["date"] in dates and r["construct"] in {control, treatment}]
        result = {"channel": label, "usable_dates": len(dates), "cells": len(rows), **fit(rows, dates)}
        output.append(result)
        print(result)
    with (HERE / "cytod_censored_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)


if __name__ == "__main__":
    main()
