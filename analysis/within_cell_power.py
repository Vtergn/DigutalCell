"""Reproduce threshold-truncated log-log work/current fits on selected approaches."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1] / "Data" / "2022_pnas" / "2022_pnas" / "bychannel"
OUT = Path(__file__).resolve().parent


def fit_one(x: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    selected = np.isfinite(x) & np.isfinite(y) & (x > threshold) & (y > 0)
    selected &= x > 0
    xx = np.log10(x[selected])
    yy = np.log10(y[selected])
    span = float(np.max(xx) - np.min(xx)) if len(xx) else 0.0
    if len(xx) < 50 or span < 0.05:
        return {"n": len(xx), "log_span": span, "alpha": float("nan"), "log_r2": float("nan")}
    design = np.column_stack([np.ones(len(xx)), xx])
    intercept, alpha = np.linalg.lstsq(design, yy, rcond=None)[0]
    residual = yy - (intercept + alpha * xx)
    variance = np.sum((yy - np.mean(yy)) ** 2)
    return {
        "n": len(xx), "log_span": span, "alpha": float(alpha),
        "log_r2": float(1 - np.sum(residual**2) / variance) if variance > 0 else float("nan"),
    }


def main() -> None:
    with (ROOT / "bychannel_summary.csv").open(newline="", encoding="utf-8-sig") as f:
        summary = {(r["uniqueID"], r["sweep"]): r for r in csv.DictReader(f)}
    traces = defaultdict(lambda: defaultdict(list))
    with (ROOT / "bychannel_approaches.csv").open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            key = (row["uniqueID"], row["sweep"])
            for field in ("force", "work", "absi_blsub_mech"):
                traces[key][field].append(float(row[field]))

    results = []
    for key, values in traces.items():
        meta = summary[key]
        y = np.asarray(values["absi_blsub_mech"])
        work = fit_one(np.asarray(values["work"]), y, float(meta["wthresh"]))
        force = fit_one(np.asarray(values["force"]), y, float(meta["fthresh"]))
        results.append({
            "cell_id": key[0], "channel": meta["construct"], "date": meta["date"],
            "responds": meta["responds"], "wthresh_fJ": meta["wthresh"],
            "fthresh_nN": meta["fthresh"],
            "work_alpha": work["alpha"], "work_log_r2": work["log_r2"],
            "work_n": work["n"], "work_log_span": work["log_span"],
            "force_alpha": force["alpha"], "force_log_r2": force["log_r2"],
            "force_n": force["n"], "force_log_span": force["log_span"],
        })
    with (OUT / "within_cell_power.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    for channel in sorted({r["channel"] for r in results}):
        rows = [r for r in results if r["channel"] == channel and float(r["responds"]) == 1]
        for predictor in ("work", "force"):
            fits = [r for r in rows if np.isfinite(r[f"{predictor}_alpha"])]
            alpha = np.asarray([r[f"{predictor}_alpha"] for r in fits])
            quality = np.asarray([r[f"{predictor}_log_r2"] for r in fits])
            print({
                "channel": channel, "predictor": predictor, "responding_cells": len(rows),
                "fitted_cells": len(fits), "median_alpha": round(float(np.median(alpha)), 3),
                "mean_alpha": round(float(np.mean(alpha)), 3),
                "median_log_r2": round(float(np.median(quality)), 3),
                "alpha_q25_q75": [round(float(np.quantile(alpha, q)), 3) for q in (.25, .75)],
            })


if __name__ == "__main__":
    main()
