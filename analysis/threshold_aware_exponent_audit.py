"""Cell-level audit of apparent work-current exponents in Young raw approaches.

The published exponent is an OLS log(I)~log(W) slope after threshold removal.
This script tests how a threshold-shifted common exponent maps to that slope
on each cell's *observed* work trajectory. Time samples never serve as
independent observations for between-channel inference.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "Data" / "2022_pnas" / "2022_pnas"
RNG = np.random.default_rng(20260928)
CHANNELS = ("mp1", "mp2", "mtraak", "mtrek1")


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_traces(folder, summary_name, trace_name, allowed):
    meta = {r["uniqueID"]: r for r in read_csv(folder / summary_name)
            if r["construct"] in allowed}
    traces = defaultdict(lambda: ([], []))
    with (folder / trace_name).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            cell = row["uniqueID"]
            if cell in meta and row["sweep"] == meta[cell]["sweep"]:
                traces[cell][0].append(float(row["work"]))
                traces[cell][1].append(float(row["absi_blsub_mech"]))
    return [(meta[c], np.asarray(w), np.asarray(i)) for c, (w, i) in traces.items()]


def ols(x, y):
    if len(x) < 50 or np.ptp(x) < 0.02:
        return np.nan, np.nan
    b = np.polyfit(x, y, 1)
    residual = y - np.polyval(b, x)
    total = np.sum((y - np.mean(y)) ** 2)
    return float(b[0]), float(1 - np.sum(residual ** 2) / total) if total > 0 else np.nan


def fit_pair(w, i, w0):
    valid = np.isfinite(w) & np.isfinite(i) & (w > w0) & (w > 0) & (i > 0)
    ww, yy = w[valid], i[valid]
    if len(ww) < 50 or np.ptp(np.log10(ww)) < 0.05:
        return None
    lx, ly = np.log10(ww), np.log10(yy)
    alpha, r2 = ols(lx, ly)
    # Same points and weights for the two fits. Threshold uncertainty is audited
    # separately; a fitted W0 would use the response twice.
    z = np.log10(ww - w0)
    beta, shifted_r2 = ols(z, ly)
    geom, _ = ols(z, lx)
    # The common-beta null gives alpha_null=beta*cov(logW,log(W-W0))/var(logW).
    geometric_factor = float(np.cov(lx, z, bias=True)[0, 1] / np.var(lx))
    # Geometry-only model uses one common beta but each observed W path/W0.
    return dict(n=int(len(ww)), alpha=alpha, alpha_r2=r2, beta=beta,
                shifted_r2=shifted_r2, span=float(np.ptp(lx)),
                wmin=float(np.min(ww)), wmax=float(np.max(ww)),
                threshold_fraction=float(w0 / np.max(ww)),
                geometric_factor=geometric_factor, geom_inverse=geom)


def matched_slopes(w, i, w0, span=0.10, n_grid=50):
    valid = np.isfinite(w) & np.isfinite(i) & (w > w0) & (i > 0)
    lx, ly = np.log10(w[valid]), np.log10(i[valid])
    if len(lx) < 50 or np.ptp(lx) < span:
        return np.nan, np.nan
    # Sort and aggregate duplicate abscissae before interpolation. Both ends of
    # the observed log-work range are tested; one chosen window would be biased.
    order = np.argsort(lx)
    xx, inverse = np.unique(lx[order], return_inverse=True)
    sums = np.bincount(inverse, weights=ly[order])
    counts = np.bincount(inverse)
    yy = sums / counts
    low = np.linspace(xx[0], xx[0] + span, n_grid)
    high = np.linspace(xx[-1] - span, xx[-1], n_grid)
    return (ols(low, np.interp(low, xx, yy))[0],
            ols(high, np.interp(high, xx, yy))[0])


def med(rows, key):
    a = np.asarray([r[key] for r in rows], float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if len(a) else np.nan


def bootstrap_median_diff(rows, key, a="mp1", b="mp2", n=4000):
    x = np.asarray([r[key] for r in rows if r["channel"] == a and np.isfinite(r[key])])
    y = np.asarray([r[key] for r in rows if r["channel"] == b and np.isfinite(r[key])])
    if not len(x) or not len(y):
        return None
    draws = np.median(RNG.choice(x, (n, len(x)), replace=True), axis=1) - np.median(
        RNG.choice(y, (n, len(y)), replace=True), axis=1)
    return [float(np.median(x) - np.median(y)), *map(float, np.quantile(draws, [.025, .975]))]


def date_cluster_bootstrap(rows, key, n=4000):
    grouped = {}
    for channel in ("mp1", "mp2"):
        by_date = defaultdict(list)
        for row in rows:
            if row["channel"] == channel and np.isfinite(row[key]):
                by_date[row["date"]].append(float(row[key]))
        grouped[channel] = list(by_date.values())
    draws = []
    for _ in range(n):
        medians = {}
        for channel, blocks in grouped.items():
            sampled = RNG.integers(0, len(blocks), len(blocks))
            medians[channel] = np.median([value for index in sampled for value in blocks[index]])
        draws.append(medians["mp1"] - medians["mp2"])
    return list(map(float, np.quantile(draws, [.5, .025, .975])))


def main():
    traces = load_traces(SOURCE / "bychannel", "bychannel_summary.csv",
                         "bychannel_approaches.csv", CHANNELS)
    chims = load_traces(SOURCE / "chimera_experiments", "capchims_summary.csv",
                        "capchims_approaches.csv", {"chimA"})
    rows = []
    for meta, w, i in traces + chims:
        if meta["construct"] != "chimA" and float(meta["responds"]) != 1:
            continue
        w0 = float(meta["wthresh"])
        fit = fit_pair(w, i, w0)
        if fit is None:
            continue
        lo10, hi10 = matched_slopes(w, i, w0, 0.10)
        lo15, hi15 = matched_slopes(w, i, w0, 0.15)
        rows.append(dict(cell_id=meta["uniqueID"], date=meta["date"],
                         channel=meta["construct"], wthresh=w0,
                         matched_early_010=lo10, matched_late_010=hi10,
                         matched_early_015=lo15, matched_late_015=hi15,
                         **fit))
    # A pooled within-cell slope is descriptive, not a channel-specific fit.
    common_beta = med(rows, "beta")
    for row in rows:
        row["null_alpha_common_beta"] = common_beta * row["geometric_factor"]
        row["alpha_residual_common_beta"] = row["alpha"] - row["null_alpha_common_beta"]
    with (HERE / "threshold_aware_exponent_cells.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    groups = {c: [r for r in rows if r["channel"] == c] for c in (*CHANNELS, "chimA")}
    metrics = ("alpha", "beta", "span", "threshold_fraction", "geometric_factor",
               "null_alpha_common_beta", "alpha_residual_common_beta",
               "matched_early_010", "matched_late_010", "alpha_r2", "shifted_r2")
    summary = {c: {"cells": len(g), "dates": len({r["date"] for r in g}),
                   **{k + "_median": med(g, k) for k in metrics},
                   "matched_010_n": int(sum(np.isfinite(r["matched_early_010"]) for r in g))}
               for c, g in groups.items()}
    differences = {k: bootstrap_median_diff(rows, k) for k in metrics}
    date_cluster_differences = {k: date_cluster_bootstrap(rows, k)
                                for k in ("alpha", "beta", "geometric_factor",
                                          "matched_early_010", "matched_late_010")}
    wt = [r for r in rows if r["channel"] in CHANNELS]
    corr = spearmanr([r["span"] for r in wt], [r["alpha"] for r in wt])

    # Null calibration: empirical cell-specific amplitude and prethreshold
    # noise; one shared beta, observed work arrays and observed thresholds.
    # This is a sensitivity check, not a channel-independent biological truth.
    null_by_beta = {}
    for beta0 in (1.0, 1.5, 2.0, 2.5, 3.0, common_beta):
        modeled = {c: np.asarray([beta0 * r["geometric_factor"] for r in groups[c]])
                   for c in CHANNELS}
        null_by_beta[f"{beta0:.3f}"] = {
            "median_alpha": {c: float(np.median(v)) for c, v in modeled.items()},
            "p1_minus_p2": float(np.median(modeled["mp1"]) - np.median(modeled["mp2"])),
            "four_channel_order": bool(all(np.median(modeled[a]) > np.median(modeled[b])
                                           for a, b in zip(CHANNELS[:-1], CHANNELS[1:]))),
        }
    # Explicit noisy simulations on the recorded trajectories. The published
    # noise field is in pA, as are the analyzed current values. Calibration of
    # each synthetic trace to its observed postthreshold maximum controls SNR.
    noisy_null = {}
    for noise_multiplier in (0.0, 1.0, 2.0):
        replicate_differences = []
        replicate_rankings = []
        replicate_medians = defaultdict(list)
        for _ in range(300):
            alphas = defaultdict(list)
            for meta, w, i in traces:
                if float(meta["responds"]) != 1:
                    continue
                w0 = float(meta["wthresh"])
                if not np.isfinite(w0) or np.max(w) <= w0:
                    continue
                positive = np.maximum(w - w0, 0)
                top = np.max(positive) ** common_beta
                if top <= 0:
                    continue
                observed = i[np.isfinite(i) & (w > w0)]
                if not len(observed):
                    continue
                amplitude = np.max(observed) / top
                noise_sd = abs(float(meta["stdev"])) * noise_multiplier
                simulated = amplitude * positive ** common_beta + RNG.normal(0, noise_sd, len(w))
                fit = fit_pair(w, simulated, w0)
                if fit is not None:
                    alphas[meta["construct"]].append(fit["alpha"])
            if any(not alphas[c] for c in CHANNELS):
                continue
            medians = {c: float(np.median(alphas[c])) for c in CHANNELS}
            for c, val in medians.items():
                replicate_medians[c].append(val)
            replicate_differences.append(medians["mp1"] - medians["mp2"])
            replicate_rankings.append(all(medians[a] > medians[b]
                                          for a, b in zip(CHANNELS[:-1], CHANNELS[1:])))
        noisy_null[str(noise_multiplier)] = {
            "p1_minus_p2_median_and_95pct": list(map(float, np.quantile(replicate_differences,
                                                                    [.5, .025, .975]))),
            "four_channel_order_fraction": float(np.mean(replicate_rankings)),
            "channel_median_alpha": {c: float(np.median(v)) for c, v in replicate_medians.items()},
            "replicates": len(replicate_differences),
        }
    # Sensitivity to uncertain onset threshold. Values near W0 create a strong
    # singularity; test small multiplicative shifts without refitting W0.
    threshold_sensitivity = {}
    for factor in (0.8, 0.9, 1.0, 1.1):
        vals = defaultdict(list)
        for meta, w, i in traces:
            if float(meta["responds"]) != 1:
                continue
            fit = fit_pair(w, i, factor * float(meta["wthresh"]))
            if fit is not None and np.isfinite(fit["beta"]):
                vals[meta["construct"]].append(fit["beta"])
        threshold_sensitivity[str(factor)] = {c: float(np.median(vals[c])) if vals[c] else None
                                               for c in CHANNELS}

    report = dict(cohort="one selected responding sweep per uniqueID; original current and work fields",
                  common_beta=common_beta, summary=summary, p1_minus_p2=differences,
                  p1_minus_p2_date_cluster=date_cluster_differences,
                  shifted_r2_better_fraction=float(np.mean([r["shifted_r2"] > r["alpha_r2"]
                                                            for r in wt])),
                  span_alpha_spearman={"rho": float(corr.statistic), "p_naive_cell": float(corr.pvalue)},
                  null_by_beta=null_by_beta, noisy_null=noisy_null,
                  threshold_sensitivity=threshold_sensitivity,
                  cautions=["Channel identity and acquisition dates may be confounded.",
                            "This public selected-sweep cohort is smaller than the paper's reported fit n.",
                            "Wth is estimated from current; shifted-beta inference inherits threshold uncertainty.",
                            "The noisy null uses paper-reported baseline SD and fixed observed thresholds, but does not rerun threshold detection."])
    (HERE / "threshold_aware_exponent_results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    colors = {"mp1": "#3967a2", "mp2": "#d6812f", "mtraak": "#5c9470",
              "mtrek1": "#9a6b92", "chimA": "#7763a3"}
    for c in (*CHANNELS, "chimA"):
        g = groups[c]
        ax[0].scatter([r["span"] for r in g], [r["alpha"] for r in g],
                      label=c, c=colors[c], alpha=.75, s=28)
    ax[0].set(xlabel="log10 work span", ylabel="apparent alpha")
    ax[0].legend(fontsize=7, ncol=2)
    for c in CHANNELS:
        g = groups[c]
        ax[1].scatter([r["null_alpha_common_beta"] for r in g], [r["alpha"] for r in g],
                      c=colors[c], alpha=.75, s=28)
    ax[1].plot([0, 12], [0, 12], "k--", lw=.8)
    ax[1].set(xlabel="geometry-only alpha (shared beta)", ylabel="observed alpha")
    for j, c in enumerate(CHANNELS):
        g = groups[c]
        ax[2].scatter(np.full(len(g), j) - .1, [r["alpha"] for r in g],
                      color=colors[c], alpha=.45, s=16)
        ax[2].scatter(np.full(len(g), j) + .1, [r["beta"] for r in g],
                      color=colors[c], alpha=.7, s=16, marker="x")
    ax[2].set_xticks(range(4), ("P1", "P2", "TRAAK", "TREK1"))
    ax[2].set(ylabel="slope (dot: alpha, x: shifted beta)")
    fig.savefig(HERE / "threshold_aware_exponent_audit.png", dpi=180)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
