"""Can a calibrated spatial mechanics rule forecast individual sarcomeres?

The future LOI-average trajectory is deliberately supplied to every method.
This isolates the *spatial decoder*; it is not an end-to-end cell forecast.
The six independent cells are tested by leaving out whole stiffness groups.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIRST, HORIZON, STRIDE = 300, 10, 10
LOADS = (5, 15, 85)


def track_arrays(data, prefix):
    raw = np.asarray(data[f"{prefix}_sarcomere_length_um"], float)
    mean_raw = np.asarray(data[f"{prefix}_length_um"], float)
    sb = np.nanmedian(raw[:, :FIRST], axis=1)
    mb = float(np.nanmedian(mean_raw[:FIRST]))
    valid = np.isfinite(sb) & (sb > 0)
    x = raw[valid] / sb[valid, None]
    m = mean_raw / mb
    # Early recording is the only test-cell information used to personalize
    # each sarcomere's mechanical response and baseline offset.
    beta = np.ones(len(x))
    for i, row in enumerate(x):
        ok = np.isfinite(row[:FIRST]) & np.isfinite(m[:FIRST])
        if ok.sum() < 60:
            continue
        mm, xx = m[:FIRST][ok], row[:FIRST][ok]
        den = np.sum((mm - mm.mean()) ** 2)
        if den > 1e-8:
            beta[i] = np.sum((mm - mm.mean()) * (xx - xx.mean())) / den
    # Clipping was fixed before the held-out evaluation: extreme slopes from
    # weak early contractions should not determine a future trajectory.
    beta = np.clip(beta, -1.0, 3.0)
    r0 = np.nanmedian(x[:, :FIRST] - m[None, :FIRST], axis=1)
    return x, m, beta, r0


def laplace(field):
    left = np.r_[field[0], field[:-1]]
    right = np.r_[field[1:], field[-1]]
    return left + right - 2 * field


def collect():
    data = np.load(HERE / "sarcomere_tracks.npz")
    tracks = {}
    for prefix in sorted(k.removesuffix("_length_um") for k in data.files
                         if k.endswith("_length_um") and "_sarcomere_" not in k):
        load = int(prefix.split("_")[0].removeprefix("load"))
        cell = prefix.split("_loi")[0]
        x, m, beta, r0 = track_arrays(data, prefix)
        windows = []
        for t in range(FIRST, len(m) - HORIZON - 1, STRIDE):
            future_m = m[t:t + HORIZON + 1]
            valid = (np.isfinite(x[:, t-1:t+HORIZON+1]).all(axis=1)
                     & np.isfinite(beta) & np.isfinite(r0))
            if valid.sum() < 8 or not np.isfinite(future_m).all():
                continue
            windows.append((x[valid, t-1:t+HORIZON+1], future_m,
                            beta[valid], r0[valid]))
        if windows:
            tracks.setdefault(load, {}).setdefault(cell, []).append((prefix, windows))
    return tracks


def predict(win, model, coeff=(0, 0, 0)):
    x, m, beta, r0 = win
    if model.startswith("shuffled"):
        seed = int(model.removeprefix("shuffled"))
        order = np.random.default_rng(1057 + 31 * len(x) + seed).permutation(len(x))
        permuted = (x[order], m, beta[order], r0[order])
        out = predict(permuted, "coupled", coeff)
        return out[np.argsort(order)]
    current = x[:, 1].copy()
    out = []
    c, a, gamma = coeff
    for h in range(HORIZON):
        dm = m[h+1] - m[h]
        if model == "persistence":
            nxt = current.copy()
        elif model == "uniform":
            nxt = current + dm
        elif model == "gain":
            nxt = current + dm * (1 + gamma * (beta - 1))
        elif model in ("coupled", "independent"):
            r = current - m[h]
            nxt = (current + dm * (1 + gamma * (beta - 1))
                   + c * laplace(r) + a * (r0-r))
        else:
            raise ValueError(model)
        current = nxt
        out.append(nxt)
    return np.stack(out, axis=1)


def cell_scores(cell_lois, model, coeff):
    # LOIs are correlated. They are averaged within a cell first.
    loivalues = []
    localonly = []
    spatial_r2 = []
    example = None
    n_windows = 0
    for prefix, windows in cell_lois:
        errs, residual_errs, numerator, denominator = [], [], 0.0, 0.0
        for win in windows:
            x, m, _, _ = win
            pred = predict(win, model, coeff)
            actual = x[:, 2:]
            errs.append(np.mean(np.abs(pred-actual)))
            # Removes observed global motion from the score, so an apparently
            # good result cannot come solely from being given future LOI mean.
            pred_spatial = pred-pred.mean(axis=0, keepdims=True)
            true_spatial = actual-actual.mean(axis=0, keepdims=True)
            residual_errs.append(np.mean(np.abs(pred_spatial-true_spatial)))
            numerator += float(np.sum((pred_spatial-true_spatial)**2))
            denominator += float(np.sum(true_spatial**2))
            if example is None and np.std(actual[:, -1]) > 0.005:
                example = (prefix, x[:, 1].copy(), actual[:, -1].copy(),
                           pred[:, -1].copy(), float(m[-1]))
        if errs:
            loivalues.append(float(np.mean(errs)))
            localonly.append(float(np.mean(residual_errs)))
            spatial_r2.append(float(1-numerator/max(denominator, 1e-12)))
            n_windows += len(errs)
    return {"mae": float(np.mean(loivalues)),
            "local_residual_mae": float(np.mean(localonly)),
            "spatial_pattern_r2": float(np.mean(spatial_r2)),
            "n_windows": n_windows, "n_lois": len(loivalues)}, example


def score_training(tracks, held_load, coeff, model="coupled"):
    return np.mean([cell_scores(lois, model, coeff)[0]["mae"]
                    for load, cells in tracks.items() if load != held_load
                    for lois in cells.values()])


def main():
    tracks = collect()
    if any(len(tracks.get(load, {})) != 2 for load in LOADS):
        raise RuntimeError("Expected exactly two cells at each of three stiffnesses")
    # Small, physically stable coefficients. No test-cell future response is
    # used for tuning. The grid includes the uniform (0,0,0) null model.
    grid = [(c, a, g) for c in (0, .03, .10)
            for a in (0, .01, .04) for g in (0, .25, .5, 1.)]
    # A local-gain decoder is also tuned independently on the training cells.
    gain_grid = (0, .25, .5, 1.)
    result = {"method": "whole-stiffness held-out spatial decoder",
              "n_independent_cells": 6, "future_global_LOI_trace": "given to all models (oracle)",
              "calibration": "first 300 frames per LOI, approximately 4.9 s",
              "horizon_frames": HORIZON, "stride_frames": STRIDE,
              "per_cell": {}, "selected_coefficients": {}}
    examples = {}
    for held in LOADS:
        best_c = min(grid, key=lambda x: score_training(tracks, held, x))
        best_ind = min((v for v in grid if v[0] == 0),
                       key=lambda x: score_training(tracks, held, x, "independent"))
        # The randomized-neighbour control gets its *own* coefficient choice
        # using only training cells; otherwise the adjacency comparison is unfair.
        shuffled_grid = [(c, best_ind[1], best_ind[2]) for c in (0, .03, .10)]
        best_shuffled = min(shuffled_grid, key=lambda coeff: np.mean([
            score_training(tracks, held, coeff, f"shuffled{j}")
            for j in range(5)]))
        best_g = min(gain_grid, key=lambda g: np.mean([
            cell_scores(lois, "gain", (0, 0, g))[0]["mae"]
            for load, cells in tracks.items() if load != held
            for lois in cells.values()]))
        result["selected_coefficients"][str(held)] = {
            "coupled": list(best_c), "independent": list(best_ind),
            "shuffled_topology": list(best_shuffled), "gain": best_g}
        for cell, lois in tracks[held].items():
            result["per_cell"][cell] = {"load_kpa": held}
            for model, coeff in (("persistence", (0, 0, 0)),
                                 ("uniform", (0, 0, 0)),
                                 ("gain", (0, 0, best_g)),
                                 ("independent", best_ind),
                                 ("coupled", best_c)):
                score, ex = cell_scores(lois, model, coeff)
                result["per_cell"][cell][model] = score
                if ex is not None and model in ("uniform", "coupled"):
                    examples[(cell, model)] = ex
            nulls = [cell_scores(lois, f"shuffled{j}", best_shuffled)[0]
                     for j in range(5)]
            result["per_cell"][cell]["shuffled_topology"] = {
                key: float(np.mean([score[key] for score in nulls]))
                for key in ("mae", "local_residual_mae", "spatial_pattern_r2",
                            "n_windows", "n_lois")}
    models = ("persistence", "uniform", "gain", "independent",
              "coupled", "shuffled_topology")
    cells = list(result["per_cell"])
    result["summary"] = {model: {
        "cell_macro_mae": float(np.mean([result["per_cell"][cell][model]["mae"]
                                          for cell in cells])),
        "cell_macro_spatial_pattern_r2": float(np.mean([
            result["per_cell"][cell][model]["spatial_pattern_r2"]
            for cell in cells])),
        "cells_better_than_uniform": int(sum(
            result["per_cell"][cell][model]["mae"] <
            result["per_cell"][cell]["uniform"]["mae"] for cell in cells)),
    } for model in models}
    rng = np.random.default_rng(20261018)
    result["paired_cell_contrasts"] = {}
    for model in ("gain", "independent", "coupled", "shuffled_topology"):
        gains = np.asarray([result["per_cell"][cell]["uniform"]["mae"]-
                            result["per_cell"][cell][model]["mae"] for cell in cells])
        boot = np.mean(gains[rng.integers(0, 6, size=(10000, 6))], axis=1)
        result["paired_cell_contrasts"][f"{model}_vs_uniform"] = {
            "mean_absolute_gain": float(gains.mean()),
            "cell_bootstrap_95": np.quantile(boot, [.025, .975]).tolist(),
            "per_cell_gain": dict(zip(cells, gains.tolist()))}
    (HERE / "spatial_decoder_gate.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for model, color in (("gain", "#4E8EAF"), ("independent", "#8174AA"),
                         ("coupled", "#C8654D"),
                         ("shuffled_topology", "#ACAF4E")):
        base = [result["per_cell"][k]["uniform"]["mae"] for k in cells]
        new = [result["per_cell"][k][model]["mae"] for k in cells]
        axs[0].scatter(base, new, label=model, c=color, s=48)
    lim = max(axs[0].get_xlim()[1], axs[0].get_ylim()[1])
    axs[0].plot([0, lim], [0, lim], c="#555", lw=1)
    axs[0].set(xlabel="Uniform scaling MAE", ylabel="Spatial decoder MAE",
               title="Each point = one held-out cell")
    axs[0].legend(frameon=False)
    cell = max(cells, key=lambda k: result["per_cell"][k]["uniform"]["mae"])
    ex_u, ex_c = examples[(cell, "uniform")], examples[(cell, "coupled")]
    n = len(ex_u[1])
    axs[1].plot(np.arange(n), ex_u[2], "o-", ms=3, label="observed")
    axs[1].plot(np.arange(n), ex_u[3], "--", label="uniform")
    axs[1].plot(np.arange(n), ex_c[3], "-.", label="coupled")
    axs[1].set(xlabel="Sarcomere along selected LOI",
               ylabel="Length / early baseline", title=f"Example: {cell}")
    axs[1].legend(frameon=False)
    fig.savefig(HERE / "spatial_decoder_gate.png", dpi=180)
    print(json.dumps({"summary": result["summary"],
                      "contrasts": result["paired_cell_contrasts"],
                      "coefficients": result["selected_coefficients"]}, indent=2))


if __name__ == "__main__":
    main()
