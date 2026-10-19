"""Prospective 1/2/3-shot peak-amplitude study on the frozen Duke signed cohort.

Effective force parameters are phenomenological. Each held-out date is excluded
from population-shape selection. Test-cell sweeps are revealed in chronological
order; a target sweep is never used in its own calibration.
"""

from __future__ import annotations

import csv
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

from dynamic_input_screen import HERE
from mechanistic_tension_simulator import assemble_records, observed_peak


GAMMAS = np.round(np.arange(0.25, 4.001, 0.25), 3)
F50_OFFSETS_NN = np.arange(-200.0, 601.0, 25.0)
SF_GRID_NN = np.array([25., 50., 75., 100., 150., 200., 300., 450.])
MODELS = ("linear", "power", "sigmoid")


def drive(force: float, threshold: float, model: str, gamma: float = 1.,
          f50_offset: float = 0., sf: float = 100.) -> float:
    if model in ("linear", "power"):
        x = max(force - threshold, 0.)
        return x if model == "linear" else x ** gamma
    f50 = threshold + f50_offset
    return max(float(expit((force - f50) / sf) -
                     expit((threshold - f50) / sf)), 0.)


def ratio(force: float, cal_force: float, threshold: float, model: str,
          gamma: float = 1., f50_offset: float = 0., sf: float = 100.) -> float:
    den = drive(cal_force, threshold, model, gamma, f50_offset, sf)
    if den < 1e-12:
        return 0.
    return drive(force, threshold, model, gamma, f50_offset, sf) / den


def peak_force(row: dict) -> float:
    return float(np.max(row["force"][:150]))


def groups_from_records() -> dict:
    groups = defaultdict(list)
    for row in assemble_records():
        groups[(row["channel"], row["cell_id"], row["date"])].append(row)
    for rows in groups.values():
        rows.sort(key=lambda r: int(r["sweep"]))
    return dict(groups)


def shots(rows: list[dict], k: int) -> list[tuple[float, float]]:
    first = [(float(rows[0]["cal_force_peak"]), float(rows[0]["cal_peak"]))]
    return first + [(peak_force(r), observed_peak(r)) for r in rows[:k-1]]


def population_shape(training_groups: list[list[dict]], model: str) -> dict:
    if model == "linear":
        return {}
    candidates = ([{"gamma": float(g)} for g in GAMMAS] if model == "power" else
                  [{"f50_offset": float(f), "sf": float(s)}
                   for s in SF_GRID_NN for f in F50_OFFSETS_NN])
    best = None
    for params in candidates:
        losses = []
        for rows in training_groups:
            threshold = float(rows[0]["threshold"])
            cal_force, cal_peak = shots(rows, 1)[0]
            loss = np.mean([
                abs(np.log1p(cal_peak * ratio(peak_force(row), cal_force,
                                                threshold, model, **params)) -
                    np.log1p(observed_peak(row)))
                for row in rows
            ])
            losses.append(float(loss))
        key = (float(np.mean(losses)), tuple(params.values()))
        if best is None or key < best[0]:
            best = (key, params)
    return best[1]


def fit_cell(rows: list[dict], k: int, model: str, population: dict) -> dict:
    obs = shots(rows, k)
    threshold = float(rows[0]["threshold"])
    cal_force = obs[0][0]
    # A is fitted in log amplitude. Zero/low-SNR observations remain in the
    # chronological sequence, with a 1 pA floor only for the log transform.
    candidates = ([{}] if model == "linear" else
                  ([{"gamma": float(g)} for g in GAMMAS] if model == "power" else
                   [{"f50_offset": float(f), "sf": population["sf"]}
                    for f in F50_OFFSETS_NN]))
    if k == 1:
        candidates = [population]
    best = None
    for params in candidates:
        xs = np.array([ratio(f, cal_force, threshold, model, **params)
                       for f, _ in obs])
        if np.any(xs <= 1e-9):
            continue
        ys = np.array([max(float(y), 1.) for _, y in obs])
        log_gain = float(np.mean(np.log(ys) - np.log(xs)))
        residual = np.log(ys) - (log_gain + np.log(xs))
        loss = float(np.mean(residual ** 2))
        # Equal-loss ties prefer the population shape. This is a tie breaker,
        # not regularization capable of making an underdetermined fit unique.
        distance = (abs(params.get("gamma", 0.) - population.get("gamma", 0.)) if
                    model == "power" else
                    abs(params.get("f50_offset", 0.) -
                        population.get("f50_offset", 0.)))
        key = (round(loss, 12), distance)
        if best is None or key < best[0]:
            best = (key, {"gain_pA": float(np.exp(log_gain)), **params,
                          "calibration_log_sse": loss})
    if best is None:
        raise ValueError(f"No valid shape for {rows[0]['cell_id']} k={k}")
    return best[1]


def predict_peak(force: float, rows: list[dict], model: str, fitted: dict) -> float:
    threshold = float(rows[0]["threshold"])
    cal_force = float(rows[0]["cal_force_peak"])
    params = {key: fitted[key] for key in ("gamma", "f50_offset", "sf")
              if key in fitted}
    return fitted["gain_pA"] * ratio(force, cal_force, threshold, model, **params)


def profile(rows: list[dict], k: int) -> list[dict]:
    obs = shots(rows, k)
    threshold = float(rows[0]["threshold"])
    cal_force = obs[0][0]
    output = []
    for f in F50_OFFSETS_NN:
        options = []
        for sf in SF_GRID_NN:
            xs = np.array([ratio(force, cal_force, threshold, "sigmoid",
                                 f50_offset=float(f), sf=float(sf)) for force, _ in obs])
            if np.any(xs <= 1e-9):
                continue
            logs = np.log(np.array([max(y, 1.) for _, y in obs])) - np.log(xs)
            options.append((float(np.mean((logs - np.mean(logs)) ** 2)), float(sf)))
        loss, sf = min(options)
        output.append({"channel": rows[0]["channel"], "cell_id": rows[0]["cell_id"],
                       "k_shot": k, "f50_offset_nN": float(f),
                       "F50_eff_nN": threshold + float(f),
                       "profile_log_sse": loss, "profile_sf_nN": sf})
    return output


def write_csv(name: str, rows: list[dict]) -> None:
    with (HERE / name).open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    groups = groups_from_records()
    folds, predictions, cell_fits, profiles = [], [], [], []
    for channel in ("mp1", "mp2"):
        dates = sorted({date for ch, _, date in groups if ch == channel})
        for date in dates:
            training = [rows for (ch, _, dt), rows in groups.items()
                        if ch == channel and dt != date]
            test = [rows for (ch, _, dt), rows in groups.items()
                    if ch == channel and dt == date]
            population = {m: population_shape(training, m) for m in MODELS}
            for model in MODELS:
                folds.append({"channel": channel, "heldout_date": date,
                              "model": model, "train_cells": len(training),
                              "test_cells": len(test), **population[model]})
            for rows in test:
                for k in (1, 2, 3):
                    if len(rows) < k:  # Need at least one strictly later target.
                        continue
                    for model in MODELS:
                        fitted = fit_cell(rows, k, model, population[model])
                        cell_fits.append({"channel": channel, "heldout_date": date,
                                          "cell_id": rows[0]["cell_id"], "k_shot": k,
                                          "model": model, "calibration_sweeps":
                                              ";".join(str(r["sweep"]) for r in rows[:k-1]),
                                          **fitted})
                        for target_index in range(k-1, len(rows)):
                            row = rows[target_index]
                            actual = observed_peak(row)
                            predicted = predict_peak(peak_force(row), rows, model, fitted)
                            predictions.append({"channel": channel,
                                                "heldout_date": date,
                                                "cell_id": row["cell_id"],
                                                "k_shot": k, "model": model,
                                                "target_sweep": row["sweep"],
                                                "target_future_index": target_index+1,
                                                "common_3shot_target": int(target_index >= 2),
                                                "target_Fpeak_nN": peak_force(row),
                                                "true_peak_pA": actual,
                                                "pred_peak_pA": predicted,
                                                "abs_peak_error_pA": abs(actual-predicted),
                                                "abs_log_peak_error":
                                                    abs(np.log1p(actual)-np.log1p(predicted))})
    # One Piezo1 cell with a long sequence and one Piezo2 cell with broad
    # observed force steps; these are illustrative, not selected by score.
    for cell in ("20191221-1-scan-80", "20190816-2-scan-80"):
        rows = next(v for (_, cid, _), v in groups.items() if cid == cell)
        for k in (1, 2, 3):
            profiles.extend(profile(rows, k))
    write_csv("effective_force_kshot_folds.csv", folds)
    write_csv("effective_force_kshot_cell_fits.csv", cell_fits)
    write_csv("effective_force_kshot_predictions.csv", predictions)
    write_csv("effective_force_kshot_profiles.csv", profiles)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax, cell in zip(axes, ("20191221-1-scan-80", "20190816-2-scan-80")):
        for k in (1, 2, 3):
            selected = [r for r in profiles if r["cell_id"] == cell and r["k_shot"] == k]
            ax.plot([r["F50_eff_nN"] for r in selected],
                    [r["profile_log_sse"] for r in selected], label=f"{k}-shot")
        ax.set_title(cell)
        ax.set_xlabel(r"Effective $F_{50}$ (nN)")
        ax.set_ylabel("Profile mean squared log peak residual")
        ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / "effective_force_kshot_profiles.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    palette = {"linear": "#2369a8", "power": "#db8734", "sigmoid": "#2a9869"}
    for ax, channel in zip(axes, ("mp1", "mp2")):
        for model in MODELS:
            medians = []
            for k in (1, 2, 3):
                matching = [r for r in predictions if r["channel"] == channel and
                            r["model"] == model and r["k_shot"] == k and
                            r["common_3shot_target"]]
                cells = sorted({r["cell_id"] for r in matching})
                cell_losses = [np.mean([r["abs_log_peak_error"] for r in matching
                                        if r["cell_id"] == cell]) for cell in cells]
                medians.append(float(np.median(cell_losses)))
                ax.scatter([k] * len(cell_losses), cell_losses, s=14,
                           color=palette[model], alpha=.32)
            ax.plot((1, 2, 3), medians, marker="o", color=palette[model],
                    label=model)
        n_cells = len({r["cell_id"] for r in predictions if
                       r["channel"] == channel and r["k_shot"] == 3})
        ax.set(title=f"{'Piezo1' if channel == 'mp1' else 'Piezo2'}: {n_cells} cells",
               xlabel="Past responsive sweeps used")
        ax.set_xticks((1, 2, 3))
        ax.grid(alpha=.2)
        ax.legend()
    axes[0].set_ylabel("Median cell absolute log peak error\n(same future targets at every shot count)")
    fig.tight_layout()
    fig.savefig(HERE / "effective_force_kshot_learning_curve.png", dpi=180)
    plt.close(fig)
    print("groups", len(groups), "future predictions", len(predictions))
    for channel in ("mp1", "mp2"):
        for k in (1, 2, 3):
            subset = [r for r in predictions if r["channel"] == channel and
                      r["k_shot"] == k and r["model"] == "linear"]
            common = [r for r in subset if r["common_3shot_target"]]
            print(channel, k, "cells", len({r["cell_id"] for r in subset}),
                  "targets", len(subset), "common cells",len({r["cell_id"] for r in common}),
                  "common targets",len(common))


if __name__ == "__main__":
    main()
