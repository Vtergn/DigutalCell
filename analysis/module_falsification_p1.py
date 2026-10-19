"""Small falsification tests for external PIEZO1 mechanical and history modules.

Mechanical test predicts held-out AFM loading speeds from other speeds in the
same measured cells. Source files contain fitted relaxation descriptors, not
raw force trajectories, so this does not validate an ODE waveform.

History test predicts one held-out patch's pulses 2-6 using other patches at
the *same observed interval*. Interval order and cross-interval patch identity
are not available; no continuous recovery rate is estimated or extrapolated.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import lsq_linear
from scipy.special import expit, logit


HERE = Path(__file__).resolve().parent


def read_csv(name: str) -> list[dict]:
    with (HERE / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(name: str, rows: list[dict]) -> None:
    with (HERE / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def held_speed_predictions(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    observations = [{"cell": r["cell_label"], "speed": float(r["speed_um_per_ms"]),
                     "tau": float(r["force_relaxation_tau_ms"]),
                     "retention": float(r["force_remaining_pct"])} for r in rows]
    predictions, fold_params = [], []
    for held_speed in sorted({r["speed"] for r in observations}):
        train = [r for r in observations if r["speed"] != held_speed]
        test = [r for r in observations if r["speed"] == held_speed]
        for endpoint in ("tau", "retention"):
            # Shared speed slope, separate intercept per previously observed
            # cell. Fit in log tau / logit retained-force fraction domains.
            def transform(value: float) -> float:
                return (float(np.log(value)) if endpoint == "tau" else
                        float(logit(np.clip(value / 100., 1e-4, 1-1e-4))))

            by_cell = defaultdict(list)
            for row in train:
                by_cell[row["cell"]].append(row)
            means = {cell: (float(np.mean([np.log(r["speed"]) for r in cell_rows])),
                            float(np.mean([transform(r[endpoint]) for r in cell_rows])))
                     for cell, cell_rows in by_cell.items()}
            numerator = sum((np.log(r["speed"])-means[r["cell"]][0]) *
                            (transform(r[endpoint])-means[r["cell"]][1]) for r in train)
            denominator = sum((np.log(r["speed"])-means[r["cell"]][0])**2 for r in train)
            slope = float(numerator / denominator)
            fold_params.append({"held_speed_um_per_ms": held_speed,
                                "endpoint": endpoint, "shared_log_speed_slope": slope,
                                "training_cells": len(by_cell)})
            for row in test:
                baseline_z = means[row["cell"]][1]
                model_z = baseline_z + slope*(np.log(held_speed)-means[row["cell"]][0])
                inverse = (np.exp if endpoint == "tau" else
                           lambda z: 100.*expit(z))
                actual = row[endpoint]
                for model, z in (("cell_constant", baseline_z),
                                 ("shared_speed_slope", model_z)):
                    predicted = float(inverse(z))
                    predictions.append({"held_speed_um_per_ms": held_speed,
                                        "cell": row["cell"], "endpoint": endpoint,
                                        "model": model, "actual": actual,
                                        "predicted": predicted,
                                        "absolute_error": abs(actual-predicted),
                                        "absolute_log_error":
                                            abs(float(np.log(actual/predicted))) if endpoint == "tau" else ""})
    return predictions, fold_params


def sequences(rows: list[dict]) -> dict[tuple[int, int], np.ndarray]:
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(int(row["interval_ms"]), int(row["workbook_column"]))][
            int(row["pulse_number"])] = float(row["ratio_to_first"])
    return {key: np.array([values.get(pulse, np.nan) for pulse in range(1, 7)])
            for key, values in grouped.items()}


def fit_state(train: list[np.ndarray]) -> tuple[float, float]:
    x, y = [], []
    for trace in train:
        for before, after in zip(trace[:-1], trace[1:]):
            if np.isfinite(before) and np.isfinite(after):
                x.append([before, 1.])
                y.append(after)
    fit = lsq_linear(np.asarray(x), np.asarray(y), bounds=(0., 1.))
    return float(fit.x[0]), float(fit.x[1])


def simulate_state(a: float, b: float) -> np.ndarray:
    result = [1.]
    for _ in range(5):
        result.append(a*result[-1]+b)
    return np.asarray(result)


def held_patch_predictions(rows: list[dict], drop_near_zero: bool = False) -> tuple[list[dict], list[dict]]:
    traces = sequences(rows)
    if drop_near_zero:
        traces = {key: value for key, value in traces.items()
                  if not np.any((value[1:] < 1e-6) & np.isfinite(value[1:]))}
    scores, fold_params = [], []
    for interval in sorted({key[0] for key in traces}):
        keys = [key for key in traces if key[0] == interval]
        for held in keys:
            train = [traces[key] for key in keys if key != held]
            a, b = fit_state(train)
            constant = float(np.nanmedian([x for trace in train for x in trace[1:]
                                           if np.isfinite(x)]))
            template = np.array([1.] + [float(np.nanmedian([trace[p] for trace in train]))
                                          for p in range(1, 6)])
            predictions = {"memoryless": np.ones(6),
                           "postpulse_constant": np.r_[1., np.repeat(constant, 5)],
                           "pulse_template": template,
                           "one_state": simulate_state(a, b)}
            fold_params.append({"interval_ms": interval, "held_source_column": held[1],
                                "training_patches": len(train), "a": a, "b": b,
                                "state_fixed_point_if_a_lt_1":
                                    b/(1-a) if a < .999999 else "",
                                "near_zero_excluded_sensitivity": int(drop_near_zero)})
            target = traces[held]
            for model, prediction in predictions.items():
                mask = np.isfinite(target[1:])
                errors = np.abs(target[1:][mask]-prediction[1:][mask])
                scores.append({"interval_ms": interval, "held_source_column": held[1],
                               "model": model, "observed_later_pulses": int(mask.sum()),
                               "mae_ratio": float(np.mean(errors)),
                               "near_zero_excluded_sensitivity": int(drop_near_zero)})
    return scores, fold_params


def summarize(scores: list[dict], key: str) -> dict:
    out = {}
    for label in sorted({r[key] for r in scores}):
        selected = [r for r in scores if r[key] == label]
        out[str(label)] = {model: {"n": len([r for r in selected if r["model"] == model]),
                                  "median_cell_mae": float(np.median([r["mae_ratio"] for r in selected
                                                                       if r["model"] == model])),
                                  "mean_cell_mae": float(np.mean([r["mae_ratio"] for r in selected
                                                                   if r["model"] == model]))}
                           for model in ("memoryless", "postpulse_constant",
                                         "pulse_template", "one_state")}
    return out


def main() -> None:
    speed_scores, speed_fits = held_speed_predictions(read_csv("wedge_afm_velocity.csv"))
    history = read_csv("wedge_recovery_wt.csv")
    state_scores, state_fits = held_patch_predictions(history)
    sensitivity_scores, sensitivity_fits = held_patch_predictions(history, True)
    write_csv("module_p1_speed_holdout_scores.csv", speed_scores)
    write_csv("module_p1_speed_holdout_fits.csv", speed_fits)
    write_csv("module_p1_history_holdout_scores.csv", state_scores+sensitivity_scores)
    write_csv("module_p1_history_holdout_fits.csv", state_fits+sensitivity_fits)
    speed_summary = {}
    for endpoint in ("tau", "retention"):
        speed_summary[endpoint] = {model: {
            "n": len([r for r in speed_scores if r["endpoint"] == endpoint and r["model"] == model]),
            "mean_absolute_error": float(np.mean([r["absolute_error"] for r in speed_scores
                                                    if r["endpoint"] == endpoint and r["model"] == model])),
            "median_absolute_error": float(np.median([r["absolute_error"] for r in speed_scores
                                                      if r["endpoint"] == endpoint and r["model"] == model])),
            "median_absolute_log_error": (float(np.median([r["absolute_log_error"] for r in speed_scores
                                                           if r["endpoint"] == endpoint and r["model"] == model]))
                                          if endpoint == "tau" else None)}
            for model in ("cell_constant", "shared_speed_slope")}
    history_summary = {"all": summarize(state_scores, "interval_ms"),
                       "exclude_single_near_zero_trace": summarize(sensitivity_scores, "interval_ms")}
    results = {"speed_holdout": speed_summary, "history_holdout": history_summary,
               "history_full_fit": {str(interval): dict(zip(("a", "b"), fit_state([
                   trace for (dt, _), trace in sequences(history).items() if dt == interval])))
                   for interval in sorted({int(r["interval_ms"]) for r in history})}}
    (HERE / "module_p1_falsification_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, endpoint, ylabel in ((axes[0], "tau", "Held-speed τ absolute log error"),
                                 (axes[1], "retention", "Held-speed retention error (pp)")):
        for model, color in (("cell_constant", "#ad7038"),
                             ("shared_speed_slope", "#2077a4")):
            ys = [float(np.median([r["absolute_log_error"] if endpoint == "tau" else
                                   r["absolute_error"] for r in speed_scores if
                                   r["endpoint"] == endpoint and r["model"] == model and
                                   r["held_speed_um_per_ms"] == speed]))
                  for speed in sorted({r["held_speed_um_per_ms"] for r in speed_scores})]
            xs = sorted({r["held_speed_um_per_ms"] for r in speed_scores})
            ax.plot(xs, ys, marker="o", color=color, label=model.replace("_", " "))
        ax.set(xscale="log", xlabel="Held-out loading speed (μm/ms)", ylabel=ylabel)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "module_p1_speed_holdout.png", dpi=180)
    plt.close(fig)
    print(json.dumps(results["speed_holdout"], indent=2))
    print("history full-fit", results["history_full_fit"])
    for interval, group in history_summary["all"].items():
        print("history",interval,{model: round(group[model]["median_cell_mae"],3)
                                  for model in group})


if __name__ == "__main__":
    main()
