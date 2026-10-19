"""Paired, split-safe inverse and unseen-load benchmark for Strocchi Zenodo data.

The released tables provide tension *features* for three load conditions, not
time traces or paired AP/Ca/tension. Intervals below are split-conformal
prediction intervals, not mechanistic parameter posteriors.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "Data" / "ventricular_cell_model"
OUTPUT = HERE / "strocchi_multiload_benchmark.json"
LOADS = (
    "contraction_isometric_stretch1.0",
    "contraction_isometric_stretch1.1",
    "contraction_isotonic",
)
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
SEEDS = range(10)


def table(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    return rows[0], rows[1:]


def paired_data():
    datasets = []
    for load in LOADS:
        parameter_names, parameters = table(ROOT / load / "parameters.csv")
        output_names, outputs = table(ROOT / load / "outputs.csv")
        if len(parameters) != len(outputs):
            raise ValueError(f"Row count mismatch: {load}")
        mapping = {tuple(p): np.asarray(y, dtype=float) for p, y in zip(parameters, outputs)}
        if len(mapping) != len(parameters):
            raise ValueError(f"Repeated parameter row: {load}")
        datasets.append((parameter_names, output_names, mapping))
    if not all(d[0] == datasets[0][0] for d in datasets):
        raise ValueError("Parameter order differs across loads")
    common = sorted(set.intersection(*(set(d[2]) for d in datasets)))
    theta = np.asarray(common, dtype=float)
    blocks = [np.stack([d[2][key] for key in common]) for d in datasets]
    if len(common) != 931 or not all(np.isfinite(x).all() for x in [theta, *blocks]):
        raise ValueError("Unexpected sample size or nonfinite data")
    return datasets[0][0], [d[1] for d in datasets], theta, blocks


def standardized(x, y, ids):
    x_mean, x_std = x[ids].mean(0), x[ids].std(0)
    y_mean, y_std = y[ids].mean(0), y[ids].std(0)
    x_std = np.where(x_std > 0, x_std, 1)
    y_std = np.where(y_std > 0, y_std, 1)
    return (x - x_mean) / x_std, (y - y_mean) / y_std, y_mean, y_std


def ridge_fit(x, y, ids, alpha):
    x1 = np.column_stack([np.ones(len(ids)), x[ids]])
    penalty = np.eye(x1.shape[1]) * alpha
    penalty[0, 0] = 0
    return np.linalg.solve(x1.T @ x1 + penalty, x1.T @ y[ids])


def predictions(x, beta):
    return np.column_stack([np.ones(len(x)), x]) @ beta


def fit_one(x, y, train, cal, test, rng):
    """Tune only inside train; reserve cal for intervals and test for scoring."""
    shuffled = rng.permutation(train)
    inner_train, inner_val = shuffled[:int(0.8 * len(train))], shuffled[int(0.8 * len(train)):]
    xs_inner, ys_inner, _, _ = standardized(x, y, inner_train)
    candidates = []
    for alpha in ALPHAS:
        beta = ridge_fit(xs_inner, ys_inner, inner_train, alpha)
        pred = predictions(xs_inner, beta)
        candidates.append((float(np.abs(ys_inner[inner_val] - pred[inner_val]).mean()), alpha))
    alpha = min(candidates)[1]

    xs, ys, y_mean, y_std = standardized(x, y, train)
    beta = ridge_fit(xs, ys, train, alpha)
    pred = predictions(xs, beta)
    residuals = np.abs(ys[cal] - pred[cal])
    # Per-coordinate split conformal; marginal, not joint or conditional.
    q_index = min(int(np.ceil((len(cal) + 1) * 0.9)) - 1, len(cal) - 1)
    q = np.sort(residuals, axis=0)[q_index]
    error = np.abs(ys[test] - pred[test])
    coverage = (error <= q).mean(0)
    return {
        "alpha": alpha,
        "mae_each": error.mean(0),
        "coverage_each": coverage,
        "interval_halfwidth_each": q,
        "mean_std_each": y_std,
        "test_pred": pred[test] * y_std + y_mean,
        "test_truth": y[test],
    }


def summarize(records, labels):
    return {
        "mae_standardized": float(np.mean([r["mae_each"].mean() for r in records])),
        "coverage_90pct_marginal_mean": float(np.mean([r["coverage_each"].mean() for r in records])),
        "interval_fullwidth_standardized_mean": float(np.mean([2 * r["interval_halfwidth_each"].mean() for r in records])),
        "per_coordinate": {
            f"{i}:{name}": {
                "mae_standardized": float(np.mean([r["mae_each"][i] for r in records])),
                "coverage_90pct": float(np.mean([r["coverage_each"][i] for r in records])),
                "interval_fullwidth_standardized": float(np.mean([2 * r["interval_halfwidth_each"][i] for r in records])),
            }
            for i, name in enumerate(labels)
        },
        "selected_alpha": dict(Counter(str(r["alpha"]) for r in records)),
    }


def screen_mechanical(theta, blocks, train, names):
    """Train-only forward linear sensitivity ranking; descriptive, not GSA."""
    all_outputs = np.column_stack(blocks)
    xs, ys, _, _ = standardized(theta, all_outputs, train)
    beta = ridge_fit(xs, ys, train, 10.0)[1:]
    score = np.sqrt(np.mean(beta ** 2, axis=1))
    mechanical = list(range(10, len(names)))
    return sorted(mechanical, key=lambda i: -score[i])[:10]


def main():
    names, output_names, theta, blocks = paired_data()
    inverse_specs = {
        "L1": (0,),
        "L1_L2": (0, 1),
        "L1_L3": (0, 2),
        "L1_L2_L3": (0, 1, 2),
    }
    # Each target load is completely hidden from test-sample calibration input.
    forward_specs = {
        "L1_to_L3": ((0,), 2),
        "L1_L2_to_L3": ((0, 1), 2),
        "L1_to_L2": ((0,), 1),
        "L1_L3_to_L2": ((0, 2), 1),
        "L2_to_L1": ((1,), 0),
        "L2_L3_to_L1": ((1, 2), 0),
    }
    inv = {key: [] for key in inverse_specs}
    forward = {key: [] for key in forward_specs}
    mean_target = {i: [] for i in range(3)}
    selected = []
    split_sizes = []
    for seed in SEEDS:
        rng = np.random.default_rng(20260928 + seed)
        perm = rng.permutation(len(theta))
        train, cal, test = perm[:558], perm[558:744], perm[744:]
        split_sizes.append([len(train), len(cal), len(test)])
        selected.append([names[i] for i in screen_mechanical(theta, blocks, train, names)])
        for label, indices in inverse_specs.items():
            x = np.column_stack([blocks[i] for i in indices])
            inv[label].append(fit_one(x, theta, train, cal, test, rng))
        for label, (indices, target) in forward_specs.items():
            x = np.column_stack([blocks[i] for i in indices])
            forward[label].append(fit_one(x, blocks[target], train, cal, test, rng))
        for i in range(3):
            scale = np.std(blocks[i][train], axis=0)
            scale = np.where(scale > 0, scale, 1)
            mean_target[i].append(float(np.mean(np.abs((blocks[i][test] - blocks[i][train].mean(0)) / scale))))

    inverse_summary = {label: summarize(records, names) for label, records in inv.items()}
    for label, records in inv.items():
        inverse_summary[label]["mae_calcium_related_10"] = float(np.mean([r["mae_each"][:10].mean() for r in records]))
        inverse_summary[label]["mae_mechanical_17"] = float(np.mean([r["mae_each"][10:].mean() for r in records]))
        inverse_summary[label]["width_mechanical_17"] = float(np.mean([2 * r["interval_halfwidth_each"][10:].mean() for r in records]))
    forward_summary = {}
    for label, records in forward.items():
        target = forward_specs[label][1]
        forward_summary[label] = summarize(records, output_names[target])
        forward_summary[label]["train_mean_target_mae_standardized"] = float(np.mean(mean_target[target]))
    per_split = []
    for i in range(len(SEEDS)):
        per_split.append({
            "seed": 20260928 + i,
            "inverse_mae": {label: float(records[i]["mae_each"].mean()) for label, records in inv.items()},
            "inverse_mechanical_mae": {label: float(records[i]["mae_each"][10:].mean()) for label, records in inv.items()},
            "heldout_load_mae": {label: float(records[i]["mae_each"].mean()) for label, records in forward.items()},
        })
    ranking = Counter(name for row in selected for name in row)
    result = {
        "source": str(ROOT),
        "n_common_parameter_sets": len(theta),
        "split_sizes_train_calibration_test": split_sizes[0],
        "n_splits": len(SEEDS),
        "load_names": dict(enumerate(LOADS)),
        "output_names_by_load": output_names,
        "inverse": inverse_summary,
        "heldout_load": forward_summary,
        "per_split": per_split,
        "train_only_forward_linear_screen_top10_frequency": dict(ranking.most_common()),
        "cautions": [
            "No paired AP or Ca outputs are available for these exact 931 parameter samples.",
            "Intervals are per-coordinate split-conformal prediction intervals under the simulator sampling distribution, not Bayesian parameter posteriors.",
            "A direct supervised ridge map predicts held-out load features; it does not propagate a mechanistic parameter posterior through a forward solver.",
            "The contraction output CSV repeats dT/dtMax,v as both column 3 and 4; positional indexing preserves both values but physiological sign interpretation requires source confirmation.",
            "Intersection of completed simulations may be a selected subset of all designs; real-cell transportability is untested.",
        ],
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "inverse": {k: {m: v[m] for m in ("mae_standardized", "mae_mechanical_17", "width_mechanical_17", "coverage_90pct_marginal_mean")} for k, v in inverse_summary.items()},
        "heldout_load": {k: {m: v[m] for m in ("mae_standardized", "train_mean_target_mae_standardized", "coverage_90pct_marginal_mean")} for k, v in forward_summary.items()},
        "top_screen": ranking.most_common(10),
    }, indent=2))


if __name__ == "__main__":
    main()
