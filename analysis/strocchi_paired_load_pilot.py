"""Small, paired load-condition observability pilot on Zenodo 7405335.

Compares parameter recovery from identical 931 parameter sets measured under
one, two, or three contraction loading conditions. This is *not* an AP/Ca/T
multimodal benchmark; the released datasets do not contain paired AP/Ca/T.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent / "Data" / "ventricular_cell_model"
OUT = Path(__file__).with_suffix(".json")
CONDITIONS = (
    "contraction_isometric_stretch1.0",
    "contraction_isometric_stretch1.1",
    "contraction_isotonic",
)


def read_table(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    return rows[0], rows[1:]


def load():
    data = {}
    for name in CONDITIONS:
        ph, pr = read_table(ROOT / name / "parameters.csv")
        oh, out = read_table(ROOT / name / "outputs.csv")
        assert len(pr) == len(out)
        assert len({tuple(x) for x in pr}) == len(pr)
        data[name] = (ph, oh, {tuple(p): np.asarray(o, dtype=float) for p, o in zip(pr, out)})
    common = sorted(set.intersection(*(set(x[2]) for x in data.values())))
    params = np.asarray(common, dtype=float)
    blocks = [np.stack([data[name][2][key] for key in common]) for name in CONDITIONS]
    assert len(common) == 931
    assert all(np.isfinite(x).all() for x in [params, *blocks])
    return data[CONDITIONS[0]][0], params, blocks


def fit_predict(x, y, train, val, test, alpha):
    mx, sx = x[train].mean(0), x[train].std(0)
    my, sy = y[train].mean(0), y[train].std(0)
    sx[sx == 0] = 1
    sy[sy == 0] = 1
    xs = (x - mx) / sx
    ys = (y - my) / sy
    xx = np.column_stack([np.ones(len(train)), xs[train]])
    penalty = np.eye(xx.shape[1]) * alpha
    penalty[0, 0] = 0
    beta = np.linalg.solve(xx.T @ xx + penalty, xx.T @ ys[train])
    pred = np.column_stack([np.ones(len(x)), xs]) @ beta
    return ys, pred, np.mean(np.abs(ys[val] - pred[val])), sy


def main():
    names, y, blocks = load()
    rng = np.random.default_rng(20260928)
    all_results = []
    for seed in range(10):
        perm = rng.permutation(len(y))
        train, val, test = perm[:650], perm[650:790], perm[790:]
        report = {"seed": seed, "n_train": len(train), "n_val": len(val), "n_test": len(test)}
        y_scale = y[train].std(0)
        y_scale[y_scale == 0] = 1
        baseline_error = np.abs((y[test] - y[train].mean(0)) / y_scale)
        report["train_mean"] = {
            "mae_all_standardized": float(baseline_error.mean()),
            "mae_calcium_related_10": float(baseline_error[:, :10].mean()),
            "mae_contraction_17": float(baseline_error[:, 10:].mean()),
        }
        for indices, label in [
            ((0,), "one_load"),
            ((0, 1), "two_loads"),
            ((0, 2), "isometric_plus_isotonic"),
            ((0, 1, 2), "three_loads"),
        ]:
            x = np.concatenate([blocks[i] for i in indices], axis=1)
            trials = [
                (alpha, *fit_predict(x, y, train, val, test, alpha)[:3])
                for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0)
            ]
            alpha, ys, pred, _ = min(trials, key=lambda z: z[3])
            errors = np.abs(ys[test] - pred[test])
            report[label] = {
                "alpha": alpha,
                "mae_all_standardized": float(errors.mean()),
                "mae_calcium_related_10": float(errors[:, :10].mean()),
                "mae_contraction_17": float(errors[:, 10:].mean()),
                "per_parameter_mae": dict(zip(names, map(float, errors.mean(0)))),
            }
        all_results.append(report)
    summary = {}
    for label in ("train_mean", "one_load", "two_loads", "isometric_plus_isotonic", "three_loads"):
        summary[label] = {
            metric: float(np.mean([r[label][metric] for r in all_results]))
            for metric in ("mae_all_standardized", "mae_calcium_related_10", "mae_contraction_17")
        }
    result = {
        "source": str(ROOT),
        "design": "same 931 parameter rows across three mechanical load conditions; fixed 650/140/141 train/validation/test for each of 10 splits; ridge alpha chosen on validation; target standardized on train only",
        "parameter_names": names,
        "summary": summary,
        "splits": all_results,
        "limits": [
            "The ionic Ca dataset has zero exact shared 10-parameter keys with these contraction rows.",
            "There are no AP traces or paired AP/Ca/T triples; this tests multiple loading conditions, not three-modal observability.",
            "Ridge parameter recovery on simulator samples is a diagnostic, not posterior calibration or real-cell validation.",
        ],
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "n_shared": len(y)}, indent=2))


if __name__ == "__main__":
    main()
