"""Train exact shared-kernel GPs for each load, excluding all held-out identities.

This first audits forward-emulator accuracy before attempting posterior inversion.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist, pdist

from strocchi_multiload_benchmark import ROOT, LOADS, paired_data


OUT = Path(__file__).with_suffix(".json")


def read(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    return rows[0], rows[1:]


def records():
    data = []
    for name in LOADS:
        _, raw_x = read(ROOT / name / "parameters.csv")
        _, raw_y = read(ROOT / name / "outputs.csv")
        data.append({tuple(map(float, row)): np.asarray(y, float) for row, y in zip(raw_x, raw_y)})
    return data


def matern52(a, b, ell):
    d = cdist(a, b) / ell
    t = np.sqrt(5) * d
    return (1 + t + t * t / 3) * np.exp(-t)


class GP:
    def __init__(self, x, y, ell_factor=0.65, nugget=0.02):
        self.xmean, self.xstd = x.mean(0), x.std(0)
        self.ystd, self.ymean = y.std(0), y.mean(0)
        self.xstd = np.where(self.xstd > 0, self.xstd, 1)
        self.ystd = np.where(self.ystd > 0, self.ystd, 1)
        self.x = (x - self.xmean) / self.xstd
        yz = (y - self.ymean) / self.ystd
        self.ell = float(np.median(pdist(self.x)) * ell_factor)
        k = matern52(self.x, self.x, self.ell)
        k.flat[::len(k) + 1] += nugget
        self.factor = cho_factor(k, lower=True, check_finite=False)
        self.alpha = cho_solve(self.factor, yz, check_finite=False)
        self.nugget = nugget

    def predict(self, x, batch=1000, variance=False):
        outputs, variances = [], []
        for start in range(0, len(x), batch):
            z = (x[start:start + batch] - self.xmean) / self.xstd
            k = matern52(z, self.x, self.ell)
            outputs.append((k @ self.alpha) * self.ystd + self.ymean)
            if variance:
                v = cho_solve(self.factor, k.T, check_finite=False)
                var = np.maximum(0, 1 - np.sum(k * v.T, axis=1))
                variances.append(var[:, None] * self.ystd[None, :] ** 2)
        predicted = np.vstack(outputs)
        if variance:
            return predicted, np.vstack(variances)
        return predicted


def load_for_split(dataset, excluded, load_idx):
    keys = sorted(k for k in dataset[load_idx] if k not in excluded)
    x = np.asarray(keys, float)
    y = np.stack([dataset[load_idx][k] for k in keys])
    return x, y


def main():
    _, _, theta, blocks = paired_data()
    dataset = records()
    rng = np.random.default_rng(20260928)
    perm = rng.permutation(len(theta))
    train, cal, test = perm[:558], perm[558:744], perm[744:]
    excluded = {tuple(row) for row in theta[np.r_[cal, test]]}
    result = {"split": [len(train), len(cal), len(test)], "excluded_identities": len(excluded), "loads": {}}
    for i, load in enumerate(LOADS):
        x_train, y_train = load_for_split(dataset, excluded, i)
        assert not any(tuple(row) in excluded for row in x_train)
        gp = GP(x_train, y_train)
        pred, var = gp.predict(theta[cal], variance=True)
        y_true = blocks[i][cal]
        zerr = (pred - y_true) / gp.ystd
        result["loads"][load] = {
            "n_train": len(x_train),
            "n_calibration": len(cal),
            "ell": gp.ell,
            "nugget": gp.nugget,
            "calibration_mae_standardized": float(np.abs(zerr).mean()),
            "calibration_mae_each_standardized": list(map(float, np.abs(zerr).mean(0))),
            "calibration_r2_each": list(map(float, 1 - np.sum((pred - y_true) ** 2, axis=0) / np.sum((y_true - y_true.mean(0)) ** 2, axis=0))),
            "gp_variance_to_empirical_squared_error": float(np.mean(var / gp.ystd**2) / np.mean(zerr**2)),
        }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
