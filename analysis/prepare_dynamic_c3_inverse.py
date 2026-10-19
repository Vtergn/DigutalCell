"""Cell-held-out C3 inverse vectors; no sealed target enters any feature."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from scipy.special import expit, logit
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parent / "human_bidirectional_output"
with h5py.File(OUT / "dynamic_discrepancy_development.mat") as f:
    old = f["waveforms"][:].astype(float)
    old_ok = f["ok"][:].astype(bool)
with h5py.File(OUT / "dynamic_c3_development.mat") as f:
    extra = f["waveforms"][:].astype(float)
    extra_ok = f["ok"][:].astype(bool)
with h5py.File(OUT / "feedback_13param_cohort.mat") as f:
    theta = f["theta"][:].T

LO = np.array([.9, .85, .85, .85, .9, .8, .4, .75, 1.8,
               4.5, 1.8, -4.1, 100.])
HI = np.array([1.1, 1.15, 1.15, 1.15, 1.1, 1.2, 1.6, .86,
               2.2, 5.5, 2.8, -.9, 140.])
RANGE = HI - LO
ids = np.flatnonzero(old_ok[:, 1])
assert len(ids) > 30
candidate_qc = extra_ok[ids].sum(axis=0).tolist()
eligible_arms = [j+1 for j, count in enumerate(candidate_qc)
                 if count == len(ids)]
y = (theta[ids] - LO) / RANGE
assert np.all((y > 0) & (y < 1))
scale = np.array([20., .1, 2.])[None, :, None]


def feature(arm):
    w = old[ids, 1]
    baseline = (w[:, 0, :, ::20] / scale).reshape(len(ids), -1)
    parts = [baseline]
    for cond, onset in [(1, 150), (2, 600)]:
        z = (w[:, cond, :, onset:onset+201:10] -
             w[:, 0, :, onset:onset+201:10]) / scale
        parts.append(z.reshape(len(ids), -1))
    if arm:
        onset = 150 if arm <= 3 else 600
        c3 = extra[ids, arm-1]
        z = (c3[:, :, onset:onset+201:10] -
             w[:, 0, :, onset:onset+201:10]) / scale
        parts.append(z.reshape(len(ids), -1))
    return np.concatenate(parts, axis=1)


def fit_predict(xtr, ytr, xte, transform):
    scaler = StandardScaler().fit(xtr)
    a, b = scaler.transform(xtr), scaler.transform(xte)
    a[:, scaler.scale_ < 1e-10] = 0
    b[:, scaler.scale_ < 1e-10] = 0
    if transform == "logit":
        target = logit(ytr)
    else:
        target = ytr
    reg = PLSRegression(n_components=min(20, len(xtr)-2),
                        scale=False).fit(a, target)
    z = reg.predict(b)
    return expit(z) if transform == "logit" else np.clip(z, 0, 1)


folds = list(KFold(n_splits=5, shuffle=True,
                   random_state=20261002).split(ids))
methods = {}
x0 = feature(0)
for method in ["clip", "logit"]:
    pred = np.full_like(y, np.nan)
    for tr, te in folds:
        pred[te] = fit_predict(x0[tr], y[tr], x0[te], method)
    err = np.abs(pred-y)
    methods[method] = {"normalized_mae": float(err.mean()),
                       "ionic_feedback_mae": float(err[:, [3, 5, 6]].mean()),
                       "exact_boundary_fraction": float(np.mean((pred == 0) | (pred == 1)))}
rows = []
for method_id, method in enumerate(["clip", "logit"]):
    for arm in [0] + eligible_arms:
        x = feature(arm)
        pred = np.full_like(y, np.nan)
        for tr, te in folds:
            pred[te] = fit_predict(x[tr], y[tr], x[te], method)
        actual = LO + RANGE * pred
        rows.extend(np.column_stack([ids+1, np.full(len(ids), arm),
                                     np.full(len(ids), method_id), actual]).tolist())

np.savetxt(OUT / "dynamic_c3_dev_inverse.csv", np.asarray(rows),
           fmt="%.12g", delimiter=",")
(OUT / "dynamic_c3_inverse_development.json").write_text(json.dumps({
    "n_development_cells": len(ids), "dev_cell_ids_1based": (ids+1).tolist(),
    "candidate_qc_pass_counts": candidate_qc,
    "eligible_arms": eligible_arms,
    "five_fold_boundary_methods": methods,
    "boundary_method_selection": "no-C3 development CVaR90, before C3 choice",
    "no_sealed_target_feature": True,
}, indent=2), encoding="utf-8")
print(json.dumps({"n": len(ids), "methods": methods,
                  "eligible_arms": eligible_arms}, indent=2))
