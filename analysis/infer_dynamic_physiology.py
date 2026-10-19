"""Develop C1/C2-only 13-parameter inverse models without sealed targets.

All model choices use five-fold development-cell CV on physiological parameter
error. The independent confirmation cohort is evaluated once after selection.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parent / "human_bidirectional_output"
with h5py.File(OUT / "dynamic_discrepancy_development.mat") as f:
    dev_w = f["waveforms"][:].astype(float)
    dev_valid = f["ok"][:].astype(bool).all(axis=1)
with h5py.File(OUT / "dynamic_discrepancy_cohort.mat") as f:
    test_w = f["waveforms"][:].astype(float)
    test_valid = f["ok"][:].astype(bool).all(axis=1)
with h5py.File(OUT / "feedback_13param_cohort.mat") as f:
    dev_theta = f["theta"][:].T
with h5py.File(OUT / "feedback_13param_confirm_cohort.mat") as f:
    test_theta = f["theta"][:].T

dev_ids = np.flatnonzero(dev_valid)
test_ids = np.flatnonzero(test_valid)
dev_w, dev_theta = dev_w[dev_valid, 0], dev_theta[dev_valid]
test_w, test_theta = test_w[test_valid], test_theta[test_valid]
assert dev_w.shape == (47, 6, 3, 1001)

LO = np.array([.9, .85, .85, .85, .9, .8, .4, .75, 1.8,
               4.5, 1.8, -4.1, 100.])
HI = np.array([1.1, 1.15, 1.15, 1.15, 1.1, 1.2, 1.6, .86,
               2.2, 5.5, 2.8, -.9, 140.])
RANGE = HI-LO
Y = (dev_theta-LO)/RANGE


def features(w: np.ndarray, include_challenge: bool) -> np.ndarray:
    """Observed baseline and causal response snippets, channel balanced."""
    scale = np.array([20.0, .1, 2.0])[None, :, None]
    baseline = (w[:, 0, :, ::20]/scale).reshape(len(w), -1)
    if not include_challenge:
        return baseline
    parts = [baseline]
    for protocol, onset in [(1, 150), (2, 600)]:
        response = (w[:, protocol, :, onset:onset+201:10] -
                    w[:, 0, :, onset:onset+201:10]) / scale
        parts.append(response.reshape(len(w), -1))
    return np.concatenate(parts, axis=1)


def fit_predict(xtr, ytr, xte, kind, complexity):
    scaler = StandardScaler().fit(xtr)
    a, b = scaler.transform(xtr), scaler.transform(xte)
    # Suppress columns whose development variance is zero.
    a[:, scaler.scale_ < 1e-10] = 0
    b[:, scaler.scale_ < 1e-10] = 0
    dim = min(complexity, len(xtr)-2)
    if kind == "pls":
        reg = PLSRegression(n_components=dim, scale=False).fit(a, ytr)
        result = reg.predict(b)
    else:
        pca = PCA(n_components=dim, random_state=0).fit(a)
        a, b = pca.transform(a), pca.transform(b)
        if kind == "ridge":
            result = Ridge(alpha=10).fit(a, ytr).predict(b)
        else:
            dd = cdist(a, a, "sqeuclidean")
            scale = np.median(dd[dd > 1e-12])
            kernel = np.exp(-dd / max(scale, 1e-12))
            cross = np.exp(-cdist(b, a, "sqeuclidean") /
                           max(scale, 1e-12))
            mean = ytr.mean(axis=0)
            result = mean + cross @ np.linalg.solve(
                kernel + .1*np.eye(len(a)), ytr-mean)
    return np.clip(result, 0, 1)


cv = KFold(n_splits=5, shuffle=True, random_state=2602)
records = []
fits = {}
for observation in ["baseline", "baseline_C1_C2"]:
    x = features(dev_w, observation != "baseline")
    candidates = [(k, d) for k in ["ridge", "rbf", "pls"]
                  for d in [5, 10, 20]]
    for kind, dim in candidates:
        prediction = np.empty_like(Y)
        for train, val in cv.split(x):
            prediction[val] = fit_predict(x[train], Y[train], x[val],
                                          kind, dim)
        mae = np.abs(prediction-Y).mean(axis=0)
        records.append({"observation": observation, "model": kind,
                        "components": dim,
                        "cv_parameter_normalized_MAE": float(mae.mean()),
                        "cv_parameter_MAE_by_index": mae.tolist()})
    best = min((r for r in records if r["observation"] == observation),
               key=lambda r: r["cv_parameter_normalized_MAE"])
    fits[observation] = best

rows = []
test_metrics = []
for observation, best in fits.items():
    xtrain = features(dev_w, observation != "baseline")
    for form in range(3):
        xtest = features(test_w[:, form], observation != "baseline")
        estimate01 = fit_predict(xtrain, Y, xtest,
                                 best["model"], best["components"])
        estimated = LO + estimate01*RANGE
        normalized_error = np.abs(estimated-test_theta)/RANGE
        test_metrics.append({"observation": observation, "form": form,
                             "normalized_MAE": float(normalized_error.mean()),
                             "normalized_MAE_by_index":
                                 normalized_error.mean(axis=0).tolist()})
        for cell_id, p in zip(test_ids, estimated):
            rows.append([cell_id+1, form+1,
                         0 if observation == "baseline" else 1, *p])

np.savetxt(OUT / "dynamic_physiology_inverse.csv", np.asarray(rows),
           delimiter=",", fmt="%.12g")
report = {"n_dev": len(dev_ids), "n_test": len(test_ids),
          "parameter_order": ["ionic1", "ionic2", "ionic3", "ionic4",
                              "ionic5", "ionic6", "g_ionic", "ca50",
                              "TRPN_n", "nperm", "beta0", "beta1", "Tref"],
          "bounds_lo": LO.tolist(), "bounds_hi": HI.tolist(),
          "all_cv_candidates": records,
          "selected_cv_models": fits,
          "confirmation_diagnostic_metrics": test_metrics,
          "selection_used_target_protocols": False,
          "selection_used_confirmation_parameters": False}
(OUT / "dynamic_physiology_inverse_score.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8")
print("SELECTED", fits)
for z in test_metrics:
    print(z["observation"], "form", z["form"],
          "normalized_MAE", z["normalized_MAE"],
          "g_ionic", z["normalized_MAE_by_index"][6])
