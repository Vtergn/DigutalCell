"""Held-out-cell score for baseline-varying six-parameter human virtual cells."""

import json
from pathlib import Path

import numpy as np

from score_human_feedback_observability import fit_predict

ROOT = Path(__file__).resolve().parent / "human_bidirectional_output"
A = np.loadtxt(ROOT / "feedback_hetero_challenges.csv", delimiter=",")
assert A.shape[1] == 45 and np.isfinite(A).all()

theta = A[:, :6]
rest, early, late, target = A[:, 6:9], A[:, 9:21], A[:, 21:33], A[:, 33:45]
apca = [0, 1, 2, 3, 4, 8, 9, 10, 11]
inputs = {
    "rest_AP_Ca": rest[:, :2],
    "rest_AP_Ca_T": rest,
    "early_AP_Ca": np.c_[rest[:, :2], early[:, apca]],
    "early_AP_Ca_T": np.c_[rest, early],
    "late_AP_Ca_T": np.c_[rest, late],
    "early_late_AP_Ca_T": np.c_[rest, early, late],
}
ycols = [0, 2, 3, 6, 8, 10]
ynames = ["delta_APD_ms", "max_delta_Vm_mV", "min_delta_Ca_uM",
          "Ta_20ms_kPa", "delta_Vm_20ms_mV", "delta_Ca_20ms_uM"]
y = target[:, ycols]
pscale = np.array([.3, .4, .4, 1.2, 3.2, .4])
yscale = np.maximum(y.std(axis=0), 1e-9)
records = {}
for noise in [0.0, .02, .10]:
    for name, xx in inputs.items():
        pe, ye = [], []
        for seed in range(20):
            rng = np.random.default_rng(10000 + seed)
            order = rng.permutation(len(A))
            test, train = order[:12], order[12:]
            nsd = noise * xx.std(axis=0)
            xtrain = xx[train] + rng.normal(size=(len(train), xx.shape[1])) * nsd
            xtest = xx[test] + rng.normal(size=(len(test), xx.shape[1])) * nsd
            pe.append(np.abs(fit_predict(xtrain, theta[train], xtest)-theta[test]))
            ye.append(np.abs(fit_predict(xtrain, y[train], xtest)-y[test]))
        pe, ye = np.concatenate(pe), np.concatenate(ye)
        records[f"{name}|noise_{noise:g}"] = {
            "parameter_normalized_MAE": float(np.mean(pe / pscale)),
            "parameter_MAE": dict(zip(
                ["CaL", "SERCA", "NCX", "g_ionic", "beta1", "Tref_scale"],
                pe.mean(axis=0).tolist())),
            "target_standardized_MAE": float(np.mean(ye / yscale)),
            "target_MAE": dict(zip(ynames, ye.mean(axis=0).tolist())),
        }

report = {
    "n_cells": len(A),
    "split": "20 repeated 36/12 held-out-cell splits; repeated splits are dependent",
    "source": "Human ToR-ORd/SAC-Land2016, cell-specific pre-pacing, six synthetic parameters",
    "notes": "All train and test cells share the same equation family. Synthetic input noise is fraction of feature population SD, not experimentally calibrated.",
    "target_population_SD": dict(zip(ynames, y.std(axis=0).tolist())),
    "results": records,
}
out = ROOT / "feedback_hetero_score.json"
out.write_text(json.dumps(report, indent=2), encoding="utf-8")
# Fixed separate split for native-ODE closure; all six inferred parameters
# are used to re-pace a fresh cell, with no true initial state supplied.
rng = np.random.default_rng(777)
order = rng.permutation(len(A))
test, train = order[:12], order[12:]
x = inputs["early_AP_Ca_T"]
inferred = fit_predict(x[train], theta[train], x[test])
inferred = np.clip(inferred,
                   [.85, .8, .8, .4, -4.1, .8],
                   [1.15, 1.2, 1.2, 1.6, -.9, 1.2])
direct = fit_predict(x[train], y[train], x[test])
np.savetxt(ROOT / "feedback_hetero_native_input.csv",
           np.c_[test + 1, theta[test], inferred, y[test], direct],
           delimiter=",", fmt="%.12g",
           header=",".join(["cell_index"]
               + ["true_" + q for q in ["CaL", "SERCA", "NCX", "g_ionic", "beta1", "Tref_scale"]]
               + ["est_" + q for q in ["CaL", "SERCA", "NCX", "g_ionic", "beta1", "Tref_scale"]]
               + ["true_" + q for q in ynames]
               + ["direct_" + q for q in ynames]),
           comments="")
for k, v in records.items():
    print(k, round(v["parameter_normalized_MAE"], 4),
          round(v["target_standardized_MAE"], 4),
          round(v["parameter_MAE"]["g_ionic"], 4))
print(out)
