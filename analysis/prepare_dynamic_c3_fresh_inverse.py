"""Train frozen C3 inverse on development only, predict new-seed cells."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from scipy.special import expit, logit
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parent / "human_bidirectional_output"
decision = json.loads((OUT / "dynamic_c3_development_scores.json").read_text())
arm = decision["selected_arm"]
method = decision["chosen_method"]
assert arm in [4, 5] and method in [0, 1]
with h5py.File(OUT / "dynamic_discrepancy_development.mat") as f:
    dev = f["waveforms"][:].astype(float)
    dev_ok = f["ok"][:].astype(bool)
with h5py.File(OUT / "dynamic_c3_development.mat") as f:
    dev_c3 = f["waveforms"][:].astype(float)
    dev_c3_ok = f["ok"][:].astype(bool)
with h5py.File(OUT / "feedback_13param_cohort.mat") as f:
    dev_theta = f["theta"][:].T
with h5py.File(OUT / "dynamic_c3_fresh.mat") as f:
    fresh = f["waveforms"][:].astype(float)
    fresh_ok = f["ok"][:].astype(bool)

LO = np.array([.9, .85, .85, .85, .9, .8, .4, .75, 1.8,
               4.5, 1.8, -4.1, 100.])
HI = np.array([1.1, 1.15, 1.15, 1.15, 1.1, 1.2, 1.6, .86,
               2.2, 5.5, 2.8, -.9, 140.])
dev_ids = np.flatnonzero(dev_ok[:, 1] & dev_c3_ok[:, arm-1])
fresh_ids = np.flatnonzero(fresh_ok.all(axis=1))
assert len(dev_ids) == 47
scale = np.array([20., .1, 2.])[None, :, None]


def featurize(baseline, c1, c2, c3=None):
    parts = [(baseline[:, :, ::20] / scale).reshape(len(baseline), -1)]
    for z, onset in [(c1, 150), (c2, 600)]:
        r = (z[:, :, onset:onset+201:10] -
             baseline[:, :, onset:onset+201:10]) / scale
        parts.append(r.reshape(len(z), -1))
    if c3 is not None:
        r = (c3[:, :, 600:801:10] -
             baseline[:, :, 600:801:10]) / scale
        parts.append(r.reshape(len(c3), -1))
    return np.concatenate(parts, axis=1)


y = (dev_theta[dev_ids]-LO)/(HI-LO)
rows = []
for current in [0, arm]:
    xdev = featurize(dev[dev_ids, 1, 0], dev[dev_ids, 1, 1],
                     dev[dev_ids, 1, 2],
                     dev_c3[dev_ids, arm-1] if current else None)
    xfresh = featurize(fresh[fresh_ids, 0], fresh[fresh_ids, 1],
                       fresh[fresh_ids, 2], fresh[fresh_ids, 3] if current else None)
    scaler = StandardScaler().fit(xdev)
    a, b = scaler.transform(xdev), scaler.transform(xfresh)
    a[:, scaler.scale_ < 1e-10] = 0
    b[:, scaler.scale_ < 1e-10] = 0
    target = y if method == 0 else logit(y)
    reg = PLSRegression(n_components=20, scale=False).fit(a, target)
    normalized = reg.predict(b)
    normalized = np.clip(normalized, 0, 1) if method == 0 else expit(normalized)
    estimated = LO + (HI-LO)*normalized
    rows.extend(np.column_stack([fresh_ids+1, np.full(len(fresh_ids), current),
                                 estimated]).tolist())

np.savetxt(OUT / "dynamic_c3_fresh_inverse.csv", np.asarray(rows),
           fmt="%.12g", delimiter=",")
(OUT / "dynamic_c3_fresh_inverse_metadata.json").write_text(json.dumps({
    "selected_arm": arm, "method": method,
    "n_development_cells": len(dev_ids),
    "n_fresh_qc_cells": len(fresh_ids),
    "new_seed_truth_parameters_used_for_fitting": False,
    "fresh_target_used_for_fitting": False,
}, indent=2), encoding="utf-8")
print("fresh inverse", "dev", len(dev_ids), "test", len(fresh_ids),
      "arm", arm, "method", method)
