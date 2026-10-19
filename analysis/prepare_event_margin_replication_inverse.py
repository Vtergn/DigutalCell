"""Frozen development PLS inverse applied to new event-margin confirmation cells."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parent / "human_bidirectional_output"
LO = np.array([.9, .85, .85, .85, .9, .8, .4, .75, 1.8,
               4.5, 1.8, -4.1, 100.])
HI = np.array([1.1, 1.15, 1.15, 1.15, 1.1, 1.2, 1.6, .86,
               2.2, 5.5, 2.8, -.9, 140.])

with h5py.File(OUT / "dynamic_discrepancy_development.mat") as f:
    dev = f["waveforms"][:].astype(float)
    dev_ok = f["ok"][:].astype(bool)
with h5py.File(OUT / "feedback_13param_cohort.mat") as f:
    dev_theta = f["theta"][:].T
with h5py.File(OUT / "dynamic_c3_event_margin_replicate.mat") as f:
    test = f["waveforms"][:].astype(float)
    test_ok = f["ok"][:].astype(bool)

dev_ids = np.flatnonzero(dev_ok[:, 1])
test_ids = np.flatnonzero(test_ok[:, :3].all(axis=1))
assert len(dev_ids) == 47
assert test.shape[0] == 128 and test.shape[1] == 4

scale = np.array([20., .1, 2.])[None, :, None]


def features(w):
    baseline, c1, c2 = w[:, 0], w[:, 1], w[:, 2]
    parts = [(baseline[:, :, ::20] / scale).reshape(len(w), -1)]
    for z, onset in [(c1, 150), (c2, 600)]:
        response = (z[:, :, onset:onset+201:10] -
                    baseline[:, :, onset:onset+201:10]) / scale
        parts.append(response.reshape(len(w), -1))
    return np.concatenate(parts, axis=1)


xdev = features(dev[dev_ids, 1])
xtest = features(test[test_ids])
ydev = (dev_theta[dev_ids] - LO) / (HI - LO)
scaler = StandardScaler().fit(xdev)
a, b = scaler.transform(xdev), scaler.transform(xtest)
a[:, scaler.scale_ < 1e-10] = 0
b[:, scaler.scale_ < 1e-10] = 0
model = PLSRegression(n_components=20, scale=False).fit(a, ydev)
estimated = LO + (HI - LO) * np.clip(model.predict(b), 0, 1)
rows = np.column_stack([test_ids + 1, estimated])
np.savetxt(OUT / "dynamic_event_margin_replication_inverse.csv", rows,
           fmt="%.12g", delimiter=",")
(OUT / "dynamic_event_margin_replication_inverse_metadata.json").write_text(
    json.dumps({"n_development": len(dev_ids), "n_test_observable": len(test_ids),
                "model": "PLS20 hard clip, frozen from earlier C3 gate",
                "test_target_used": False, "test_parameters_used": False}, indent=2),
    encoding="utf-8")
print(f"inverse: development={len(dev_ids)} test_observable={len(test_ids)}")

