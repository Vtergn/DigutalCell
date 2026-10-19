"""Pre-specified noise and within-framework structural-discrepancy stress test."""

import json
from pathlib import Path

import numpy as np

from score_human_feedback_observability import fit_predict

ROOT = Path(__file__).resolve().parent / "human_bidirectional_output"
source = np.loadtxt(ROOT / "feedback_13param_cohort.csv", delimiter=",")
altered = np.loadtxt(ROOT / "feedback_13param_discrepancy.csv", delimiter=",")
assert source.shape == altered.shape and source.shape[1] == 113
valid = (source[:, -1] == 1) & (altered[:, -1] == 1)
source, altered = source[valid], altered[valid]
assert np.allclose(source[:, :13], altered[:, :13])


def unpack(a):
    base = a[:, 13:16]
    feature = a[:, 16:-1].reshape(len(a), 12, 8).transpose(0, 2, 1)
    y = feature[:, 5:8, :][:, :, [0, 2, 3, 6, 8, 10]]
    return base, feature, y


sb, sf, sy = unpack(source)
ab, af, ay = unpack(altered)
floor = np.array([.1, .1, .001, .1, .1, .0005])


def score(pred, true, scales):
    active = scales >= floor
    err = np.abs(pred-true) / np.maximum(scales, floor)
    per = (err*active).sum(axis=(0, 2)) / (len(pred)*active.sum(axis=1))
    return float(per.mean()), per.tolist()


records = []
for seed in range(30):
    rng = np.random.default_rng(90000 + seed)
    order = rng.permutation(len(source))
    nt = max(10, round(.2 * len(source)))
    nv = max(10, round(.2 * len(source)))
    test, val, train = order[:nt], order[nt:nt+nv], order[nt+nv:]
    dev = np.r_[train, val]
    scales = np.maximum(sy[train].std(axis=0), 1e-9)
    # Repeat the nested selector without seeing alternate-generator outputs.
    val_errors = []
    for k in range(5):
        xtrain = np.c_[sb[train], sf[train, k, :]]
        xval = np.c_[sb[val], sf[val, k, :]]
        pv = fit_predict(xtrain, sy[train].reshape(len(train), -1), xval)
        val_errors.append(score(pv.reshape(len(val), 3, 6), sy[val], scales)[0])
    selected = int(np.argmin(val_errors))
    one = {"seed": seed, "selected": selected}
    for k, name in [(selected, "selected"), (2, "fixed_mid")]:
        xdev = np.c_[sb[dev], sf[dev, k, :]]
        xs = np.c_[sb[test], sf[test, k, :]]
        xa = np.c_[ab[test], af[test, k, :]]
        model_y = sy[dev].reshape(len(dev), -1)
        pred_source = fit_predict(xdev, model_y, xs).reshape(len(test), 3, 6)
        pred_altered = fit_predict(xdev, model_y, xa).reshape(len(test), 3, 6)
        one[name + "_source"] = score(pred_source, sy[test], scales)[0]
        one[name + "_discrepancy"] = score(pred_altered, ay[test], scales)[0]
        for frac in [.02, .10]:
            sd = xdev.std(axis=0)
            noisy = xa + rng.normal(size=xa.shape) * frac * sd
            pred = fit_predict(xdev, model_y, noisy).reshape(len(test), 3, 6)
            one[name + f"_discrepancy_noise{frac}"] = score(pred, ay[test], scales)[0]
    basepred = fit_predict(sb[dev], sy[dev].reshape(len(dev), -1), ab[test])
    one["baseline_discrepancy"] = score(
        basepred.reshape(len(test), 3, 6), ay[test], scales)[0]
    records.append(one)

summary = {key: float(np.mean([r[key] for r in records]))
           for key in records[0] if key not in ("seed", "selected")}
out = {
    "n_cells": len(source), "n_repeated_splits": 30,
    "model_discrepancy": "True generator has gain exponent 0.5, inference/training generator exponent 0. This is a controlled law change within one simulator framework, not an independent published model.",
    "input_noise": "Independent feature noise at 2% or 10% of source population SD, uncalibrated to hardware.",
    "summary": summary, "splits": records,
}
path = ROOT / "feedback_13param_stress_score.json"
path.write_text(json.dumps(out, indent=2), encoding="utf-8")
for k, v in summary.items():
    print(k, round(v, 4))
print(path)
