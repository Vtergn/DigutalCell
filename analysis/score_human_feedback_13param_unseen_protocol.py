"""Evaluate target protocols withheld from direct-model population training.

This is a strict leave-protocol-out check on a fixed 52/12 cell split. The
direct model sees responses to four *other* mechanical protocols from the
training cells, plus the same 300-ms calibration features as the native twin.
No target-protocol response enters the direct training set.
"""

import json
from pathlib import Path

import numpy as np

from score_human_feedback_observability import fit_predict


ROOT = Path(__file__).resolve().parent / "human_bidirectional_output"
SRC = np.loadtxt(ROOT / "feedback_13param_cohort.csv", delimiter=",")
ALT = np.loadtxt(ROOT / "feedback_13param_discrepancy.csv", delimiter=",")
valid = (SRC[:, -1] == 1) & (ALT[:, -1] == 1)
SRC, ALT = SRC[valid], ALT[valid]
assert SRC.shape == ALT.shape and np.allclose(SRC[:, :13], ALT[:, :13])

YCOLS = [0, 2, 3, 6, 8, 10]
META = np.array([
    [150, 1.10, 10], [600, 1.10, 10], [300, 1.10, 10],
    [150, 1.05, 10], [600, 1.05, 10], [225, 1.075, 10],
    [225, 1.125, 20], [750, 1.08, 10],
])
TARGETS = ["interpolation_225_1075_10", "amplitude_duration_225_1125_20",
           "phase_750_108_10"]
FLOOR = np.array([.1, .1, .001, .1, .1, .0005])


def unpack(a):
    f = a[:, 16:-1].reshape(len(a), 12, 8).transpose(0, 2, 1)
    return np.c_[a[:, 13:16], f[:, 2]], f[:, :, YCOLS]


def inputs(x, protocols):
    n = len(x)
    p = META[protocols]
    m = np.c_[p[:, 0] / 1000, (p[:, 1] - 1) / .1, p[:, 2] / 10]
    return np.c_[np.repeat(x, len(protocols), axis=0), np.tile(m, (n, 1))]


sx, sy = unpack(SRC)
ax, ay = unpack(ALT)
rng = np.random.default_rng(777)
order = rng.permutation(len(SRC))
test, train = order[:12], order[12:]
available = np.array([0, 1, 3, 4])
target = np.array([5, 6, 7])
xt = inputs(sx[train], available)
yt = sy[train][:, available].reshape(-1, 6)
floor_scale = sy[test][:, target].std(axis=0)
active = floor_scale >= FLOOR


def score(pred, truth):
    err = np.abs(pred - truth)
    scaled = err / np.maximum(floor_scale, FLOOR)
    per = (scaled * active).sum(axis=(0, 2)) / (len(pred) * active.sum(axis=1))
    return {"mean": float(per.mean()), "per_target": per.tolist(),
            "physical_MAE": err.mean(axis=0).tolist()}


out = {"n_train_cells": len(train), "n_test_cells": len(test),
       "training_protocols": available.tolist(), "target_protocols": target.tolist(),
       "targets": TARGETS,
       "note": "Direct model is never trained on target-protocol outcomes. Its load input includes onset, amplitude, and hold duration; hold=20 ms and onset=750 ms are outside training support. Native ODE parameters were inferred from a source-trained inverse model; comparison uses the same fixed 12 test cells.",
       "summary": {}}
native_data = np.loadtxt(ROOT / "feedback_13param_native_predictions.csv", delimiter=",")
assert native_data.shape == (24, 22)
for name, scenario, x, y in [("same_model", 0, sx, sy),
                             ("discrepancy", 1, ax, ay)]:
    pred = fit_predict(xt, yt, inputs(x[test], target)).reshape(len(test), 3, 6)
    truth = y[test][:, target]
    native = native_data[native_data[:, 1] == scenario, 4:22].reshape(len(test), 3, 6)
    out["summary"][name] = {"direct_unseen_protocol": score(pred, truth),
                            "native_ODE": score(native, truth)}

path = ROOT / "feedback_13param_unseen_protocol_score.json"
path.write_text(json.dumps(out, indent=2), encoding="utf-8")
for k, v in out["summary"].items():
    print(k, "direct", v["direct_unseen_protocol"]["mean"],
          "native", v["native_ODE"]["mean"])
print(path)
