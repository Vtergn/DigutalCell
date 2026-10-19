"""Score source and structural-mismatch 13-parameter native ODE closure."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent / "human_bidirectional_output"
inp = np.loadtxt(ROOT / "feedback_13param_native_input.csv", delimiter=",")
native = np.loadtxt(ROOT / "feedback_13param_native_predictions.csv", delimiter=",")
assert inp.shape == (24, 64) and native.shape == (24, 22)
assert np.allclose(inp[:, :2], native[:, :2])
assert np.all(native[:, 3] == 1) and np.isfinite(native).all()

floor = np.array([.1, .1, .001, .1, .1, .0005])
source_truth = inp[inp[:, 1] == 0, 28:46].reshape(12, 3, 6)
scale = source_truth.std(axis=0)
active = scale >= floor


def error(pred, truth):
    e = np.abs(pred-truth) / np.maximum(scale, floor)
    per = (e*active).sum(axis=(0, 2)) / (len(pred)*active.sum(axis=1))
    return {"mean": float(per.mean()), "per_target": per.tolist(),
            "physical_MAE": np.abs(pred-truth).mean(axis=0).tolist()}


summary = {}
for name, scenario in [("same_model", 0), ("discrepancy", 1)]:
    mask = inp[:, 1] == scenario
    truth = inp[mask, 28:46].reshape(12, 3, 6)
    direct = inp[mask, 46:64].reshape(12, 3, 6)
    forward = native[mask, 4:22].reshape(12, 3, 6)
    summary[name] = {
        "native_ODE": error(forward, truth),
        "direct_RBF": error(direct, truth),
        "parameter_normalized_MAE": float(np.mean(
            np.abs(inp[mask, 15:28] - inp[mask, 2:15]) /
            np.array([.2, .3, .3, .3, .2, .4, 1.2, .11, .4, 1, 1, 3.2, 40]))),
    }

out = {
    "n_test_cells": 12,
    "calibration": "Fixed mid-phase 300 ms, lambda 1.10 mechanical pulse plus resting AP/Ca/T.",
    "targets": ["225ms_1075_10ms", "225ms_1125_20ms", "750ms_108_10ms"],
    "population_supervision": "Direct RBF and inverse RBF train on source-model target outputs of 52 other cells; target protocols therefore are not unseen to the population learner.",
    "discrepancy": "True generator stretch-gain exponent 0.5 vs inference/forward-model exponent 0.",
    "endpoint_floor": floor.tolist(),
    "summary": summary,
}
path = ROOT / "feedback_13param_native_score.json"
path.write_text(json.dumps(out, indent=2), encoding="utf-8")
for key, item in summary.items():
    print(key, "native", item["native_ODE"]["mean"],
          "direct", item["direct_RBF"]["mean"],
          "param", item["parameter_normalized_MAE"])
print(path)
