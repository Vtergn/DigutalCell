"""Nested, held-out-cell mechanical protocol selection on the 13-parameter cohort."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from score_human_feedback_observability import fit_predict

ROOT = Path(__file__).resolve().parent / "human_bidirectional_output"
A = np.loadtxt(ROOT / "feedback_13param_cohort.csv", delimiter=",")
assert A.shape[1] == 113
A = A[A[:, -1] == 1]
assert np.isfinite(A).all()

THETA = A[:, :13]
BASE = A[:, 13:16]
FEATURE = A[:, 16:-1].reshape(len(A), 12, 8).transpose(0, 2, 1)
YCOLS = [0, 2, 3, 6, 8, 10]
Y = FEATURE[:, 5:8, :][:, :, YCOLS]
PARAM_NAMES = ["CaL", "SERCA", "NCX", "IKr", "IK1", "INaL",
               "g_ionic", "ca50", "TRPN_n", "nperm", "beta0", "beta1", "Tref"]
PARAM_RANGE = np.array([.2, .3, .3, .3, .2, .4, 1.2, .11, .4, 1.0, 1.0, 3.2, 40])
PROTOCOLS = ["early150_110", "late600_110", "mid300_110",
             "early150_105", "late600_105"]
TARGETS = ["interpolation_225_1075_10ms",
           "amplitude_duration_OOD_225_1125_20ms",
           "phase_OOD_750_108_10ms"]
PHYSIO_FLOOR = np.array([.1, .1, .001, .1, .1, .0005])


def x_of(k: int) -> np.ndarray:
    return np.c_[BASE, FEATURE[:, k, :]]


def loss(pred: np.ndarray, true: np.ndarray, scale: np.ndarray) -> float:
    # Exclude endpoints whose between-cell variation is below a fixed
    # physical resolution floor. In particular, late-phase ΔAPD and T20
    # are effectively zero and must not dominate a standardized score.
    active = scale >= PHYSIO_FLOOR
    if pred.ndim == 3:
        err = np.abs(pred-true) / np.maximum(scale, PHYSIO_FLOOR)
        per_family = (err * active).sum(axis=(0, 2)) / (len(pred)*active.sum(axis=1))
        return float(per_family.mean())
    return float(np.mean((np.abs(pred-true)/np.maximum(scale, PHYSIO_FLOOR))[:, active]))


records = []
for seed in range(30):
    rng = np.random.default_rng(90000 + seed)
    order = rng.permutation(len(A))
    ntest = max(10, round(.2 * len(A)))
    nval = max(10, round(.2 * len(A)))
    test, val, train = order[:ntest], order[ntest:ntest+nval], order[ntest+nval:]
    scales = np.maximum(Y[train].std(axis=0), 1e-6)
    # The design loop sees only train and validation *cells*. Their target
    # challenge outcomes are available in the synthetic design simulator.
    val_loss = []
    for k in range(5):
        pred = fit_predict(x_of(k)[train], Y[train].reshape(len(train), -1),
                           x_of(k)[val]).reshape(len(val), 3, 6)
        val_loss.append(loss(pred, Y[val], scales))
    choice = int(np.argmin(val_loss))
    record = {"seed": seed, "selected_protocol": PROTOCOLS[choice],
              "validation_loss": val_loss, "n_train": len(train),
              "n_val": len(val), "n_test": len(test)}
    # Refit on design + validation cells only after selecting the protocol.
    dev = np.r_[train, val]
    for key, k in [("selected", choice), ("fixed_early", 0),
                   ("fixed_late", 1), ("fixed_mid", 2),
                   ("balanced_random", int(rng.integers(5)))]:
        pred = fit_predict(x_of(k)[dev], Y[dev].reshape(len(dev), -1),
                           x_of(k)[test]).reshape(len(test), 3, 6)
        tp = fit_predict(x_of(k)[dev], THETA[dev], x_of(k)[test])
        record[key] = {
            "overall_target": loss(pred, Y[test], scales),
            "target_by_family": [loss(pred[:, j], Y[test, j], scales[j])
                                 for j in range(3)],
            "parameter_normalized": float(np.mean(np.abs(tp-THETA[test])/PARAM_RANGE)),
            "g_MAE": float(np.mean(np.abs(tp[:, 6]-THETA[test, 6]))),
            "parameter_MAE": np.abs(tp-THETA[test]).mean(axis=0).tolist(),
        }
    # Static baseline, trained on the same development cells.
    base_pred = fit_predict(BASE[dev], Y[dev].reshape(len(dev), -1),
                            BASE[test]).reshape(len(test), 3, 6)
    base_theta = fit_predict(BASE[dev], THETA[dev], BASE[test])
    record["baseline"] = {
        "overall_target": loss(base_pred, Y[test], scales),
        "target_by_family": [loss(base_pred[:, j], Y[test, j], scales[j])
                             for j in range(3)],
        "parameter_normalized": float(np.mean(
            np.abs(base_theta-THETA[test])/PARAM_RANGE)),
        "g_MAE": float(np.mean(np.abs(base_theta[:, 6]-THETA[test, 6]))),
        "parameter_MAE": np.abs(base_theta-THETA[test]).mean(axis=0).tolist(),
    }
    records.append(record)

summary = {}
for key in ["baseline", "selected", "fixed_early", "fixed_late",
            "fixed_mid", "balanced_random"]:
    vals = np.array([r[key]["overall_target"] for r in records])
    per = np.array([r[key]["target_by_family"] for r in records])
    summary[key] = {
        "mean": float(vals.mean()), "median": float(np.median(vals)),
        "per_target_mean": dict(zip(TARGETS, per.mean(axis=0).tolist())),
    }
    if "parameter_normalized" in records[0][key]:
        summary[key]["parameter_normalized_mean"] = float(np.mean(
            [r[key]["parameter_normalized"] for r in records]))
        summary[key]["g_MAE_mean"] = float(np.mean(
            [r[key]["g_MAE"] for r in records]))
        summary[key]["parameter_MAE_mean"] = dict(zip(PARAM_NAMES,
            np.mean([r[key]["parameter_MAE"] for r in records],axis=0).tolist()))

counts = {name: sum(r["selected_protocol"] == name for r in records)
          for name in PROTOCOLS}
out = {
    "n_successful_cells": len(A), "protocols": PROTOCOLS, "targets": TARGETS,
    "selection_counts": counts, "summary": summary, "splits": records,
    "endpoint_floor": PHYSIO_FLOOR.tolist(),
    "limitations": [
        "All cells and protocols are generated by the same equation family.",
        "Target outcomes for development cells are used during protocol design and supervised training.",
        "Target protocols are outside calibration amplitude/phase for some tests, but are not unseen to the population learner.",
        "Thirty repeated splits are correlated, not thirty independent cohorts.",
    ],
}
path = ROOT / "feedback_13param_design_score.json"
path.write_text(json.dumps(out, indent=2), encoding="utf-8")
print("n", len(A), "selection", counts)
for key, item in summary.items():
    print(key, round(item["mean"], 4),
          [round(q, 4) for q in item["per_target_mean"].values()])
print(path)
