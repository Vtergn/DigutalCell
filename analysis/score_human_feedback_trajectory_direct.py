"""Cell-held-out and protocol-held-out trajectory-conditioned direct baselines.

The direct predictors get the entire imposed length trace and the entire
calibration AP/Ca/tension trace. They train on five mechanical protocols for
52 source-model cells, then face three wholly withheld target protocols in 12
other cells. Hyperparameter/model selection only uses grouped inner CV on the
52 development cells. This is a direct supervised baseline, not a simulator.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


ROOT = Path(__file__).resolve().parent / "human_bidirectional_output"
CSV = np.loadtxt(ROOT / "feedback_13param_cohort.csv", delimiter=",")
with h5py.File(ROOT / "feedback_13param_cohort.mat") as f:
    W = f["waveforms"][:]  # cell, protocol, channel, time
    ONSET = f["onsets"][:].ravel().astype(int)
    LAM = f["lambdas"][:].ravel()
    HOLD = f["holds"][:].ravel().astype(int)
assert CSV.shape == (64, 113) and W.shape == (64, 8, 3, 1001)
assert np.all(CSV[:, -1] == 1)
F = CSV[:, 16:-1].reshape(64, 12, 8).transpose(0, 2, 1)
Y = F[:, :, [0, 2, 3, 6, 8, 10]]
BASE = CSV[:, 13:16]
FLOOR = np.array([.1, .1, .001, .1, .1, .0005])
CAL = 2  # 300 ms, lambda 1.10
DEV_PROTOCOL = np.arange(5)
TEST_PROTOCOL = np.arange(5, 8)
rng = np.random.default_rng(777)
order = rng.permutation(64)
TEST_CELL, TRAIN_CELL = order[:12], order[12:]


def length_trace(j):
    x = np.ones(1001)
    onset, end, amp = ONSET[j], ONSET[j] + HOLD[j] + 2, LAM[j] - 1
    x[onset:onset+2] = 1 + amp * np.arange(2)
    x[onset+1:onset+HOLD[j]+1] = 1 + amp
    x[onset+HOLD[j]+1:end+1] = 1 + amp * (end - np.arange(onset+HOLD[j]+1, end+1))
    return x


LOAD = np.stack([length_trace(j) for j in range(8)])
assert np.allclose(LOAD.max(axis=1), LAM)


def compact_features(cell, protocol):
    """Local electrical context plus integral and full history of length input."""
    j = np.asarray(protocol)
    c = np.asarray(cell)
    calwave = W[c, CAL].astype(float)
    local_offsets = np.array([-75, -25, 0, 25, 75])
    loc = np.clip(ONSET[j, None] + local_offsets, 0, 1000)
    context = np.take_along_axis(calwave, loc[:, None, :], axis=2)
    # Per-cell known baseline, 12 calibration features, phase-local AP/Ca/T,
    # length-trajectory integrals, and protocol geometry.
    strain = LOAD[j] - 1
    geom = np.c_[ONSET[j] / 1000, (LAM[j]-1) / .1, HOLD[j] / 10,
                 strain.sum(axis=1) / 10, (strain**2).sum(axis=1) / .1]
    return np.c_[BASE[c], F[c, CAL], context.reshape(len(c), -1), geom]


def long_features(cell, protocol):
    c = np.asarray(cell)
    j = np.asarray(protocol)
    compact = compact_features(c, j)
    # Full 1-ms mechanical history and 10-ms multimodal calibration waveform.
    # The trajectory features are kept as their own block for kernel weighting.
    waveform = W[c, CAL, :, ::10].astype(float).reshape(len(c), -1)
    trajectory = (LOAD[j] - 1) / .1
    return compact, waveform, trajectory


def rows(cells, protocols):
    c = np.repeat(cells, len(protocols))
    j = np.tile(protocols, len(cells))
    return c, j


def score(pred, truth, scale, active):
    error = np.abs(pred-truth) / np.maximum(scale, FLOOR)
    return float(np.mean(error[:, active]))


def predict(kind, xtrain, ytrain, xtest, alpha, pca_dim=0):
    cp, wave, mech = xtrain
    cq, wavq, mechq = xtest
    sc = StandardScaler().fit(cp)
    ctrain, ctest = sc.transform(cp), sc.transform(cq)
    if kind == "poly2":
        poly = PolynomialFeatures(2, include_bias=False)
        a = poly.fit_transform(ctrain)
        b = poly.transform(ctest)
        scaler = StandardScaler().fit(a)
        a, b = scaler.transform(a), scaler.transform(b)
        return Ridge(alpha=alpha).fit(a, ytrain).predict(b)
    if kind == "ridge_full":
        ws = StandardScaler().fit(wave)
        pw = PCA(n_components=pca_dim, whiten=True, random_state=1).fit(ws.transform(wave))
        wtrain = pw.transform(ws.transform(wave))
        wtest = pw.transform(ws.transform(wavq))
        # The full 1001-sample trace is encoded by fixed smooth time bins.
        mt = mech[:, :1000].reshape(len(mech), -1, 20).mean(axis=2)
        mq = mechq[:, :1000].reshape(len(mechq), -1, 20).mean(axis=2)
        ms = StandardScaler().fit(mt)
        mtrain, mtest = ms.transform(mt), ms.transform(mq)
        rawscale = StandardScaler().fit(mech)
        rawtrain, rawtest = rawscale.transform(mech), rawscale.transform(mechq)
        # Preserve the entire 1-ms trajectory and a smooth time-bin projection.
        # Each functional block is normalized so its dimensionality alone does
        # not determine ridge regularization.
        a = np.c_[ctrain/np.sqrt(ctrain.shape[1]),
                  wtrain/np.sqrt(wtrain.shape[1]),
                  mtrain/np.sqrt(mtrain.shape[1]),
                  rawtrain/np.sqrt(rawtrain.shape[1])]
        b = np.c_[ctest/np.sqrt(ctest.shape[1]),
                  wtest/np.sqrt(wtest.shape[1]),
                  mtest/np.sqrt(mtest.shape[1]),
                  rawtest/np.sqrt(rawtest.shape[1])]
        return Ridge(alpha=alpha).fit(a, ytrain).predict(b)
    if kind == "rbf_functional":
        ws = StandardScaler().fit(wave)
        pw = PCA(n_components=pca_dim, whiten=True, random_state=1).fit(ws.transform(wave))
        wt, wq = pw.transform(ws.transform(wave)), pw.transform(ws.transform(wavq))
        # Multi-kernel product: calibration cell similarity times mechanical
        # history similarity. Timing/length summary stays in compact features.
        cell_train = np.c_[ctrain[:, :15], wt]
        cell_test = np.c_[ctest[:, :15], wq]
        cell_scale = np.median(cdist(cell_train, cell_train, "sqeuclidean"))
        mech_scale = np.median(cdist(mech, mech, "sqeuclidean"))
        kc = np.exp(-cdist(cell_train, cell_train, "sqeuclidean") / max(cell_scale, 1e-8))
        km = np.exp(-cdist(mech, mech, "sqeuclidean") / max(mech_scale, 1e-8))
        k = kc * km
        q = np.exp(-cdist(cell_test, cell_train, "sqeuclidean") / max(cell_scale, 1e-8))
        q *= np.exp(-cdist(mechq, mech, "sqeuclidean") / max(mech_scale, 1e-8))
        mean = ytrain.mean(axis=0)
        return mean + q @ np.linalg.solve(k + alpha*np.eye(len(k)), ytrain-mean)
    raise ValueError(kind)


def get_data(cells, protocols):
    c, j = rows(cells, protocols)
    return long_features(c, j), Y[c, j], c


models = [("poly2", alpha, 0) for alpha in [.1, 10, 100]]
models += [("ridge_full", alpha, dim) for dim in [5, 12] for alpha in [.1, 10, 100]]
models += [("rbf_functional", alpha, dim) for dim in [5, 12] for alpha in [.02, .2, 2]]
full_scale = Y[TRAIN_CELL][:, DEV_PROTOCOL].reshape(-1, 6).std(axis=0)
active = full_scale >= FLOOR
cv = GroupKFold(n_splits=4)
cv_results = []
for kind, alpha, dim in models:
    losses, protocol_losses = [], []
    for tr_i, va_i in cv.split(TRAIN_CELL, groups=TRAIN_CELL):
        xtr, ytr, _ = get_data(TRAIN_CELL[tr_i], DEV_PROTOCOL)
        xva, yva, _ = get_data(TRAIN_CELL[va_i], DEV_PROTOCOL)
        p = predict(kind, xtr, ytr, xva, alpha, dim)
        losses.append(score(p, yva, full_scale, active))
    # A second, task-matched validation: hold out an entire source protocol
    # *and* disjoint cells. These are development protocols only; 5:8 remain
    # sealed until final evaluation.
    inner = GroupKFold(n_splits=2)
    for tr_i, va_i in inner.split(TRAIN_CELL, groups=TRAIN_CELL):
        for withheld in [0, 1, 4]:
            remain = DEV_PROTOCOL[DEV_PROTOCOL != withheld]
            xtr, ytr, _ = get_data(TRAIN_CELL[tr_i], remain)
            xva, yva, _ = get_data(TRAIN_CELL[va_i], [withheld])
            p = predict(kind, xtr, ytr, xva, alpha, dim)
            protocol_losses.append(score(p, yva, full_scale, active))
    cv_results.append({"kind": kind, "alpha": alpha, "pca_dim": dim,
                       "validation_mean": float(np.mean(losses)), "folds": losses,
                       "protocol_validation_mean": float(np.mean(protocol_losses)),
                       "protocol_folds": protocol_losses})
best = min(cv_results, key=lambda z: z["protocol_validation_mean"])
xtr, ytr, _ = get_data(TRAIN_CELL, DEV_PROTOCOL)
xte, yte, _ = get_data(TEST_CELL, TEST_PROTOCOL)
pred = predict(best["kind"], xtr, ytr, xte, best["alpha"], best["pca_dim"])
test_scale = Y[TEST_CELL][:, TEST_PROTOCOL].std(axis=0)
test_active = test_scale >= FLOOR
pred, yte = pred.reshape(12, 3, 6), yte.reshape(12, 3, 6)
native = np.loadtxt(ROOT / "feedback_13param_native_predictions.csv", delimiter=",")
native = native[native[:, 1] == 0, 4:22].reshape(12, 3, 6)


def test_summary(z):
    per = [score(z[:, j], yte[:, j], test_scale[j], test_active[j])
           for j in range(3)]
    return {"mean": float(np.mean(per)), "per_protocol": per,
            "physical_MAE": np.abs(z-yte).mean(axis=0).tolist()}


out = {"train_cells": len(TRAIN_CELL), "test_cells": len(TEST_CELL),
       "train_protocols": DEV_PROTOCOL.tolist(),
       "test_protocols": TEST_PROTOCOL.tolist(),
       "selection": "Grouped 4-fold cell CV plus 2x3 disjoint-cell/leave-one-development-protocol-out CV. Select by the latter; all three final target protocols withheld.",
       "model_grid": cv_results, "selected": best,
       "direct_trajectory": test_summary(pred), "native_ODE": test_summary(native),
       "caveat": "Strongest among a prespecified small direct-model grid, not a universal upper bound for trajectory neural operators. Only one 12-cell final test split."}
out["family_results"] = {}
for family in sorted({m["kind"] for m in cv_results}):
    winner = min((m for m in cv_results if m["kind"] == family),
                 key=lambda z: z["protocol_validation_mean"])
    fp = predict(family, xtr, ytr, xte,
                 winner["alpha"], winner["pca_dim"]).reshape(12, 3, 6)
    out["family_results"][family] = {"selected": winner,
                                     "test": test_summary(fp)}
forms_csv = ROOT / "feedback_13param_two_forms.csv"
forms_mat = ROOT / "feedback_13param_two_forms_waveforms.mat"
if forms_csv.exists() and forms_mat.exists():
    forms = np.loadtxt(forms_csv, delimiter=",")
    with h5py.File(forms_mat) as h:
        alternate_wave = h["calibration_waveforms"][:].transpose(2, 1, 0)
    assert forms.shape == (24, 66) and alternate_wave.shape == (24, 3, 1001)
    out["cross_form"] = {}
    source_base, source_f, source_w = BASE.copy(), F.copy(), W.copy()
    for label, block_slice in [("saturating_SAC", slice(0, 12)),
                               ("reduced_TnC_buffer", slice(12, 24))]:
        block = forms[block_slice]
        ids = block[:, 0].astype(int)-1
        assert np.array_equal(ids, TEST_CELL) and np.all(block[:, -1] == 1)
        feat = block[:, 17:-1].reshape(12, 4, 12)
        BASE = source_base.copy()
        F = source_f.copy()
        W = source_w.copy()
        BASE[ids] = block[:, 14:17]
        F[ids, CAL] = feat[:, 0]
        W[ids, CAL] = alternate_wave[block_slice]
        c, j = rows(TEST_CELL, TEST_PROTOCOL)
        xalt = long_features(c, j)
        alt_pred = predict(best["kind"], xtr, ytr, xalt,
                           best["alpha"], best["pca_dim"]).reshape(12, 3, 6)
        alt_truth = feat[:, 1:, :][:, :, [0, 2, 3, 6, 8, 10]]
        # The metric's physical scale and active endpoints are frozen from
        # the source-model test split for cross-form comparability.
        per = [score(alt_pred[:, k], alt_truth[:, k],
                     test_scale[k], test_active[k]) for k in range(3)]
        out["cross_form"][label] = {"mean": float(np.mean(per)),
                                   "per_protocol": per}
    BASE, F, W = source_base, source_f, source_w
path = ROOT / "feedback_13param_trajectory_direct_score.json"
path.write_text(json.dumps(out, indent=2), encoding="utf-8")
print("selected", best)
print("direct", out["direct_trajectory"]["mean"], out["direct_trajectory"]["per_protocol"])
print("native", out["native_ODE"]["mean"], out["native_ODE"]["per_protocol"])
for family, result in out["family_results"].items():
    print(family, result["test"]["mean"], result["test"]["per_protocol"])
for label, result in out.get("cross_form", {}).items():
    print(label, "direct trajectory", result["mean"], result["per_protocol"])
