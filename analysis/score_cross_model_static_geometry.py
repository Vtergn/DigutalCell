"""Frozen static-load functional geometry test across two human model families."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.spatial.distance import cdist
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneOut

ROOT = Path(__file__).resolve().parent
A_PATH = ROOT / "human_bidirectional_output/feedback_static_cohort_checkpoint.mat"
B_PATH = ROOT / "mechchem_cross_model/output/cohort_checkpoint.mat"
OUT = ROOT / "mechchem_cross_model/output/cross_model_static_score.json"

FLOOR = np.array([1, .002, .2, .5, .5, .002, .002, .002,
                  .2, .2, .2, .002, .2, .2], float)
EC = np.array([0, 1, 3, 4, 5, 6, 7, 11])
CA = np.array([1, 5, 6, 7, 11])
FULL = np.arange(14)


def clean_features(path: Path):
    obj = loadmat(path)
    f = np.asarray(obj["features"], float)
    # A model occasionally fails to repolarize within the observed beat.
    # This is a censored APD, not a missing simulation.
    censored = np.isnan(f[:, :, 0]) & np.isfinite(f[:, :, 1:]).all(axis=2)
    f[:, :, 0] = np.where(censored, 1000, f[:, :, 0])
    valid = np.isfinite(f).all(axis=(1, 2))
    return f[valid], {
        "cells_total": len(f), "cells_valid": int(valid.sum()),
        "apd_censored_per_load": censored.sum(axis=0).tolist(),
    }


def source_scaling(f, endpoints=EC):
    base = f[:, 0]
    challenge = f[:, 1] - base
    future = np.concatenate([f[:, 2, endpoints] - base[:, endpoints],
                             f[:, 3, endpoints] - base[:, endpoints]], axis=1)
    bmean = base.mean(axis=0)
    cmean = challenge.mean(axis=0)
    ymean = future.mean(axis=0)
    bsd = np.maximum(base.std(axis=0), FLOOR)
    csd = np.maximum(challenge.std(axis=0), FLOOR)
    ysd = np.maximum(future.std(axis=0), np.tile(FLOOR[endpoints], 2))
    return bmean, cmean, ymean, bsd, csd, ysd


def vectors(f, s, endpoints=EC):
    bm, cm, ym, bs, cs, ys = s
    b = (f[:, 0] - bm) / bs
    c = (f[:, 1] - f[:, 0] - cm) / cs
    y = np.concatenate([f[:, 2, endpoints] - f[:, 0, endpoints],
                        f[:, 3, endpoints] - f[:, 0, endpoints]], axis=1)
    y = (y - ym) / ys
    return b, c, y


def neighbors(b, c, y, k=5):
    future = cdist(y, y) / np.sqrt(y.shape[1])
    def per_cell(x):
        d = cdist(x, x)
        np.fill_diagonal(d, np.inf)
        ix = np.argsort(d, axis=1)[:, :min(k, len(d)-1)]
        return future[np.arange(len(d))[:, None], ix].mean(axis=1)
    baseline = per_cell(b / np.sqrt(b.shape[1]))
    challenge = per_cell(np.concatenate([b / np.sqrt(b.shape[1]),
                                          c / np.sqrt(c.shape[1])], axis=1))
    return baseline, challenge


def bootstrap_and_shuffle(b, c, y, baseline, challenge):
    rng = np.random.default_rng(20261003)
    delta = baseline - challenge
    boot = [rng.choice(delta, len(delta), replace=True).mean() for _ in range(2000)]
    shuf = []
    for _ in range(1000):
        _, cs = neighbors(b, c[rng.permutation(len(c))], y)
        shuf.append((baseline-cs).mean())
    return dict(delta_mean=float(delta.mean()),
                paired_bootstrap95=np.quantile(boot, [.025, .975]).tolist(),
                shuffled_p=float((1+sum(v >= delta.mean() for v in shuf))/(len(shuf)+1)),
                shuffled_max=float(np.max(shuf)))


def cross_prediction(a, b):
    sa = source_scaling(a)
    xa, ca, ya = vectors(a, sa)
    xb, cb, yb = vectors(b, sa)
    report = {}
    # PLS components are fixed before viewing B outcomes.
    for name, xtrain, xtest, ncomp in [
        ("baseline_3d", xa, xb, 3),
        ("challenge_5d", np.concatenate([xa, ca], axis=1),
         np.concatenate([xb, cb], axis=1), 5),
    ]:
        model = PLSRegression(n_components=ncomp, scale=False)
        model.fit(xtrain, ya)
        pred = model.predict(xtest)
        report[name] = float(np.mean(np.linalg.norm(pred-yb, axis=1)
                                     / np.sqrt(yb.shape[1])))
    report["zero_change"] = float(np.mean(np.linalg.norm(
        (0-sa[2])/sa[5]-yb, axis=1)/np.sqrt(yb.shape[1])))
    return report


def b_only_cv(b):
    sb = source_scaling(b)
    xb, cb, yb = vectors(b, sb)
    out = {}
    for name, x, nc in [("baseline_3d", xb, 3),
                        ("challenge_5d", np.concatenate([xb, cb], axis=1), 5)]:
        losses=[]
        for tr, te in LeaveOneOut().split(x):
            model=PLSRegression(n_components=nc, scale=False)
            model.fit(x[tr],yb[tr])
            pred=model.predict(x[te])
            losses.append(np.linalg.norm(pred-yb[te]) / np.sqrt(yb.shape[1]))
        out[name]=float(np.mean(losses))
    return out


def main():
    a, qa = clean_features(A_PATH)
    b, qb = clean_features(B_PATH)
    if len(b) < 12:
        raise RuntimeError(f"Need >=12 complete independent B cells, got {len(b)}")
    results={"model_A":qa,"model_B":qb,"protocol":"static length lambda 1/1.05 -> 1.075/1.10", "k":5}
    sa=source_scaling(a); sb=source_scaling(b)
    cross_scale=(sa[0],sa[1],sb[2],sa[3],sa[4],sb[5])
    for name, f, sc in [("A",a,sa),("B",b,sb),
                        ("B_using_A_observation_scaling",b,cross_scale)]:
        x,c,y=vectors(f,sc)
        bb,cc=neighbors(x,c,y)
        results[name]={"baseline_future_neighbor_distance":float(bb.mean()),
                       "challenge_future_neighbor_distance":float(cc.mean()),
                       **bootstrap_and_shuffle(x,c,y,bb,cc)}
    results["A_to_B_pls"] = cross_prediction(a,b)
    results["B_only_leave_one_cell_out_pls"] = b_only_cv(b)
    for name, f in [("A_Ca_only",a),("B_Ca_only",b)]:
        x,c,y=vectors(f,source_scaling(f,CA),CA)
        bb,cc=neighbors(x,c,y)
        results[name]={"baseline_future_neighbor_distance":float(bb.mean()),
                       "challenge_future_neighbor_distance":float(cc.mean()),
                       **bootstrap_and_shuffle(x,c,y,bb,cc)}
    for name, f in [("A_AP_Ca_T",a),("B_AP_Ca_T",b)]:
        x,c,y=vectors(f,source_scaling(f,FULL),FULL)
        bb,cc=neighbors(x,c,y)
        results[name]={"baseline_future_neighbor_distance":float(bb.mean()),
                       "challenge_future_neighbor_distance":float(cc.mean()),
                       **bootstrap_and_shuffle(x,c,y,bb,cc)}
    OUT.write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(results,indent=2,ensure_ascii=False))


if __name__ == "__main__":
    main()
