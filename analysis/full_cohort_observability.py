"""Frozen-split observability audit for the merged virtual cardiomyocyte cohort."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cohort", type=Path)
    args = parser.parse_args()
    with h5py.File(args.cohort) as f:
        v = f["V"][:, :601:5]
        ca = f["Cai_uM"][:, :601:5]
        lam = f["lambda"][()]
        t = f["Ta_kPa"][:, :, :601:5]
        theta = f["theta"][()]
        lo, hi = f["lo"][()], f["hi"][()]
        split = f["split_code"][()]
        shards = json.loads(f.attrs["shards_json"])
    idx = [int(np.argmin(abs(lam - x))) for x in [1.0, 1.1]]
    assert np.allclose(lam[idx], [1.0, 1.1])
    y = (theta - lo) / (hi - lo)
    tr, va, te = (np.flatnonzero(split == k) for k in range(3))
    designs = {
        "AP": [v],
        "AP+Ca": [v, ca],
        "AP+Ca+T1.0": [v, ca, t[:, idx[0]]],
        "AP+Ca+T1.0+T1.1": [v, ca, t[:, idx[0]], t[:, idx[1]]],
    }
    result = {"n_cells": len(y), "train_n": len(tr), "validation_n": len(va),
              "test_n": len(te), "shards": shards, "modalities": {}}
    errors = {}
    mechanical_errors = {}
    for label, parts in designs.items():
        x = np.concatenate(parts, axis=1)
        scores = []
        for alpha in [1., 10., 100., 1000.]:
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(x[tr], y[tr])
            scores.append(np.mean(abs(model.predict(x[va]) - y[va])))
        alpha = [1., 10., 100., 1000.][int(np.argmin(scores))]
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(x[np.r_[tr, va]], y[np.r_[tr, va]])
        pred = model.predict(x[te])
        err = abs(pred - y[te])
        errors[label] = err.mean(axis=1)
        mechanical_errors[label] = err[:, 4:].mean(axis=1)
        result["modalities"][label] = {
            "validation_selected_alpha": alpha,
            "test_all_parameter_MAE_prior_scaled": float(err.mean()),
            "test_electrical_MAE_prior_scaled": float(err[:, :4].mean()),
            "test_mechanical_MAE_prior_scaled": float(err[:, 4:].mean()),
            "test_beta0_R2": float(r2_score(y[te, 8], pred[:, 8])),
            "test_beta1_R2": float(r2_score(y[te, 9], pred[:, 9])),
            "test_beta0_MAE_prior_scaled": float(err[:, 8].mean()),
            "test_beta1_MAE_prior_scaled": float(err[:, 9].mean()),
        }
    rng = np.random.default_rng(20260928)
    base = errors["AP+Ca"]
    for label in ["AP+Ca+T1.0", "AP+Ca+T1.0+T1.1"]:
        difference = base - errors[label]
        bootstrap = rng.integers(0, len(te), (3000, len(te)))
        result["modalities"][label]["all_parameter_MAE_reduction_vs_AP_Ca"] = float(difference.mean())
        result["modalities"][label]["paired_bootstrap_95pct_CI"] = np.quantile(
            difference[bootstrap].mean(axis=1), [.025, .975]).tolist()
        mechanical_delta = mechanical_errors["AP+Ca"] - mechanical_errors[label]
        result["modalities"][label]["mechanical_MAE_reduction_vs_AP_Ca"] = float(mechanical_delta.mean())
        result["modalities"][label]["mechanical_paired_bootstrap_95pct_CI"] = np.quantile(
            mechanical_delta[bootstrap].mean(axis=1), [.025, .975]).tolist()
    result["active_peak_kPa"] = {
        str(float(lam[j])): np.quantile(t[:, j].max(axis=1), [.05, .5, .95]).tolist()
        for j in idx
    }
    output = args.cohort.parent / "full_cohort_observability.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "shards"}, indent=2))


if __name__ == "__main__":
    main()
