"""Export a fair ridge-initialized, true-equation point-twin benchmark.

Fits only train/validation cells, exports exactly the already-sealed 16 test
cell IDs used for posterior true-forward evaluation. Parameter values are
clipped to the benchmark prior bounds before running the ODE.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "analysis/virtual_cardiomyocyte_5000/merged_cohort.h5"
IDS = ROOT / "analysis/virtual_cardiomyocyte_5000/posterior_draws_16x8.json"
OUTPUT = ROOT / "analysis/virtual_cardiomyocyte_5000/ridge_twin_16.csv"


def main():
    selected = json.loads(IDS.read_text())["test_cell_indices_zero_based"]
    with h5py.File(COHORT) as f:
        lam = f["lambda"][()]
        ids = [int(np.argmin(abs(lam - x))) for x in [1., 1.1]]
        assert np.allclose(lam[ids], [1., 1.1])
        v = f["V"][:, :601:5]
        ca = f["Cai_uM"][:, :601:5]
        ta = f["Ta_kPa"][:, ids, :601:5]
        theta = f["theta"][()]
        lo, hi = f["lo"][()], f["hi"][()]
        split = f["split_code"][()]
    assert np.all(split[selected] == 2)
    y = (theta-lo)/(hi-lo)
    x = np.concatenate([v, ca, ta[:,0], ta[:,1]], axis=1)
    fit_ids = np.flatnonzero(split != 2)
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.))
    model.fit(x[fit_ids], y[fit_ids])
    pred_unit_raw = model.predict(x[selected])
    pred_unit = np.clip(pred_unit_raw, 0, 1)
    pred_theta = lo + pred_unit*(hi-lo)
    OUTPUT.parent.mkdir(exist_ok=True)
    with OUTPUT.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cell_index_zero_based","mode","draw_index"] +
                        [f"theta_{i+1}" for i in range(12)])
        for cell, values in zip(selected, pred_theta):
            writer.writerow([cell,3,0] + values.tolist())
    diagnostics = {
        "cohort": str(COHORT), "selected_test_cells": selected,
        "alpha": 1., "fit_n": len(fit_ids),
        "test_parameter_MAE_prior_scaled": float(abs(pred_unit-y[selected]).mean()),
        "test_mechanical_MAE_prior_scaled": float(abs(pred_unit[:,4:]-y[selected,4:]).mean()),
        "n_clipped_coordinates": int(np.count_nonzero(pred_unit != pred_unit_raw)),
        "scope": "Ridge-initialized point twin; parameters sent through true ODE at sealed target load.",
    }
    OUTPUT.with_suffix(".json").write_text(json.dumps(diagnostics,indent=2),encoding="utf-8")
    print(json.dumps(diagnostics,indent=2))


if __name__ == "__main__":
    main()
