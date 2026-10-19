"""Frozen small-grid reverse MCT challenge benchmark (see protocol MD)."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import RBFInterpolator

from score_mct2024_reference import beats


ROOT = Path(__file__).resolve().parent / "mct_feedback_audit"
DATA = ROOT / "reverse_cohort"
FIELDS = ("APD_above_minus67_mV_ms", "Ca_peak_uM")
DESCRIPTIVE_FIELDS = ("gel_stress_peak_kPa", "pCa_multiplier_peak", "force_peak_kPa")
LOADS = ("l02", "l06", "l10")


def read_case(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path) as f:
        return {k: f["a"][k][()].ravel() for k in
                ("t_ms", "Vm_mV", "Cai_uM", "t_mech_ms", "Lneg", "force", "gelForce", "pCaCoeff")}


def ridge_loo(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    pred = np.empty_like(y)
    for i in range(len(y)):
        train = np.arange(len(y)) != i
        xmean, ymean = x[train].mean(0), y[train].mean(0)
        xsd = np.maximum(x[train].std(0), 1e-9)
        ysd = np.maximum(y[train].std(0), 1e-9)
        xx, yy = (x[train]-xmean)/xsd, (y[train]-ymean)/ysd
        w = np.linalg.solve(xx.T @ xx + np.eye(x.shape[1]), xx.T @ yy)
        pred[i] = ymean + ysd * (((x[i]-xmean)/xsd) @ w)
    return pred


def rbf_loo(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Post-score exploratory nonlinear direct baseline, no parameter states."""
    pred = np.empty_like(y)
    for i in range(len(y)):
        train = np.arange(len(y)) != i
        xmean, ymean = x[train].mean(0), y[train].mean(0)
        xsd = np.maximum(x[train].std(0), 1e-9)
        ysd = np.maximum(y[train].std(0), 1e-9)
        model = RBFInterpolator((x[train]-xmean)/xsd, (y[train]-ymean)/ysd,
                                kernel="thin_plate_spline", smoothing=1e-6)
        pred[i] = ymean + ysd * model(((x[i]-xmean)/xsd)[None, :])[0]
    return pred


def errors(y: np.ndarray, pred: np.ndarray) -> dict:
    abs_err = np.abs(y-pred)
    return {"MAE": dict(zip(FIELDS, map(float, abs_err.mean(0)))),
            "median_AE": dict(zip(FIELDS, map(float, np.median(abs_err, axis=0))))}


def main() -> None:
    rows = []
    for gain in (0.25, 0.50, 0.75, 1.00, 1.25):
        for scale in (0.90, 0.95, 1.00, 1.05, 1.10):
            ident = f"g{round(100*gain):03d}_p{round(100*scale):03d}"
            row = {"id": ident, "mct_gain": gain, "pCa0_scale": scale}
            for load in LOADS:
                path = DATA / f"{ident}_{load}.mat"
                if not path.exists():
                    raise FileNotFoundError(path)
                scored = beats(read_case(path))
                row[load] = {k: float(scored[-1][k]) for k in FIELDS + DESCRIPTIVE_FIELDS}
                row[f"{load}_drift_17_to_20"] = {k: float(scored[-1][k]-scored[0][k]) for k in FIELDS}
            rows.append(row)
    values = {load: np.array([[r[load][f] for f in FIELDS] for r in rows]) for load in LOADS}
    low, mid, high = (values[x] for x in LOADS)
    predictors = {
        "low_no_change": low,
        "mid_no_change": mid,
        "low_ridge_LOO": ridge_loo(low, high),
        "low_mid_ridge_LOO": ridge_loo(np.concatenate((low, mid), axis=1), high),
        "per_variant_linear_extrapolation": 2*mid-low,
        "low_RBF_LOO_postscore": rbf_loo(low, high),
        "low_mid_RBF_LOO_postscore": rbf_loo(np.concatenate((low, mid), axis=1), high),
    }
    scores = {name: errors(high, pred) for name, pred in predictors.items()}
    # Compare only models allowed the corresponding information. Counts are
    # descriptive across the designed grid, not statistical replicates.
    paired = {}
    a = np.abs(high-predictors["low_ridge_LOO"])
    b = np.abs(high-predictors["low_mid_ridge_LOO"])
    for k, field in enumerate(FIELDS):
        paired[field] = {"low_mid_ridge_better": int(np.sum(b[:, k] < a[:, k])),
                         "tied": int(np.sum(np.isclose(b[:, k], a[:, k]))),
                         "worse": int(np.sum(b[:, k] > a[:, k])),
                         "mean_MAE_reduction": float(np.mean(a[:, k]-b[:, k]))}
    max_drift = {field: float(max(abs(r[f"{load}_drift_17_to_20"][field])
                                  for r in rows for load in LOADS)) for field in FIELDS}
    result = {
        "n_designed_variants": len(rows),
        "domain": "rabbit MCT2024 synthetic parameter grid; not experimental cells",
        "loads_CL": [0.02, 0.06, 0.10],
        "split": "leave-one-variant-out; held-out high load never used as calibration input",
        "scores": scores,
        "paired_ridge_comparison": paired,
        "maximum_absolute_beat17_to20_change": max_drift,
        "rows": rows,
        "predictions": {name: pred.tolist() for name, pred in predictors.items()},
        "interpretation_boundary": "25 fixed grid points and 20-beat transient protocol; no population generalization or individual mechanistic inversion established.",
    }
    (ROOT / "reverse_grid25_score.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"scores": scores, "paired_ridge_comparison": paired,
                      "maximum_absolute_beat17_to20_change": max_drift}, indent=2))


if __name__ == "__main__":
    main()
