"""Audit paired virtual cells and run a small cell-held-out observability pilot.

The ridge analysis is diagnostic only. It does not constitute calibrated
posterior inference or an unseen-load predictive comparison.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
COHORT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "analysis/virtual_cardiomyocyte_96_adaptive"
OUT = COHORT / "audit.json"


def col(path: Path, key: str) -> np.ndarray:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return np.array([float(row[key]) for row in csv.DictReader(f)], dtype=float)


with h5py.File(COHORT / "cohort.mat") as f:
    ok = f["success"][()].ravel().astype(bool)
    theta = f["theta"][()].T[ok]
    lo = f["lo"][()].ravel()
    hi = f["hi"][()].ravel()
    v = f["V"][()].T[ok]
    ca = f["Cai_uM"][()].T[ok]
    ta = f["Ta_kPa"][()].transpose(2, 1, 0)[ok]
    lam = f["lambda"][()].ravel()
    cycle = f["calcium_cycle_drift"][()].ravel()[ok]
    land_cycle = f["land_cycle_drift"][()].T[ok]
    beats = f["electrical_beats_used"][()].ravel()[ok]
    split = f["split_code"][()].ravel()[ok] if "split_code" in f else None

names = [
    "ICaL", "Jup", "INaCa", "IKr", "Tref", "ca50", "TRPN_n",
    "nperm", "beta0", "beta1", "phi", "perm50",
]
assert theta.shape[1] == len(names)
assert ta.shape[1] == len(lam)
idx = {round(float(x), 3): i for i, x in enumerate(lam)}
assert 1.0 in idx and 1.1 in idx

# The source code's beta terms vanish exactly at lambda=1: Lfac=1 and
# ca50 shift=0. They must be excluded from claimed single-load recovery.
params01 = (theta - lo) / (hi - lo)
keep = np.arange(0, 601, 5)
modules = {
    "AP": [v[:, keep]],
    "AP+Ca": [v[:, keep], ca[:, keep]],
    "AP+Ca+T1.0": [v[:, keep], ca[:, keep], ta[:, idx[1.0], keep]],
    "AP+Ca+T1.0+T1.1": [
        v[:, keep], ca[:, keep], ta[:, idx[1.0], keep], ta[:, idx[1.1], keep]
    ],
}
folds = KFold(n_splits=min(5, len(theta)), shuffle=True, random_state=20260928)
model_results = {}
frozen_results = {}
frozen_cell_errors = {}
for label, parts in modules.items():
    x = np.concatenate(parts, axis=1)
    pred = np.full_like(params01, np.nan)
    for tr, te in folds.split(x):
        fit = make_pipeline(
            StandardScaler(),
            RidgeCV(alphas=np.array([1.0, 10.0, 100.0, 1000.0]), cv=3),
        )
        fit.fit(x[tr], params01[tr])
        pred[te] = fit.predict(x[te])
    model_results[label] = {
        "mean_parameter_absolute_error_on_prior_scale": float(np.mean(np.abs(pred - params01))),
        "electrical_parameter_absolute_error": float(np.mean(np.abs(pred[:, :4] - params01[:, :4]))),
        "mechanical_parameter_absolute_error": float(np.mean(np.abs(pred[:, 4:] - params01[:, 4:]))),
        "parameter_r2": {
            name: float(r2_score(params01[:, i], pred[:, i]))
            for i, name in enumerate(names)
        },
    }
    if split is not None and np.all(np.bincount(split.astype(int), minlength=3) >= 2):
        train = np.flatnonzero(split == 0)
        valid = np.flatnonzero(split == 1)
        test = np.flatnonzero(split == 2)
        validation_losses = []
        for alpha in [1.0, 10.0, 100.0, 1000.0]:
            fit = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            fit.fit(x[train], params01[train])
            validation_losses.append(float(np.mean(np.abs(fit.predict(x[valid]) - params01[valid]))))
        best_alpha = [1.0, 10.0, 100.0, 1000.0][int(np.argmin(validation_losses))]
        fit = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha))
        fit.fit(x[np.r_[train,valid]], params01[np.r_[train,valid]])
        pred_test = fit.predict(x[test])
        frozen_results[label] = {
            "train_n": len(train), "validation_n": len(valid), "test_n": len(test),
            "alpha_selected_on_validation": best_alpha,
            "test_electrical_parameter_absolute_error": float(np.mean(np.abs(pred_test[:, :4] - params01[test, :4]))),
            "test_mechanical_parameter_absolute_error": float(np.mean(np.abs(pred_test[:, 4:] - params01[test, 4:]))),
        }
        frozen_cell_errors[label] = np.mean(np.abs(pred_test[:, 4:] - params01[test, 4:]), axis=1)

paired_improvements = {}
if "AP+Ca" in frozen_cell_errors:
    rng = np.random.default_rng(20260928)
    base = frozen_cell_errors["AP+Ca"]
    for key in ["AP+Ca+T1.0", "AP+Ca+T1.0+T1.1"]:
        delta = base - frozen_cell_errors[key]
        sample = rng.integers(0, len(delta), size=(5000, len(delta)))
        paired_improvements[key] = {
            "mean_prior_scaled_MAE_reduction": float(delta.mean()),
            "paired_bootstrap_95pct_interval": np.quantile(delta[sample].mean(axis=1), [.025, .975]).tolist(),
        }

coverage = {}
for value in [1.0, 1.1]:
    ours = np.max(ta[:, idx[value], :], axis=1)
    path = ROOT / f"Data/ventricular_cell_model/contraction_isometric_stretch{value:.1f}/outputs.csv"
    reference = col(path, "Tpeak,v")
    coverage[str(value)] = {
        "ours_n": len(ours),
        "ours_tpeak_q05_q50_q95_kPa": np.quantile(ours, [.05, .5, .95]).tolist(),
        "strocchi_n": len(reference),
        "strocchi_tpeak_q05_q50_q95_kPa": np.quantile(reference, [.05, .5, .95]).tolist(),
        "ours_fraction_inside_strocchi_q05_q95": float(np.mean(
            (ours >= np.quantile(reference, .05)) & (ours <= np.quantile(reference, .95))
        )),
    }

result = {
    "sampled_cells": int(len(ok)),
    "passed_cells": int(ok.sum()),
    "electrical_beats_used_q50_q95_max": [
        float(np.median(beats)), float(np.quantile(beats, .95)), float(np.max(beats))
    ],
    "max_passed_calcium_cycle_drift": float(np.max(cycle)),
    "max_passed_land_cycle_drift": float(np.max(land_cycle)),
    "strocchi_active_peak_sanity": coverage,
    "held_out_cell_ridge_pilot": model_results,
    "frozen_split_ridge_pilot": frozen_results,
    "paired_test_mechanical_improvements_vs_AP_plus_Ca": paired_improvements,
    "scope": "small-sample diagnostic, not posterior inference or counterfactual prediction",
}
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
labels = list(model_results)
mech = [model_results[k]["mechanical_parameter_absolute_error"] for k in labels]
elec = [model_results[k]["electrical_parameter_absolute_error"] for k in labels]
fig, ax = plt.subplots(figsize=(8.5, 4.5), constrained_layout=True)
x = np.arange(len(labels))
ax.plot(x, mech, "o-", label="mechanical parameters")
ax.plot(x, elec, "o-", label="electrical parameters")
ax.set_xticks(x, [s.replace("+", "+\n") for s in labels])
ax.set_ylabel("Held-out-cell MAE / parameter prior width")
ax.set_ylim(bottom=0)
ax.grid(axis="y", alpha=.25)
ax.legend()
fig.savefig(COHORT / "modality_observability_pilot.png", dpi=160)
plt.close(fig)
print(json.dumps(result, ensure_ascii=False, indent=2))
