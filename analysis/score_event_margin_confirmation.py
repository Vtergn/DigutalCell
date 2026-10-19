"""One-time target reveal for the frozen event-margin warning score."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

OUT = Path(__file__).resolve().parent / "human_bidirectional_output"
with h5py.File(OUT / "dynamic_c3_event_margin.mat") as f:
    truth = f["waveforms"][:].astype(float)
    truth_ok = f["ok"][:].astype(bool)
with h5py.File(OUT / "dynamic_event_margin_targets.mat") as f:
    pred = f["pred"][:].astype(float)
    ok = f["ok"][:].ravel().astype(bool)
    tab = f["tab"][:].T.astype(float)
    local = f["local_differences"][:].T.astype(float)
    local_fail = f["local_failure"][:].T.astype(bool)
    duration = f["pred_duration"][:].ravel().astype(float)
    cal_residual = f["calibration_residual"][:].ravel().astype(float)

assert truth.shape == (64, 4, 3, 1001)
assert pred.shape[0] == len(ok) == len(tab) == len(local)
assert local.shape[1] == 6
LO = np.array([.9, .85, .85, .85, .9, .8, .4, .75, 1.8,
               4.5, 1.8, -4.1, 100.])
HI = np.array([1.1, 1.15, 1.15, 1.15, 1.1, 1.2, 1.6, .86,
               2.2, 5.5, 2.8, -.9, 140.])

eligible = np.flatnonzero(truth_ok.all(axis=1))
row_by_cell = {int(c)-1: j for j, c in enumerate(tab[:, 0])}
assert set(row_by_cell) == set(np.flatnonzero(truth_ok[:, :3].all(axis=1)))
common = [c for c in eligible if ok[row_by_cell[c]]]

records = []
for cell in common:
    row = row_by_cell[cell]
    target = pred[row, 1]
    aligned = truth[cell, 0] + target - pred[row, 0]
    actual = truth[cell, 3]
    err = float(np.abs(aligned[0, 300:602]-actual[0, 300:602]).mean())
    local_score = (np.inf if local_fail[row].any() else
                   float(np.nanmax(local[row])))
    dominant = ("local_QC_failure" if local_fail[row].any() else
                ["IKr", "INaL", "g_ionic"][int(np.nanargmax(local[row])) // 2])
    estimate = tab[row, 1:]
    bounds = int(np.sum(np.isclose(estimate, LO, rtol=0, atol=1e-9) |
                        np.isclose(estimate, HI, rtol=0, atol=1e-9)))
    records.append({"cell": int(cell+1), "v_mae_mV": err,
                    "tail_gt5": int(err > 5),
                    "local_waveform_sensitivity_mV": local_score,
                    "local_failure": int(local_fail[row].any()),
                    "dominant_perturbation": dominant,
                    "bound_count": bounds,
                    "calibration_residual_mV": float(cal_residual[row]),
                    "pred_duration_above_minus50_ms": float(duration[row])})


def metrics(y, score):
    score = np.asarray(score, float)
    rank = score.copy()
    if np.isinf(rank).any():
        finite = rank[np.isfinite(rank)]
        rank[np.isinf(rank)] = (finite.max() + 1 if len(finite) else 1)
    k = max(1, int(np.ceil(.1*len(y))))
    top = np.argsort(-rank)[:k]
    return {"AUROC": float(roc_auc_score(y, rank)) if len(set(y)) == 2 else None,
            "average_precision": float(average_precision_score(y, rank)) if y.sum() else None,
            "top_decile_recall": float(y[top].sum()/y.sum()) if y.sum() else None,
            "spearman_rho_with_error": float(spearmanr(rank, errors).statistic),
            "n_local_QC_failure": int(np.isinf(score).sum())}


y = np.array([r["tail_gt5"] for r in records])
errors = np.array([r["v_mae_mV"] for r in records])
scores = {
    "local_waveform_sensitivity": np.array([
        r["local_waveform_sensitivity_mV"] for r in records]),
    "bound_count": np.array([r["bound_count"] for r in records]),
    "calibration_residual": np.array([
        r["calibration_residual_mV"] for r in records]),
    "predicted_duration": np.array([
        r["pred_duration_above_minus50_ms"] for r in records]),
}
result = {"seed": 20261021, "n_source": 64,
          "n_truth_complete": len(eligible), "n_native_success": int(ok.sum()),
          "n_common": len(common), "n_tail_gt5": int(y.sum()),
          "n_truth_valid_native_failed": int(sum(
              not ok[row_by_cell[c]] for c in eligible)),
          "n_truth_failed_native_valid": int(sum(
              ok[row_by_cell[c]] for c in range(64) if c not in eligible)),
          "v_mae_mean": float(errors.mean()) if len(errors) else None,
          "v_mae_median": float(np.median(errors)) if len(errors) else None,
          "v_mae_cvar90": float(np.sort(errors)[-max(1, int(np.ceil(.1*len(errors)))):].mean())
          if len(errors) else None,
          "scores": {name: metrics(y, value) for name, value in scores.items()},
          "per_cell": records}

# Cell bootstrap quantifies how little a handful of severe cases can support.
rng = np.random.default_rng(20261022)
for name, value in scores.items():
    rank = value.copy()
    if np.isinf(rank).any():
        finite = rank[np.isfinite(rank)]
        rank[np.isinf(rank)] = finite.max()+1 if len(finite) else 1
    auc_samples = []
    ap_samples = []
    for _ in range(3000):
        take = rng.integers(len(y), size=len(y))
        if len(set(y[take])) < 2:
            continue
        auc_samples.append(roc_auc_score(y[take], rank[take]))
        ap_samples.append(average_precision_score(y[take], rank[take]))
    result["scores"][name]["bootstrap_AUROC_ci95"] = (
        np.quantile(auc_samples, [.025, .975]).tolist() if auc_samples else None)
    result["scores"][name]["bootstrap_AP_ci95"] = (
        np.quantile(ap_samples, [.025, .975]).tolist() if ap_samples else None)
    result["scores"][name]["bootstrap_valid_resamples"] = len(auc_samples)

with (OUT / "dynamic_event_margin_confirmation_cells.csv").open(
        "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
(OUT / "dynamic_event_margin_confirmation_scores.json").write_text(
    json.dumps(result, indent=2), encoding="utf-8")

fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
for name, label, ax in [
    ("local_waveform_sensitivity", "Local waveform sensitivity (mV)", axes[0]),
    ("bound_count", "Inferred parameters at prior bound", axes[1]),
]:
    x = scores[name].copy()
    if np.isinf(x).any():
        finite = x[np.isfinite(x)]
        x[np.isinf(x)] = finite.max()+1 if len(finite) else 1
    ax.scatter(x[y == 0], errors[y == 0], s=25, alpha=.7, label="≤5 mV")
    ax.scatter(x[y == 1], errors[y == 1], s=42, marker="^", label=">5 mV")
    ax.axhline(5, color="gray", linestyle="--", linewidth=.8)
    ax.set_xlabel(label)
    ax.set_ylabel("Held-out long-hold V MAE (mV)")
    ax.grid(alpha=.2)
axes[0].legend(frameon=False)
fig.savefig(OUT / "dynamic_event_margin_confirmation.png", dpi=180)
print(json.dumps({k: v for k, v in result.items() if k != "per_cell"}, indent=2))
