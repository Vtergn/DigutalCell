"""Audit whether a small observational analogue posterior supports adaptive probes.

This is a feasibility diagnostic, not a digital-twin posterior or an adaptive
policy evaluation. Future responses enter scores only, never the feature map.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
OUT = HERE / "human_bidirectional_output"
SCALE = np.array([20., .1, 2.])[None, :, None]
WINDOW = slice(300, 602)


def features(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """w: cells x condition x modality x time; condition 0/1/2 = B/C1/C2."""
    b = (w[:, 0, :, ::20] / SCALE).reshape(len(w), -1)
    parts = [b]
    for cond, onset in ((1, 150), (2, 600)):
        response = (w[:, cond, :, onset:onset + 201:10]
                    - w[:, 0, :, onset:onset + 201:10]) / SCALE
        parts.append(response.reshape(len(w), -1))
    return b, np.concatenate(parts, axis=1)


def weighted_quantile(x: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(x)
    cdf = np.cumsum(weights[order])
    return float(x[order[np.searchsorted(cdf, q, side="left")]])


def distance_matrices(dev: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = StandardScaler().fit(dev)
    a, b = scale.transform(dev), scale.transform(test)
    # PCA removes redundant sampled time points without seeing future targets.
    pca = PCA(n_components=8, random_state=0).fit(a)
    a, b = pca.transform(a), pca.transform(b)
    # Each PC is standardized to prevent its eigenvalue from simply counting
    # correlated waveform samples as independent measurements.
    sd = np.maximum(np.std(a, axis=0), 1e-8)
    a, b = a / sd, b / sd
    dd = np.linalg.norm(a[:, None] - a[None], axis=2)
    td = np.linalg.norm(b[:, None] - a[None], axis=2)
    return dd, td


def analogue_weights(dist: np.ndarray, target_ess: float) -> np.ndarray:
    # Query-specific bandwidth fixed by the chosen effective particle count.
    # This is an empirical analogue distribution, not p(theta | data).
    lo, hi = 1e-6, 1e6
    for _ in range(70):
        h = (lo * hi) ** .5
        logits = -(dist / h) ** 2 / 2
        logits -= np.max(logits)
        w = np.exp(logits)
        w /= w.sum()
        ess = 1 / np.sum(w ** 2)
        if ess < target_ess:
            lo = h
        else:
            hi = h
    return w


def assess(dist: np.ndarray, truth: np.ndarray, bank: np.ndarray,
           ess: float, leave_one_out: bool = False) -> dict:
    scores = []
    event_true = np.max(np.abs(truth), axis=1) > 30
    for i, d in enumerate(dist):
        d = d.copy()
        if leave_one_out:
            d[i] = np.inf
        w = analogue_weights(d, ess)
        mean = w @ bank
        mae = float(np.mean(np.abs(mean - truth[i])))
        lo = np.array([weighted_quantile(bank[:, t], w, .05)
                       for t in range(bank.shape[1])])
        hi = np.array([weighted_quantile(bank[:, t], w, .95)
                       for t in range(bank.shape[1])])
        # Marginal pointwise coverage is weaker than whole-trajectory coverage.
        marginal = float(np.mean((truth[i] >= lo) & (truth[i] <= hi)))
        simultaneous = bool(np.all((truth[i] >= lo) & (truth[i] <= hi)))
        event_prob = float(w @ (np.max(np.abs(bank), axis=1) > 30))
        scores.append((mae, marginal, simultaneous, event_prob))
    a = np.asarray(scores)
    auc = (float(roc_auc_score(event_true, a[:, 3]))
           if np.any(event_true) and not np.all(event_true) else None)
    return {
        "n": len(truth), "ess": ess,
        "mean_waveform_mae_mV": float(a[:, 0].mean()),
        "median_waveform_mae_mV": float(np.median(a[:, 0])),
        "p90_waveform_mae_mV": float(np.quantile(a[:, 0], .9)),
        "pointwise_90pct_coverage": float(a[:, 1].mean()),
        "simultaneous_90pct_coverage": float(a[:, 2].mean()),
        "mean_predicted_large_response_probability": float(a[:, 3].mean()),
        "observed_large_response_rate": float(event_true.mean()),
        "large_response_brier": float(np.mean((a[:, 3] - event_true) ** 2)),
        "large_response_climatology_brier": float(np.mean((event_true.mean() - event_true) ** 2)),
        "large_response_auc": auc,
        "large_response_probability_max": float(a[:, 3].max()),
    }


with h5py.File(OUT / "dynamic_discrepancy_development.mat") as f:
    ok = f["ok"][:, 1].astype(bool)
    raw = f["waveforms"][:][ok, 1]
dev_b, dev_c = features(raw[:, :3])
dev_future = raw[:, 3, 0, WINDOW] - raw[:, 0, 0, WINDOW]

with h5py.File(OUT / "dynamic_c3_fresh.mat") as f:
    ok = f["ok"][:, [0, 1, 2, 4]].astype(bool).all(axis=1)
    test_ids = np.flatnonzero(ok) + 1
    raw = f["waveforms"][:][ok]
test_b, test_c = features(raw[:, :3])
test_future = raw[:, 4, 0, WINDOW] - raw[:, 0, 0, WINDOW]

result = {
    "limitations": [
        "Empirical analogue distribution, not a mechanistic parameter posterior.",
        "Fresh-seed cohort was already analyzed in the prior C3 experiment; not a sealed adaptive confirmation.",
        "No adaptive probe is selected or tested here.",
    ],
    "definition": "future response = long-hold voltage minus baseline voltage, 300:602 ms",
    "large_response_threshold_mV": 30,
    "development": {}, "previously_seen_fresh_seed": {},
}
for label, a, b in (("baseline", dev_b, test_b), ("baseline_C1_C2", dev_c, test_c)):
    dd, td = distance_matrices(a, b)
    for ess in (5., 10., 20.):
        key = f"{label}_ESS{int(ess)}"
        result["development"][key] = assess(dd, dev_future, dev_future, ess, True)
        result["previously_seen_fresh_seed"][key] = assess(td, test_future, dev_future, ess)

# Explicitly separate large *true response* risk from native-twin failure risk.
_, target_dist = distance_matrices(dev_c, test_c)
future_variance = []
large_response_probability = []
for d in target_dist:
    w = analogue_weights(d, 10.)
    mu = w @ dev_future
    future_variance.append(float(w @ np.mean((dev_future - mu) ** 2, axis=1)))
    large_response_probability.append(float(w @ (np.max(np.abs(dev_future), axis=1) > 30)))
scores = json.loads((OUT / "dynamic_c3_fresh_scores.json").read_text(encoding="utf-8"))
id_to_index = {int(cell_id): j for j, cell_id in enumerate(test_ids)}
paired_idx = [id_to_index[int(cell_id)] for cell_id in scores["paired_cell_ids_1based"]]
native_error = np.asarray(scores["paired_errors"]["no_c3"])
assert len(paired_idx) == len(native_error)
variance = np.asarray(future_variance)[paired_idx]
response_risk = np.asarray(large_response_probability)[paired_idx]
result["native_twin_error_association"] = {
    "n_paired": len(native_error),
    "n_native_error_above_5_mV": int(np.sum(native_error > 5)),
    "spearman_future_variance_vs_native_error": float(spearmanr(variance, native_error).statistic),
    "spearman_large_response_probability_vs_native_error": float(spearmanr(response_risk, native_error).statistic),
    "top_two_native_error_ids_1based": [int(scores["paired_cell_ids_1based"][j])
                                         for j in np.argsort(native_error)[-2:][::-1]],
    "top_two_variance_rank_among_44": [int(np.where(np.argsort(-variance) == j)[0][0] + 1)
                                       for j in np.argsort(native_error)[-2:][::-1]],
    "warning": "Association is retrospective and the cohort was previously analyzed."
}

path = OUT / "adaptive_probe_feasibility.json"
path.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
