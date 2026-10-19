"""Score the locked within-core dynamic model-discrepancy benchmark.

No target waveform is used for fitting or model selection. Source-family
short-pulse cohort only is used by the direct multioutput ridge baseline.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


OUT = Path(__file__).resolve().parent / "human_bidirectional_output"
with h5py.File(OUT / "dynamic_discrepancy_cohort.mat") as f:
    W = f["waveforms"][:]  # cell, form, protocol, channel, ms
    OK = f["ok"][:].astype(bool)  # cell, form
    LOAD = f["lambda_tracks"][:]  # protocol, ms
with h5py.File(OUT / "feedback_13param_cohort.mat") as f:
    SOURCE = f["waveforms"][:]  # cell, protocol, channel, ms
    ONSET = f["onsets"][:].ravel().astype(int)
    LAM = f["lambdas"][:].ravel()
    HOLD = f["holds"][:].ravel().astype(int)

assert W.shape[1:] == (3, 6, 3, 1001), W.shape
assert LOAD.shape == (6, 1001), LOAD.shape
assert SOURCE.shape == (64, 8, 3, 1001), SOURCE.shape
assert OK.shape == (W.shape[0], 3)
valid = np.flatnonzero(OK.all(axis=1))
assert len(valid) >= 10, "Too few cells for paired scoring"
W = W[valid].astype(float)


def source_length(j: int) -> np.ndarray:
    t = np.arange(1001)
    amp = LAM[j] - 1
    out = np.ones(1001)
    out[(t >= ONSET[j]) & (t < ONSET[j] + 1)] = 1
    out[(t >= ONSET[j] + 1) & (t < ONSET[j] + 1 + HOLD[j])] = 1 + amp
    out[(t >= ONSET[j] + 1 + HOLD[j]) & (t < ONSET[j] + 2 + HOLD[j])] = 1 + amp
    return out


# Explicit protocol list matches the locked generator. Each tuple is
# (onset, peak stretch, plateau duration), with 1-ms on/off ramps.
PULSES = {
    3: [(300, 1.10, 100)],
    4: [(300, 1.15, 10)],
    5: [(150, 1.08, 10), (600, 1.10, 20)],
}


def target_mask(j: int) -> np.ndarray:
    z = np.zeros(1001, bool)
    for onset, _, hold in PULSES[j]:
        z[onset:min(1001, onset + hold + 202)] = True
    return z


def transported_response(form: int, target: int) -> np.ndarray:
    """Causal, phase-shifted, amplitude-scaled linear superposition."""
    prediction = W[:, form, 0].copy()
    for onset, peak, _ in PULSES[target]:
        # Choose the closest available strain; ties favor C2.
        cal = 1 if abs(peak - 1.05) < abs(peak - 1.10) else 2
        source_onset = 150 if cal == 1 else 600
        source_amp = .05 if cal == 1 else .10
        shift = onset - source_onset
        delta = W[:, form, cal] - W[:, form, 0]
        shifted = np.zeros_like(delta)
        start = max(0, shift)
        stop = min(1001, 1001 + shift)
        shifted[:, :, start:stop] = delta[:, :, start-shift:stop-shift]
        prediction += shifted * ((peak - 1) / source_amp)
    return prediction


def compact_direct_features(cal_a: np.ndarray, cal_b: np.ndarray,
                            length: np.ndarray) -> np.ndarray:
    # All three observed full calibration waveforms, plus the complete
    # imposed target length trajectory; fixed 10-ms compression is for
    # computational conditioning, not target-specific feature selection.
    traces = np.concatenate([cal_a[:, :, ::10], cal_b[:, :, ::10]], axis=1)
    mech = length[:, ::10] - 1
    return np.concatenate([traces.reshape(len(traces), -1), mech], axis=1)


def fit_direct() -> tuple[np.ndarray, dict]:
    # C1 = source protocol 3 (early 1.05); C2 = protocol 1 (late 1.10).
    # Train labels use only 0,2,4; all are short-pulse source protocols.
    design = [0, 2, 4]
    ci = np.repeat(np.arange(64), len(design))
    pj = np.tile(design, 64)
    loads = np.stack([source_length(int(j)) for j in pj])
    x = compact_direct_features(SOURCE[ci, 3], SOURCE[ci, 1], loads)
    y = SOURCE[ci, pj].reshape(len(ci), -1)
    # Fixed model settings; no sealed-target-based selection. The response
    # is modeled in a 24D PCA output basis fitted only to source labels.
    xs = StandardScaler().fit(x)
    z = xs.transform(x)
    ys = StandardScaler().fit(y)
    yy = ys.transform(y)
    pca = PCA(n_components=24, random_state=0).fit(yy)
    model = Ridge(alpha=10).fit(z, pca.transform(yy))
    out = np.empty((len(valid), 3, 3, 3, 1001))
    for k, j in enumerate([3, 4, 5]):
        lv = np.repeat(LOAD[j][None], len(valid), axis=0)
        xt = compact_direct_features(W[:, 0, 1], W[:, 0, 2], lv)
        # For each hidden form, replace only the calibration waveforms.
        for form in range(3):
            xt = compact_direct_features(W[:, form, 1], W[:, form, 2], lv)
            out[:, form, k] = ys.inverse_transform(
                pca.inverse_transform(model.predict(xs.transform(xt)))).reshape(
                    len(valid), 3, 1001)
    return out, {"train_cells": 64, "train_examples": len(ci),
                 "label_protocols": design, "pca_outputs": 24,
                 "ridge_alpha": 10}


direct, direct_info = fit_direct()
prior_cal = W[:, 0, 1:3] - W[:, 0, 0, None]
observed_cal = W[:, :, 1:3] - W[:, :, 0, None]
# Least-squares response gain, independently for V/Ca/T. Only observed C1/C2
# are used; the source-prior parameter vector is intentionally oracle matched.
numerator = (prior_cal[:, None] * observed_cal).sum(axis=(2, 4))
denominator = (prior_cal[:, None] ** 2).sum(axis=(2, 4))
gain = np.divide(numerator, denominator, out=np.zeros_like(numerator),
                 where=denominator > 1e-12)
gain = np.clip(gain, 0, 3)
calibrated_prior = (W[:, :, 0, None] + gain[:, :, None, :, None]
                    * (W[:, 0:1, 3:6] - W[:, 0:1, 0, None]))
pred = {"zero_response": np.repeat(W[:, :, 0, None], 3, axis=2),
        "linear_transport": np.stack(
            [[transported_response(form, j) for j in [3, 4, 5]]
             for form in range(3)], axis=1).transpose(2, 0, 1, 3, 4),
        "matched_parameter_wrong_physics": np.repeat(W[:, 0:1, 3:6], 3, axis=1),
        "calibrated_prior_gain": calibrated_prior,
        "direct_ridge_sequence": direct,
        "oracle_true_form": W[:, :, 3:6]}

truth = W[:, :, 3:6]
assert all(p.shape == truth.shape for p in pred.values())

scores = {}
for name, p in pred.items():
    per_protocol = []
    for k, j in enumerate([3, 4, 5]):
        e = np.abs(p[:, :, k] - truth[:, :, k])
        mask = target_mask(j)
        per_protocol.append({
            "target": ["T1_long", "T2_large", "T3_compound"][k],
            "target_window_mae": e[:, :, :, mask].mean(axis=(0, 3)).tolist(),
            "full_beat_mae": e.mean(axis=(0, 3)).tolist(),
            "cellwise_target_window_mae": e[:, :, :, mask].mean(axis=3).tolist(),
        })
    scores[name] = per_protocol

result = {
    "scope": "within-core cross-feedback-form, one beat; not independent-model validation",
    "forms": ["prior", "saturating_SAC", "no_direct_SAC"],
    "channels_units": ["V_mV", "Ca_uM", "T_kPa"],
    "n_generated": int(OK.shape[0]),
    "n_valid_all_forms": int(len(valid)),
    "valid_zero_based_cell_ids": valid.tolist(),
    "valid_each_form": OK.sum(axis=0).tolist(),
    "direct_model": direct_info,
    "scores": scores,
}
(OUT / "dynamic_discrepancy_scores.json").write_text(
    json.dumps(result, indent=2), encoding="utf-8")
for name, entries in scores.items():
    print(name)
    for z in entries:
        print(z["target"], np.round(z["target_window_mae"], 4))
print("VALID", len(valid), "/", len(OK))
