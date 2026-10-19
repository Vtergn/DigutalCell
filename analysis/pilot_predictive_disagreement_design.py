"""Exploratory leave-one-cell-out action-utility pilot on development cells."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform

OUT = Path(__file__).resolve().parent / "human_bidirectional_output"
screen = json.loads((OUT / "predictive_probe_library_screen.json").read_text())
assert screen["gate_pass"]
probe_ids = np.array(screen["selected_ids_1based"], dtype=int)-1
assert len(probe_ids) == 6

with h5py.File(OUT / "dynamic_discrepancy_development.mat") as f:
    wave_all = f["waveforms"][:].astype(float)
    valid = f["ok"][:].astype(bool)[:, 1]
with h5py.File(OUT / "predictive_probe_library.mat") as f:
    baseline_all = f["baseline"][:].astype(float)
    probe_all = f["waveforms"][:].astype(float)
    probe_ok = f["ok"][:].astype(bool)
with h5py.File(OUT / "feedback_13param_cohort.mat") as f:
    theta_all = f["theta"][:].T

ids = np.flatnonzero(valid & probe_ok.all(axis=1))
assert len(ids) == 47
wave = wave_all[ids, 1]
baseline = baseline_all[ids]
probes = probe_all[ids]
theta = theta_all[ids]
assert np.nanmax(np.abs(wave[:, 0]-baseline)) < 1e-6

LO = np.array([.9, .85, .85, .85, .9, .8, .4, .75, 1.8,
               4.5, 1.8, -4.1, 100.])
HI = np.array([1.1, 1.15, 1.15, 1.15, 1.1, 1.2, 1.6, .86,
               2.2, 5.5, 2.8, -.9, 140.])
theta01 = (theta-LO)/(HI-LO)


def rms_distance(x):
    """Pairwise root mean square distance between row vectors."""
    return squareform(pdist(x, metric="euclidean"))/np.sqrt(x.shape[1])


# Equal-weight baseline, C1, C2 observability arms.
base_scale = np.array([20.0, .1, 2.0])[None, :, None]
delta_scale = np.array([5.0, .02, 2.0])[None, :, None]
arms = [(baseline[:, :, ::25]/base_scale).reshape(len(ids), -1)]
for cond, onset in [(1, 150), (2, 600)]:
    delta = wave[:, cond, :, onset:onset+201:10] - \
        baseline[:, :, onset:onset+201:10]
    arms.append((delta/delta_scale).reshape(len(ids), -1))
cal_d2 = sum(rms_distance(a)**2 for a in arms)/3

probe_response = ((probes-baseline[:, None])[:, :, :, ::25] /
                  np.array([5.0, .02, 2.0])[None, None, :, None])
probe_response = probe_response.reshape(len(ids), 18, -1)
future = wave[:, 3, 0, 300:602]-baseline[:, 0, 300:602]
base_v = baseline[:, 0, 300:602]


def evaluate_probe(held, candidates, prior, probe_id):
    resp = probe_response[candidates, probe_id]
    obs = probe_response[held, probe_id]
    pair_distance = rms_distance(resp)
    upper = pair_distance[np.triu_indices(len(candidates), 1)]
    bandwidth = float(np.median(upper[upper > 1e-12])) if np.any(
        upper > 1e-12) else 1.0
    d_obs = np.sqrt(np.mean((resp-obs)**2, axis=1))
    log_weight = np.log(np.maximum(prior, 1e-100)) - .5*(d_obs/bandwidth)**2
    post = np.exp(log_weight-log_weight.max())
    post /= post.sum()
    prediction = base_v[held] + post @ future[candidates]
    truth = base_v[held] + future[held]
    return float(np.mean(np.abs(prediction-truth)))


arms_names = ["no_probe", "fixed_prior_C3", "fixed_high_energy",
              "random_safe_expected", "parameter_information",
              "goal_information", "risk_targeted"]
records = []
random_probe_errors = []
selection = {name: [] for name in arms_names[4:]}
fallback_count = 0
for held in range(len(ids)):
    order = np.argsort(cal_d2[held], kind="stable")
    candidates = order[order != held][:12]
    assert len(candidates) == 12 and held not in candidates
    bandwidth = max(float(np.sqrt(cal_d2[held, candidates[-1]])), 1e-8)
    q = np.exp(-cal_d2[held, candidates]/(2*bandwidth**2))
    q /= q.sum()
    current = base_v[held] + q @ future[candidates]
    target = base_v[held] + future[held]
    errors = {"no_probe": float(np.mean(np.abs(current-target)))}

    pair = np.triu_indices(len(candidates), 1)
    qpair = (q[:, None]*q[None, :])[pair]
    d_future = np.mean(np.abs(
        future[candidates, None]-future[None, candidates]), axis=2)[pair]
    d_parameter = np.mean((theta01[candidates, None]-
                           theta01[None, candidates])**2, axis=2)[pair]
    dangerous = np.maximum(d_future-5.0, 0)**2
    if dangerous.sum() == 0:
        dangerous = d_future**2
        fallback_count += 1

    utility = {"parameter_information": [], "goal_information": [],
               "risk_targeted": []}
    for probe_id in probe_ids:
        d_probe = rms_distance(probe_response[candidates, probe_id])[pair]
        distinguishability = 1-np.exp(-.5*d_probe**2)
        utility["parameter_information"].append(float(
            np.sum(qpair*d_parameter*distinguishability)))
        utility["goal_information"].append(float(
            np.sum(qpair*d_future**2*distinguishability)))
        utility["risk_targeted"].append(float(
            np.sum(qpair*dangerous*distinguishability)))

    for name in utility:
        chosen = int(probe_ids[np.argmax(utility[name])])
        selection[name].append(chosen+1)
        errors[name] = evaluate_probe(held, candidates, q, chosen)
    errors["fixed_prior_C3"] = evaluate_probe(held, candidates, q, 3)
    errors["fixed_high_energy"] = evaluate_probe(held, candidates, q, 11)
    all_probe_errors = [evaluate_probe(held, candidates, q, int(j))
                        for j in probe_ids]
    random_probe_errors.extend(all_probe_errors)
    errors["random_safe_expected"] = float(np.mean(all_probe_errors))
    errors["oracle_best_of_six"] = float(np.min(all_probe_errors))
    records.append({"cell": int(ids[held]+1), **errors,
                    **{f"selected_{name}": selection[name][-1]
                       for name in utility}})


def summarize(values):
    a = np.asarray(values, float)
    k = int(np.ceil(.1*len(a)))
    return {"mean_mV": float(a.mean()), "median_mV": float(np.median(a)),
            "p90_mV": float(np.quantile(a, .9)),
            "cvar90_mV": float(np.sort(a)[-k:].mean()),
            "n_gt5_mV": int((a > 5).sum())}


result = {"n_cells": len(records), "candidate_count": 12,
          "selected_probe_ids": (probe_ids+1).tolist(),
          "risk_fallback_to_goal_count": fallback_count,
          "arms": {name: summarize([r[name] for r in records])
                   for name in arms_names},
          "selected_counts": {name: dict(sorted(Counter(picks).items()))
                              for name, picks in selection.items()},
          "same_development_population_used_to_screen_library": True,
          "withheld_cell_future_used_to_choose_probe": False,
          "is_calibrated_posterior": False}
result["posthoc_oracle_diagnostic"] = summarize(
    [r["oracle_best_of_six"] for r in records])
result["random_safe_mixture"] = summarize(random_probe_errors)
result["random_safe_mixture"]["expected_n_gt5_of_47"] = float(
    np.sum(np.asarray(random_probe_errors) > 5)/len(probe_ids))
result["risk_equals_goal_on_all_cells"] = all(
    r["selected_risk_targeted"] == r["selected_goal_information"]
    for r in records)

with (OUT / "predictive_disagreement_pilot_cells.csv").open(
        "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
(OUT / "predictive_disagreement_pilot_scores.json").write_text(
    json.dumps(result, indent=2), encoding="utf-8")

fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
labels = ["No probe", "Fixed old", "Fixed energy", "Random mean",
          "Parameter", "Goal", "Risk pairs"]
means = [result["arms"][name]["mean_mV"] for name in arms_names]
cvars = [result["arms"][name]["cvar90_mV"] for name in arms_names]
axes[0].bar(labels, means, color="#377da8")
axes[1].bar(labels, cvars, color="#b26748")
axes[0].set_ylabel("Held-out future V MAE, mean (mV)")
axes[1].set_ylabel("Held-out future V MAE, CVaR90 (mV)")
for ax in axes:
    ax.tick_params(axis="x", labelrotation=55)
    ax.grid(axis="y", alpha=.2)
fig.savefig(OUT / "predictive_disagreement_pilot.png", dpi=180)
print(json.dumps(result, indent=2))
