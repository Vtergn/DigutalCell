"""Target-blind safety and response-diversity screen for the frozen probes."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

OUT = Path(__file__).resolve().parent / "human_bidirectional_output"
with h5py.File(OUT / "predictive_probe_library.mat") as f:
    baseline = f["baseline"][:].astype(float)
    baseline_ok = f["baseline_ok"][:].ravel().astype(bool)
    waveforms = f["waveforms"][:].astype(float)
    ok = f["ok"][:].astype(bool)

assert baseline.shape == (64, 3, 1001)
assert waveforms.shape == (64, 18, 3, 1001)
assert ok.shape == (64, 18)
times = np.arange(0, 1001, 25)
scale = np.array([5.0, .02, 2.0])[None, None, :, None]
response = ((waveforms-baseline[:, None]) / scale)[:, :, :, times]
responses = response.reshape(64, 18, -1)

energy = np.full(18, np.nan)
safe_count = np.sum(ok & baseline_ok[:, None], axis=0)
for j in range(18):
    valid = ok[:, j] & baseline_ok
    if valid.any():
        energy[j] = np.median(np.sqrt(np.mean(responses[valid, j]**2, axis=1)))

correlation = np.eye(18)
for a in range(18):
    for b in range(a+1, 18):
        common = ok[:, a] & ok[:, b] & baseline_ok
        if common.sum() < 3:
            rho = np.nan
        else:
            da = pdist(responses[common, a], metric="euclidean")
            db = pdist(responses[common, b], metric="euclidean")
            rho = float(spearmanr(da, db).statistic)
        correlation[a, b] = correlation[b, a] = rho

eligible = [j for j in range(18) if safe_count[j] >= 61 and energy[j] >= .1]
selected = []
if eligible:
    first = min(eligible, key=lambda j: (-energy[j], j))
    selected = [first]
    remaining = [j for j in eligible if j != first]
    while remaining and len(selected) < 12:
        worst = {j: max(correlation[j, k] if np.isfinite(correlation[j, k])
                        else 1 for k in selected) for j in remaining}
        next_id = min(remaining, key=lambda j: (worst[j], j))
        if worst[next_id] >= .95:
            break
        selected.append(next_id)
        remaining.remove(next_id)

rows = [{"probe_id": j+1, "safe_count": int(safe_count[j]),
         "response_energy": float(energy[j]),
         "eligible": int(j in eligible), "selected": int(j in selected)}
        for j in range(18)]
with (OUT / "predictive_probe_library_screen.csv").open(
        "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

result = {"n_development_cells": 64,
          "n_baseline_valid": int(baseline_ok.sum()),
          "safe_minimum": 61,
          "energy_minimum": .1,
          "redundancy_maximum_spearman": .95,
          "n_eligible": len(eligible),
          "eligible_ids_1based": [j+1 for j in eligible],
          "selected_ids_1based": [j+1 for j in selected],
          "gate_pass": len(selected) >= 5,
          "safe_count": safe_count.tolist(),
          "response_energy": energy.tolist(),
          "distance_correlation": correlation.tolist(),
          "future_target_used": False}
(OUT / "predictive_probe_library_screen.json").write_text(
    json.dumps(result, indent=2), encoding="utf-8")

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
axes[0].bar(np.arange(1, 19), safe_count, color="#3179a6")
axes[0].axhline(61, color="#a44", linestyle="--", linewidth=1)
axes[0].set(xlabel="Probe ID", ylabel="QC-passing cells / 64",
            title="Simulator QC")
axes[0].set_xticks(np.arange(1, 19))
im = axes[1].imshow(correlation, vmin=0, vmax=1, cmap="viridis")
axes[1].set(xlabel="Probe ID", ylabel="Probe ID",
            title="Cell-distance Spearman correlation")
axes[1].set_xticks(np.arange(18), np.arange(1, 19), fontsize=7)
axes[1].set_yticks(np.arange(18), np.arange(1, 19), fontsize=7)
fig.colorbar(im, ax=axes[1], fraction=.04)
fig.savefig(OUT / "predictive_probe_library_screen.png", dpi=180)
print(json.dumps({k: v for k, v in result.items() if k !=
                  "distance_correlation"}, indent=2))
