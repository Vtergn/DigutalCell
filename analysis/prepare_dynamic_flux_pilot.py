"""Prepare a strictly C1/C2-calibrated, narrow SAC-flux correction pilot.

This is a bounded parametric residual precursor, not a trained neural UDE.
The shared slope is learned from the separate development cohort's C1/C2
responses only. For a new cell, the baseline selects a physics prototype;
C1/C2 estimate a SAC presence/strength gate and a shrunk slope correction.
No target waveform enters inference or selection.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

OUT = Path(__file__).resolve().parent / "human_bidirectional_output"
with h5py.File(OUT / "dynamic_discrepancy_development.mat") as f:
    dev = f["waveforms"][:].astype(float)
    dvalid = f["ok"][:].astype(bool).all(axis=1)
with h5py.File(OUT / "dynamic_discrepancy_cohort.mat") as f:
    test = f["waveforms"][:].astype(float)
    tvalid = f["ok"][:].astype(bool).all(axis=1)

dev_ids = np.flatnonzero(dvalid)
test_ids = np.flatnonzero(tvalid)
dev, test = dev[dvalid], test[tvalid]

# Shared residual shape from development data only. Compare direct-SAC
# increment to its zero-SAC counterpart so TnC-driven effects remain.
def sac_coefficient(a: np.ndarray, b: np.ndarray, j: int) -> float:
    onset = 150 if j == 1 else 600
    window = slice(onset, min(1001, onset+110))
    # V and Ca are separately scaled to physical response magnitudes.
    scale = np.array([5.0, .02])[:, None]
    x = ((a[0, j, :2, window] - a[2, j, :2, window]) / scale).ravel()
    y = ((b[j, :2, window] - a[2, j, :2, window]) / scale).ravel()
    denom = float(x @ x)
    return float(x @ y / denom) if denom > 1e-6 else np.nan


betas = []
for i in range(len(dev)):
    g1 = sac_coefficient(dev[i], dev[i, 1], 1)
    g2 = sac_coefficient(dev[i], dev[i, 1], 2)
    if g1 > .1 and g2 > .1:
        betas.append(np.log(g1/g2)/.05)
shared_beta = float(np.clip(np.median(betas), -10, 15))

# Match *baseline only* for the individual physiological prototype; then
# use C1/C2 to estimate effective SAC presence and residual slope.
scale = np.sqrt(np.mean(np.var(dev[:, 0, 0, :, ::10], axis=0), axis=1))
scale = np.maximum(scale, [1.0, .005, .05])
source_base = (dev[:, 0, 0, :, ::10] / scale[None, :, None]).reshape(len(dev), -1)
rows = []
for truth_form in range(3):
    for i, cell_id in enumerate(test_ids):
        obs_base = (test[i, truth_form, 0, :, ::10] / scale[:, None]).ravel()
        nearest = int(np.argmin(((source_base - obs_base)**2).mean(axis=1)))
        proto = dev[nearest]
        g1 = sac_coefficient(proto, test[i, truth_form], 1)
        g2 = sac_coefficient(proto, test[i, truth_form], 2)
        # C2 estimates presence/strength. C1/C2 ratio decides whether the
        # shared discrepancy is active; only then is a small individual
        # slope correction allowed. The decision is fixed before targets.
        gate = float(np.clip(g2, 0, 2)) if np.isfinite(g2) else 0.0
        if gate < .05 or not np.isfinite(g1) or g1 <= 0:
            beta = 0.0
        else:
            individual_beta = float(np.clip(np.log(max(g1, .02)/max(g2, .02))/.05,
                                            -10, 15))
            beta = (.75*shared_beta + .25*individual_beta
                    if individual_beta > shared_beta/2 else 0.0)
        rows.append([int(cell_id)+1, truth_form+1, int(dev_ids[nearest])+1,
                     gate, beta, g1, g2])

table = np.asarray(rows)
np.savetxt(OUT / "dynamic_flux_pilot_input.csv", table, delimiter=",",
           header="test_cell_1based,truth_form_1based,prototype_cell_1based,sac_gate,beta,g1,g2",
           comments="", fmt="%.12g")
metadata = {"n_development_valid": int(len(dev)),
            "n_confirmation_valid": int(len(test)),
            "shared_beta": shared_beta,
            "n_beta_development_cells": len(betas),
            "beta_rule": "if individual C1/C2 slope > half shared: 75% shared + 25% individual; else 0",
            "gate_rule": "C2 direct-SAC coefficient clipped [0,2]",
            "prototype_rule": "nearest baseline AP/Ca/T at 10ms, channel-scaled",
            "target_observation_used": False,
            "scope": "bounded parametric SAC-flux pilot, not neural UDE"}
(OUT / "dynamic_flux_pilot_metadata.json").write_text(
    json.dumps(metadata, indent=2), encoding="utf-8")
print(metadata)
