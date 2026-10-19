"""Sealed-target scores after inverse theta, native re-pacing, and SAC gating."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

OUT = Path(__file__).resolve().parent / "human_bidirectional_output"
with h5py.File(OUT / "dynamic_discrepancy_cohort.mat") as f:
    truth = f["waveforms"][:].astype(float)
    original_valid = f["ok"][:].astype(bool).all(axis=1)
with h5py.File(OUT / "dynamic_inverse_calibration.mat") as f:
    cal = f["cal"][:].astype(float)
    calibration_ok = f["ok"][:].ravel().astype(bool)
    tab = f["tab"][:].astype(float)
    mapping = f["map"][:].ravel().astype(int)-1
with h5py.File(OUT / "dynamic_inverse_targets.mat") as f:
    prior = f["prior_pred"][:].astype(float)
    gated = f["gate_pred"][:].astype(float)
    prior_ok = f["prior_ok"][:].ravel().astype(bool)
    gate_ok = f["gate_ok"][:].ravel().astype(bool)

if tab.shape[0] == 16:
    tab = tab.T
assert tab.shape[1] == 16 and len(tab) == len(mapping) == len(gated)
assert prior.shape[1:] == (3, 3, 1001)
assert gated.shape[1:] == (3, 3, 1001)

windows = [[(300, 602)], [(300, 512)], [(150, 362), (600, 822)]]
records = []
for observation_kind in [0, 1]:
    for form in [0, 1, 2]:
        rows = np.flatnonzero((tab[:, 2] == observation_kind) &
                              (tab[:, 1] == form+1))
        for target in range(3):
            mask = np.zeros(1001, bool)
            for a, b in windows[target]:
                mask[a:b] = True
            for method in ["native_physics", "native_physics_gate"]:
                if method == "native_physics":
                    good = calibration_ok[mapping[rows]] & prior_ok[mapping[rows]]
                    prediction = prior[mapping[rows[good]], target]
                else:
                    good = calibration_ok[mapping[rows]] & gate_ok[rows]
                    prediction = gated[rows[good], target]
                current = rows[good]
                cell = tab[current, 0].astype(int)-1
                assert np.all(original_valid[cell])
                actual = truth[cell, form, target+3]
                aligned = truth[cell, form, 0] + prediction - cal[mapping[current], 0]
                for alignment, estimate in [("raw", prediction),
                                            ("baseline_aligned", aligned)]:
                    e = np.abs(estimate[:, :, mask] - actual[:, :, mask])
                    records.append({"observation_kind": observation_kind,
                                    "truth_form": form, "target": target,
                                    "method": method, "alignment": alignment,
                                    "n_valid": len(current),
                                    "cell_ids_zero_based": cell.tolist(),
                                    "mae_V_Ca_T": e.mean(axis=(0, 2)).tolist()
                                    if len(e) else [None]*3,
                                    "cellwise_mae": e.mean(axis=2).tolist()
                                    if len(e) else []})

report = {"development_model_selection": "five-fold cell CV on theta only",
          "test_parameter_truth_used_for_fitting": False,
          "target_truth_used_for_fitting": False,
          "n_original_valid_cells": int(original_valid.sum()),
          "n_unique_inverse_states": len(cal),
          "n_repace_success": int(calibration_ok.sum()),
          "n_prior_target_success": int(prior_ok.sum()),
          "n_gate_target_success": int(gate_ok.sum()),
          "results": records}
(OUT / "dynamic_inverse_target_scores.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8")
print("REPACE", report["n_repace_success"], "/", len(cal))
for r in records:
    if r["alignment"] == "baseline_aligned":
        print(r["observation_kind"], r["truth_form"], r["target"],
              r["method"], r["n_valid"], np.round(r["mae_V_Ca_T"], 4))
