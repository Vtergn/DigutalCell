"""Score the locked 2x2 decomposition and calibration-only SAC model choice."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.metrics import confusion_matrix

OUT = Path(__file__).resolve().parent / "human_bidirectional_output"
with h5py.File(OUT / "dynamic_discrepancy_cohort.mat") as f:
    truth = f["waveforms"][:].astype(float)
    truth_ok = f["ok"][:].astype(bool).all(axis=1)
with h5py.File(OUT / "dynamic_inverse_calibration.mat") as f:
    cal = f["cal"][:].astype(float)
    tab = f["tab"][:].astype(float)
    mapping = f["map"][:].ravel().astype(int)-1
with h5py.File(OUT / "dynamic_model_candidates.mat") as f:
    candidates_cal = f["candidate_cal"][:].astype(float)
    candidates = f["candidate_target"][:].astype(float)
    candidate_ok = f["candidate_ok"][:].astype(bool)
with h5py.File(OUT / "dynamic_inverse_targets.mat") as f:
    source_gate = f["gate_pred"][:].astype(float)
    source_gate_ok = f["gate_ok"][:].ravel().astype(bool)

if tab.shape[0] == 16:
    tab = tab.T
assert candidates.shape[1:] == (3, 3, 3, 1001)
assert candidates_cal.shape[1:] == (3, 2, 3, 1001)
assert candidate_ok.shape == (len(candidates), 3)
gate = np.loadtxt(OUT / "dynamic_inverse_gate.csv", delimiter=",")
shared = json.loads((OUT / "dynamic_flux_pilot_metadata.json").read_text())["shared_beta"]
assert len(tab) == len(gate) == len(mapping) == len(source_gate)

windows = [[(300, 602)], [(300, 512)], [(150, 362), (600, 822)]]
rng = np.random.default_rng(20261002)


def target_mask(target):
    mask = np.zeros(1001, bool)
    for a, b in windows[target]:
        mask[a:b] = True
    return mask


def cell_error(pred, actual, mask):
    return np.abs(pred[:, :, mask]-actual[:, :, mask]).mean(axis=2)


# Frozen rule from the earlier flux pilot: absent / saturating / original.
selected = np.zeros(len(tab), int)
selected[gate[:, 4] < .1] = 2
contrast = np.log(np.maximum(gate[:, 5], .02) /
                  np.maximum(gate[:, 6], .02))
selected[(gate[:, 4] >= .1) & (contrast > .05*shared/2)] = 1

# An explicitly diagnostic alternative: full calibration waveform distance,
# with fixed physiological channel scales and no target-dependent tuning.
distance_selected = np.zeros(len(tab), int)
scales = np.array([5.0, .02, .5])[None, :, None]
for row in range(len(tab)):
    cell, form = int(tab[row, 0])-1, int(tab[row, 1])-1
    u = mapping[row]
    values = []
    for model in range(3):
        err = []
        for j, onset in [(0, 150), (1, 600)]:
            window = slice(onset, onset+110)
            model_response = (candidates_cal[u, model, j, :, window] -
                              cal[u, 0, :, window])
            observed_response = (truth[cell, form, j+1, :, window] -
                                 truth[cell, form, 0, :, window])
            err.append(np.mean(((model_response-observed_response)/scales[0])**2))
        values.append(np.mean(err))
    distance_selected[row] = int(np.argmin(values))

records = []
for obs_kind in [0, 1]:
    for form in range(3):
        rows = np.flatnonzero((tab[:, 2] == obs_kind) &
                              (tab[:, 1] == form+1))
        cells = tab[rows, 0].astype(int)-1
        assert np.all(truth_ok[cells])
        for method, picks in [("fixed_gain_rule", selected),
                              ("full_calibration_distance", distance_selected)]:
            records.append({"observation_kind": obs_kind, "truth_form": form,
                            "method": method,
                            "confusion_counts": confusion_matrix(
                                np.full(len(rows), form), picks[rows],
                                labels=[0, 1, 2]).tolist(),
                            "correct_fraction": float(np.mean(picks[rows] == form))})
        for target in range(3):
            mask = target_mask(target)
            methods = {
                "correct_form_at_theta_hat": np.full(len(rows), form),
                "fixed_gain_rule": selected[rows],
                "full_calibration_distance": distance_selected[rows],
            }
            for name, picks in methods.items():
                keep = candidate_ok[mapping[rows], picks]
                current = rows[keep]
                cc = cells[keep]
                uu = mapping[current]
                pp = picks[keep]
                pred = candidates[uu, pp, target]
                # Test baseline is observable; subtract the inferred-model
                # unperturbed response to score the induced response.
                aligned = truth[cc, form, 0] + pred - cal[uu, 0]
                actual = truth[cc, form, target+3]
                error = cell_error(aligned, actual, mask)
                records.append({"observation_kind": obs_kind,
                                "truth_form": form, "target": target,
                                "method": name, "n": len(current),
                                "cell_ids": cc.tolist(),
                                "mae_V_Ca_T": error.mean(axis=0).tolist()
                                if len(error) else [None]*3,
                                "cellwise_mae": error.tolist()})

# 2x2 oracle decomposition uses *the same common cells* in each arm.
decomposition = []
for obs_kind in [0, 1]:
    rows = np.flatnonzero((tab[:, 2] == obs_kind) & (tab[:, 1] == 2))
    keep = candidate_ok[mapping[rows], 0] & candidate_ok[mapping[rows], 1]
    rows = rows[keep]
    cells = tab[rows, 0].astype(int)-1
    uu = mapping[rows]
    for target in range(3):
        mask = target_mask(target)
        actual = truth[cells, 1, target+3]
        baseline = truth[cells, 1, 0]
        arms = {
            "theta_hat_original_SAC": baseline + candidates[uu, 0, target] - cal[uu, 0],
            "theta_true_original_SAC": truth[cells, 0, target+3],
            "theta_hat_saturating_SAC": baseline + candidates[uu, 1, target] - cal[uu, 0],
            "theta_true_saturating_SAC": actual,
        }
        result = {"observation_kind": obs_kind, "target": target,
                  "n_common": len(cells), "cell_ids": cells.tolist(),
                  "arms": {}}
        for name, prediction in arms.items():
            e = cell_error(prediction, actual, mask)
            result["arms"][name] = {"mae_V_Ca_T": e.mean(axis=0).tolist(),
                                    "cellwise_mae": e.tolist()}
        decomposition.append(result)

result = {"scope": "same-core synthetic; θtrue arms are oracle diagnostics",
          "shared_beta": shared,
          "candidate_valid_by_form": candidate_ok.sum(axis=0).tolist(),
          "selection_records": records,
          "decomposition": decomposition}
(OUT / "dynamic_model_selection_scores.json").write_text(
    json.dumps(result, indent=2), encoding="utf-8")
for d in decomposition:
    print("2x2 obs", d["observation_kind"], "target", d["target"],
          "n", d["n_common"], {k: round(v["mae_V_Ca_T"][0], 4)
                                 for k, v in d["arms"].items()})
for r in records:
    if "target" in r:
        print("SELECT", r["observation_kind"], r["truth_form"],
              r["target"], r["method"], r["n"],
              np.round(r["mae_V_Ca_T"], 4))
