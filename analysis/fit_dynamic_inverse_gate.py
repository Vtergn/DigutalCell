"""Fit SAC presence/strength using only observed C1/C2 and inverse ODE calibrations."""
from pathlib import Path
import json

import h5py
import numpy as np
from sklearn.metrics import roc_auc_score

OUT = Path(__file__).resolve().parent / "human_bidirectional_output"
with h5py.File(OUT / "dynamic_inverse_calibration.mat") as f:
    cal = f["cal"][:].astype(float)  # unique inverse, five traces, channel, time
    ok = f["ok"][:].ravel().astype(bool)
    tab = f["tab"][:].astype(float)
    mapping = f["map"][:].ravel().astype(int)-1
if tab.shape[0] == 16:
    tab = tab.T
with h5py.File(OUT / "dynamic_discrepancy_cohort.mat") as f:
    observed = f["waveforms"][:].astype(float)

assert cal.shape[1:] == (5, 3, 1001)
assert tab.shape[1] == 16
assert len(mapping) == len(tab)
rows = []
for row, (cell_id, form, obs_kind, *_) in enumerate(tab):
    cell_id, form, obs_kind = int(cell_id)-1, int(form)-1, int(obs_kind)
    u = mapping[row]
    if not ok[u]:
        rows.append([cell_id+1, form+1, obs_kind, u+1,
                     np.nan, np.nan, np.nan, 0])
        continue
    gains = []
    for j, onset in [(1, 150), (2, 600)]:
        window = slice(onset, onset+110)
        scale = np.array([5.0, .02])[None, :, None]
        # Model direct-SAC increment around the same inferred physiology.
        direct = ((cal[u, j, :2, window] - cal[u, j+2, :2, window]) /
                  scale[0]).ravel()
        # Compare response deltas so observed baseline is not confused with
        # the inferred physiology's baseline offset.
        observation = ((observed[cell_id, form, j, :2, window] -
                        observed[cell_id, form, 0, :2, window] -
                        (cal[u, j+2, :2, window] -
                         cal[u, 0, :2, window])) / scale[0]).ravel()
        denom = direct @ direct
        gains.append(float(direct @ observation / denom)
                     if denom > 1e-8 else np.nan)
    g1, g2 = gains
    gate = float(np.clip(g2, 0, 2)) if np.isfinite(g2) else 0.0
    rows.append([cell_id+1, form+1, obs_kind, u+1,
                 gate, g1, g2, 1])

arr = np.asarray(rows)
np.savetxt(OUT / "dynamic_inverse_gate.csv", arr,
           delimiter=",", fmt="%.12g")
summary = []
for obs_kind in [0, 1]:
    data = arr[(arr[:, 2] == obs_kind) & (arr[:, -1] == 1)]
    forms = data[:, 1].astype(int)
    result = {"observation_kind": obs_kind, "n": len(data),
              "median_gate_by_form": [float(np.median(data[forms == j, 4]))
                                       for j in [1, 2, 3]],
              "presence_AUC": float(roc_auc_score(forms != 3, data[:, 4]))}
    present = data[forms != 3]
    contrast = np.log(np.maximum(present[:, 5], .02) /
                      np.maximum(present[:, 6], .02))
    result["form_AUC"] = float(roc_auc_score(
        present[:, 1] == 2, contrast))
    summary.append(result)
(OUT / "dynamic_inverse_gate_score.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8")
for x in summary:
    print(x)
