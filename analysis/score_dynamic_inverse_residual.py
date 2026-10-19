"""Paired test of the locked flux residual after theta inverse and re-pacing."""
import json
from pathlib import Path

import h5py
import numpy as np

OUT = Path(__file__).resolve().parent / "human_bidirectional_output"
with h5py.File(OUT / "dynamic_discrepancy_cohort.mat") as f:
    truth = f["waveforms"][:].astype(float)
with h5py.File(OUT / "dynamic_inverse_calibration.mat") as f:
    cal = f["cal"][:].astype(float)
    tab = f["tab"][:].astype(float)
    mapping = f["map"][:].ravel().astype(int)-1
with h5py.File(OUT / "dynamic_inverse_targets.mat") as f:
    gated = f["gate_pred"][:].astype(float)
    gate_ok = f["gate_ok"][:].ravel().astype(bool)
with h5py.File(OUT / "dynamic_inverse_residual_targets.mat") as f:
    residual = f["residual_pred"][:].astype(float)
    residual_ok = f["residual_ok"][:].ravel().astype(bool)
if tab.shape[0] == 16:
    tab = tab.T
assert gated.shape == residual.shape == (len(tab), 3, 3, 1001)
windows = [[(300, 602)], [(300, 512)], [(150, 362), (600, 822)]]
rng = np.random.default_rng(20261002)
results = []
for obs_kind in [0, 1]:
    for form in [0, 1, 2]:
        rows = np.flatnonzero((tab[:, 1] == form+1) &
                              (tab[:, 2] == obs_kind) & gate_ok & residual_ok)
        cell = tab[rows, 0].astype(int)-1
        for target, spans in enumerate(windows):
            mask = np.zeros(1001, bool)
            for a, b in spans:
                mask[a:b] = True
            reference = truth[cell, form, target+3][:, :, mask]
            base = truth[cell, form, 0] - cal[mapping[rows], 0]
            pg = (gated[rows, target] + base)[:, :, mask]
            pr = (residual[rows, target] + base)[:, :, mask]
            eg = np.abs(pg-reference).mean(axis=2)
            er = np.abs(pr-reference).mean(axis=2)
            difference = er-eg
            draw = rng.integers(0, len(rows), (10000, len(rows)))
            ci = np.quantile(difference[draw].mean(axis=1), [.025, .975], axis=0).T
            results.append({"observation_kind": obs_kind, "form": form,
                            "target": target, "n_paired": len(rows),
                            "gate_MAE_V_Ca_T": eg.mean(axis=0).tolist(),
                            "residual_MAE_V_Ca_T": er.mean(axis=0).tolist(),
                            "residual_minus_gate": difference.mean(axis=0).tolist(),
                            "difference_95pct_cell_bootstrap": ci.tolist()})

out = {"method": "same theta and state; only SAC flux slope differs",
       "results": results}
(OUT / "dynamic_inverse_residual_scores.json").write_text(
    json.dumps(out, indent=2), encoding="utf-8")
for r in results:
    print(r["observation_kind"], r["form"], r["target"],
          "n", r["n_paired"], "gate", np.round(r["gate_MAE_V_Ca_T"], 4),
          "residual", np.round(r["residual_MAE_V_Ca_T"], 4),
          "delta", np.round(r["residual_minus_gate"], 4))
