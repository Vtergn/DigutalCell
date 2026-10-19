"""Sealed-target waveform scores for the C1/C2 narrow flux pilot."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

OUT = Path(__file__).resolve().parent / "human_bidirectional_output"
with h5py.File(OUT / "dynamic_discrepancy_cohort.mat") as f:
    truth = f["waveforms"][:].astype(float)
    valid = f["ok"][:].astype(bool).all(axis=1)
with h5py.File(OUT / "dynamic_flux_pilot_predictions.mat") as f:
    pred = f["waveforms"][:].astype(float)
    ok = f["ok"][:].astype(bool)
    tab = f["tab"][:].astype(float)
with h5py.File(OUT / "dynamic_discrepancy_development.mat") as f:
    dev = f["waveforms"][:].astype(float)

if tab.shape[0] == 7:
    tab = tab.T

ids = np.flatnonzero(valid)
assert pred.shape == (len(ids)*3, 3, 3, 3, 1001), pred.shape
assert ok.shape == (len(ids)*3, 3)
assert tab.shape == (len(ids)*3, 7)

windows = [[(300, 602)], [(300, 512)], [(150, 362), (600, 822)]]
results = []
for form in range(3):
    rows = np.flatnonzero(tab[:, 1] == form+1)
    assert len(rows) == len(ids)
    assert np.array_equal(tab[rows, 0].astype(int)-1, ids)
    true = truth[ids, form, 3:6]
    true_baseline = truth[ids, form, 0]
    prototype_baseline = dev[tab[rows, 2].astype(int)-1, 0, 0]
    for target in range(3):
        mask = np.zeros(1001, bool)
        for a, b in windows[target]:
            mask[a:b] = True
        for method, name in enumerate(["personalized_physics_prototype",
                                       "prototype_plus_sac_gate",
                                       "prototype_gate_flux_residual"]):
            good = ok[rows, method]
            raw = pred[rows[good], method, target]
            aligned = true_baseline[good] + raw - prototype_baseline[good]
            for alignment, estimate in [("raw", raw),
                                        ("baseline_aligned", aligned)]:
                err = np.abs(estimate[:, :, mask] -
                             true[good, target][:, :, mask])
                results.append({"truth_form": form, "target": target,
                                "method": name, "alignment": alignment,
                                "n_valid": int(good.sum()),
                                "row_ids": rows[good].tolist(),
                                "window_mae_V_Ca_T": err.mean(axis=(0, 2)).tolist()
                                if len(err) else [None]*3,
                                "cellwise_mae": err.mean(axis=2).tolist()
                                if len(err) else []})

metadata = json.loads((OUT / "dynamic_flux_pilot_metadata.json").read_text())
out = {"metadata": metadata, "n_test_rows": len(tab),
       "n_simulation_valid_by_method": ok.sum(axis=0).tolist(),
       "gate_by_true_form": [{"form": i,
                              "median": float(np.median(tab[tab[:, 1] == i+1, 3])),
                              "fraction_below_0.1": float(np.mean(
                                  tab[tab[:, 1] == i+1, 3] < .1)),
                              "residual_active_fraction": float(np.mean(
                                  tab[tab[:, 1] == i+1, 4] > 0))}
                             for i in range(3)],
       "results": results}
(OUT / "dynamic_flux_pilot_scores.json").write_text(
    json.dumps(out, indent=2), encoding="utf-8")
print("VALID", ok.sum(axis=0), "of", len(tab))
print("GATES", out["gate_by_true_form"])
for x in results:
    print(x["truth_form"], x["target"], x["method"], x["alignment"],
          x["n_valid"],
          np.round(x["window_mae_V_Ca_T"], 4))
