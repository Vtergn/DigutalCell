"""Select bounded inverse and one C3 using development cells only."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

OUT = Path(__file__).resolve().parent / "human_bidirectional_output"
with h5py.File(OUT / "dynamic_c3_development_targets.mat") as f:
    pred = f["pred"][:].astype(float)
    ok = f["ok"][:].ravel().astype(bool)
    tab = f["tab"][:].T.astype(float)
with h5py.File(OUT / "dynamic_discrepancy_development.mat") as f:
    truth = f["waveforms"][:].astype(float)
    truth_ok = f["ok"][:].astype(bool)
with h5py.File(OUT / "dynamic_c3_development.mat") as f:
    c3_ok = f["ok"][:].astype(bool)

assert pred.shape[0] == len(tab) == len(ok)
development_ids = np.flatnonzero(truth_ok[:, 1])
assert len(development_ids) == 47


def metrics(values):
    e = np.asarray(values, float)
    n = len(e)
    if not n:
        return None
    k = int(np.ceil(.1*n))
    return {"n": n, "mean": float(e.mean()), "median": float(np.median(e)),
            "p90": float(np.quantile(e, .9)),
            "cvar90": float(np.sort(e)[-k:].mean()),
            "n_above_5_mV": int((e > 5).sum())}


records = []
for method in [0, 1]:
    for arm in [0, 4, 5]:
        rows = np.flatnonzero((tab[:, 1] == arm) & (tab[:, 2] == method))
        assert len(rows) == 47
        cells = tab[rows, 0].astype(int)-1
        assert set(cells) == set(development_ids)
        keep = ok[rows]
        good_rows, good_cells = rows[keep], cells[keep]
        actual = truth[good_cells, 1, 3]
        baseline = truth[good_cells, 1, 0]
        target = pred[good_rows, 1]
        aligned = baseline + target-pred[good_rows, 0]
        e = np.abs(aligned[:, 0, 300:602]-actual[:, 0, 300:602]).mean(axis=1)
        # Raw (unaligned) second-event characteristics are interpretable.
        event = []
        for i, cell in enumerate(good_cells):
            av, pv = actual[i, 0], target[i, 0]
            def read(v):
                crossing = np.flatnonzero((v[300:601] < -50) &
                                           (v[301:602] >= -50))
                return ((int(crossing[0]+300) if len(crossing) else None),
                        int(np.sum(v[300:602] > -50)))
            ta, da = read(av)
            tp, dp = read(pv)
            event.append({"cell": int(cell), "truth_t": ta, "pred_t": tp,
                          "truth_duration": da, "pred_duration": dp})
        records.append({"method": method, "arm": arm,
                        "n_truth": len(cells), "n_native_valid": int(keep.sum()),
                        "cell_ids": good_cells.tolist(),
                        "cellwise_mae_V": e.tolist(),
                        "summary": metrics(e), "events": event})

baseline = [r for r in records if r["arm"] == 0]
chosen_method = min(baseline,
                    key=lambda r: (r["n_truth"]-r["n_native_valid"],
                                   r["summary"]["cvar90"]))["method"]
base_record = next(r for r in records if r["arm"] == 0 and
                   r["method"] == chosen_method)
eligible = [r for r in records if r["method"] == chosen_method and
            r["arm"] in [4, 5] and
            r["n_native_valid"] >= base_record["n_native_valid"] and
            c3_ok[development_ids, r["arm"]-1].all()]
selected_arm = (min(eligible,
                    key=lambda r: (r["summary"]["cvar90"],
                                   r["summary"]["p90"],
                                   r["summary"]["mean"], r["arm"]))["arm"]
                if eligible else None)
result = {"development_only": True, "n_cells": len(development_ids),
          "chosen_method": chosen_method, "selected_arm": selected_arm,
          "protocol": ({4: [600, 1.10, 20], 5: [600, 1.10, 35]}
                       .get(selected_arm)),
          "records": records}
(OUT / "dynamic_c3_development_scores.json").write_text(
    json.dumps(result, indent=2), encoding="utf-8")
print("chosen boundary method", chosen_method, "selected C3", selected_arm)
for r in records:
    print(r["method"], r["arm"], r["n_native_valid"], r["summary"])
