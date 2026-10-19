"""One-time new-seed C3 confirmation; paired cellwise long-hold risk."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

OUT = Path(__file__).resolve().parent / "human_bidirectional_output"
decision = json.loads((OUT / "dynamic_c3_development_scores.json").read_text())
selected = decision["selected_arm"]
assert selected in [4, 5]
with h5py.File(OUT / "dynamic_c3_fresh.mat") as f:
    truth = f["waveforms"][:].astype(float)
    truth_ok = f["ok"][:].astype(bool)
with h5py.File(OUT / "dynamic_c3_fresh_targets.mat") as f:
    pred = f["pred"][:].astype(float)
    ok = f["ok"][:].ravel().astype(bool)
    tab = f["tab"][:].T.astype(float)
assert pred.shape[0] == len(tab) == len(ok)


def risk(e):
    e = np.asarray(e)
    n = len(e)
    k = int(np.ceil(.1*n))
    return {"n": n, "mean": float(e.mean()),
            "median": float(np.median(e)), "p90": float(np.quantile(e, .9)),
            "cvar90": float(np.sort(e)[-k:].mean()),
            "n_above_5_mV": int((e > 5).sum())}


eligible = np.flatnonzero(truth_ok.all(axis=1))
data = {}
for arm in [0, selected]:
    rows = np.flatnonzero(tab[:, 1] == arm)
    row_by_cell = dict(zip(tab[rows, 0].astype(int)-1, rows))
    assert set(row_by_cell) == set(eligible)
    data[arm] = {c: row_by_cell[c] for c in eligible if ok[row_by_cell[c]]}
common = sorted(set(data[0]) & set(data[selected]))
assert common

errors = {}
events = {}
for arm in [0, selected]:
    e = []
    details = []
    for cell in common:
        row = data[arm][cell]
        target = pred[row, 1]
        aligned = truth[cell, 0] + target - pred[row, 0]
        actual = truth[cell, 4]
        e.append(np.abs(aligned[0, 300:602]-actual[0, 300:602]).mean())
        def read(v):
            crossing = np.flatnonzero((v[300:601] < -50) &
                                       (v[301:602] >= -50))
            return (int(crossing[0]+300) if len(crossing) else None,
                    int(np.sum(v[300:602] > -50)))
        ta, da = read(actual[0])
        tp, dp = read(target[0])
        details.append({"cell": int(cell+1), "truth_crossing": ta,
                        "pred_crossing": tp, "truth_duration": da,
                        "pred_duration": dp})
    errors[arm] = np.asarray(e)
    events[arm] = details

rng = np.random.default_rng(20261010)
n = len(common)
boot_idx = rng.integers(n, size=(10000, n))
cvar_diff = []
for idx in boot_idx:
    cvar_diff.append(risk(errors[selected][idx])["cvar90"] -
                     risk(errors[0][idx])["cvar90"])
result = {
    "new_seed": 20261009, "selected_arm": selected,
    "n_truth_complete": len(eligible),
    "native_success_no_c3": len(data[0]),
    "native_success_with_c3": len(data[selected]),
    "n_paired": n, "paired_cell_ids_1based": [int(c+1) for c in common],
    "no_c3": risk(errors[0]), "with_c3": risk(errors[selected]),
    "paired_cvar90_difference": float(risk(errors[selected])["cvar90"]-
                                       risk(errors[0])["cvar90"]),
    "paired_cvar90_difference_ci95": np.quantile(cvar_diff, [.025, .975]).tolist(),
    "paired_errors": {"no_c3": errors[0].tolist(),
                      "with_c3": errors[selected].tolist()},
    "events": {"no_c3": events[0], "with_c3": events[selected]},
}
for arm, key in [(0, "no_c3"), (selected, "with_c3")]:
    rows = events[arm]
    result[key]["event_presence_mismatch"] = sum(
        (r["truth_crossing"] is None) != (r["pred_crossing"] is None)
        for r in rows)
    result[key]["duration_abs_error_ms"] = float(np.mean([
        abs(r["pred_duration"]-r["truth_duration"]) for r in rows]))
    finite = [abs(r["pred_crossing"]-r["truth_crossing"]) for r in rows
              if r["truth_crossing"] is not None and r["pred_crossing"] is not None]
    result[key]["conditional_crossing_abs_error_ms"] = (
        float(np.mean(finite)) if finite else None)
    result[key]["crossing_abs_error_ms_including_joint_absence"] = float(np.mean([
        0 if r["truth_crossing"] is None and r["pred_crossing"] is None
        else abs(r["pred_crossing"]-r["truth_crossing"])
        for r in rows if (r["truth_crossing"] is None) ==
        (r["pred_crossing"] is None)]))

(OUT / "dynamic_c3_fresh_scores.json").write_text(
    json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps({k: result[k] for k in ["n_truth_complete", "n_paired",
                                      "no_c3", "with_c3",
                                      "paired_cvar90_difference",
                                      "paired_cvar90_difference_ci95"]}, indent=2))
