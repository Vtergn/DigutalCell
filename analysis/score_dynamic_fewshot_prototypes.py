"""C1/C2-only physiological prototype individualization and mechanism gate.

The development cohort is disjoint from the confirmation cohort. A prototype
is selected only from baseline, C1 and C2 waveforms. Its target waveform is
then the native ODE solution cached by the development generator. No target
value participates in selection, scaling or hyperparameter choice.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

OUT = Path(__file__).resolve().parent / "human_bidirectional_output"
with h5py.File(OUT / "dynamic_discrepancy_development.mat") as f:
    dev = f["waveforms"][:].astype(float)
    dev_ok = f["ok"][:].astype(bool).all(axis=1)
with h5py.File(OUT / "dynamic_discrepancy_cohort.mat") as f:
    test = f["waveforms"][:].astype(float)
    test_ok = f["ok"][:].astype(bool).all(axis=1)

dev = dev[dev_ok]
test = test[test_ok]
assert dev.shape[1:] == test.shape[1:] == (3, 6, 3, 1001)

# Locked, unit-balanced cell variability scale. Include baseline and both
# observed calibration interventions, downsampled to 10-ms spacing.
cell_sd = np.sqrt(np.mean(np.var(dev[:, 0, :3], axis=0), axis=(0, 2)))
cell_sd = np.maximum(cell_sd, [1.0, .005, .05])


def feature(a: np.ndarray) -> np.ndarray:
    return (a[:, :3, :, ::10] / cell_sd[None, None, :, None]).reshape(len(a), -1)


def choose(candidates: np.ndarray, observation: np.ndarray) -> np.ndarray:
    a = feature(candidates)
    b = feature(observation)
    d2 = ((b[:, None] - a[None]) ** 2).mean(axis=2)
    return d2.argmin(axis=1)


rows = []
for truth_form in range(3):
    # Physics-only: all candidate prototypes use the original SAC law.
    best_source = choose(dev[:, 0], test[:, truth_form])
    source_pred = dev[best_source, 0, 3:6]

    # Mechanism-gated upper bound: known equation candidates, but the form
    # and cell prototype are inferred from the two observed probes.
    cand = dev.transpose(1, 0, 2, 3, 4).reshape(-1, 6, 3, 1001)
    best_gated = choose(cand, test[:, truth_form])
    gated_pred = cand[best_gated, 3:6]
    chosen_form = best_gated // len(dev)
    truth = test[:, truth_form, 3:6]
    for target, windows in enumerate([[(300, 602)], [(300, 512)],
                                      [(150, 362), (600, 822)]]):
        mask = np.zeros(1001, bool)
        for start, stop in windows:
            mask[start:stop] = True
        for method, pred in [("physics_prototype", source_pred),
                             ("known_form_gate_prototype", gated_pred)]:
            error = np.abs(pred[:, target][:, :, mask] -
                           truth[:, target][:, :, mask])
            rows.append({"truth_form": truth_form, "target": target,
                         "method": method,
                         "mae_V_Ca_T": error.mean(axis=(0, 2)).tolist(),
                         "cellwise_mae": error.mean(axis=2).tolist()})
    print("truth", truth_form, "source IDs", best_source[:5],
          "gated form counts", np.bincount(chosen_form, minlength=3))

result = {"development_valid": int(len(dev)),
          "confirmation_valid": int(len(test)),
          "feature_scales": cell_sd.tolist(),
          "observations": "baseline, C1, C2; full multimodal waveforms at 10 ms",
          "model_scope": "prototype native ODE; gated version knows candidate equation families",
          "results": rows}
(OUT / "dynamic_fewshot_prototype_scores.json").write_text(
    json.dumps(result, indent=2), encoding="utf-8")
for z in rows:
    print(z["truth_form"], z["target"], z["method"],
          np.round(z["mae_V_Ca_T"], 4))
