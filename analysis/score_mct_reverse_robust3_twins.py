"""Score three-parameter, off-grid native-ODE point twins against direct models."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from score_mct2024_reference import beats


ROOT = Path(__file__).resolve().parent / "mct_feedback_audit" / "reverse_robust3"
FIELDS = ("APD_above_minus67_mV_ms", "Ca_peak_uM")


def read_case(path: Path):
    with h5py.File(path) as f:
        return {k: f["a"][k][()].ravel() for k in
                ("t_ms", "Vm_mV", "Cai_uM", "t_mech_ms", "Lneg", "force", "gelForce", "pCaCoeff")}


def score(truth: np.ndarray, pred: np.ndarray) -> dict:
    e = abs(truth-pred)
    return {"APD_MAE_ms": float(e[:, 0].mean()), "Ca_peak_MAE_uM": float(e[:, 1].mean()),
            "APD_median_AE_ms": float(np.median(e[:, 0])),
            "Ca_peak_median_AE_uM": float(np.median(e[:, 1])),
            "APD_max_AE_ms": float(e[:, 0].max()),
            "Ca_peak_max_AE_uM": float(e[:, 1].max())}


def main() -> None:
    cohort = json.loads((ROOT / "cohort_score.json").read_text(encoding="utf-8"))
    direct = json.loads((ROOT / "direct_predictions.json").read_text(encoding="utf-8"))
    ids = direct["test_ids"]
    truth = np.array(direct["truth"])
    predictors = {k: np.array(v) for k, v in direct["predictions"].items()}
    for mode in ("clean_low", "clean_lowmid", "noisy_lowmid"):
        for rep in ((0,) if mode != "noisy_lowmid" else (0, 1, 2)):
            native = []
            for ident in ids:
                path = ROOT / "native_twins" / f"{ident}_{mode}_{rep}.mat"
                if not path.exists():
                    raise FileNotFoundError(path)
                b = beats(read_case(path), final_beat=40)[-1]
                native.append([b[k] for k in FIELDS])
            predictors[f"native_{mode}_rep{rep}"] = np.array(native)
    scores = {k: score(truth, v) for k, v in predictors.items()}
    paired = {}
    twin = predictors["native_clean_lowmid_rep0"]
    for comparator in ("native_clean_low_rep0", "lowmid_ridge_clean",
                       "lowmid_RBF_clean", "linear_extrap_clean"):
        e1, e0 = abs(truth-twin), abs(truth-predictors[comparator])
        paired[comparator] = {field: {"twin_better": int(np.sum(e1[:, j] < e0[:, j])),
                                       "twin_worse": int(np.sum(e1[:, j] > e0[:, j]))}
                              for j, field in enumerate(FIELDS)}
    for rep in range(3):
        twin = predictors[f"native_noisy_lowmid_rep{rep}"]
        for comparator in (f"lowmid_ridge_aug_noise{rep}",
                           f"lowmid_KRR_aug_noise{rep}", f"linear_extrap_noise{rep}"):
            e1, e0 = abs(truth-twin), abs(truth-predictors[comparator])
            paired[comparator] = {field: {"twin_better": int(np.sum(e1[:, j] < e0[:, j])),
                                           "twin_worse": int(np.sum(e1[:, j] > e0[:, j]))}
                                  for j, field in enumerate(FIELDS)}
    result = {
        "scope": "3-parameter off-grid, same native rabbit MCT2024 equations",
        "n_train_cells": 48, "n_test_cells": 16,
        "all_conditions_pass_convergence": cohort["all_conditions_pass_convergence"],
        "convergence_failures": cohort["convergence_failures"],
        "scores": scores, "paired_counts": paired,
        "truth": truth.tolist(),
        "predictions": {k: v.tolist() for k, v in predictors.items()},
        "interpretation_boundary": "Same-equation simulation, clean synthetic forward training, arbitrary noise stress, no model discrepancy or real-data validation.",
    }
    (ROOT / "twin_score.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"all_conditions_pass_convergence": result["all_conditions_pass_convergence"],
                      "clean_scores": {k: v for k, v in scores.items() if "clean" in k},
                      "noise_scores": {k: v for k, v in scores.items() if "noise" in k or "noisy" in k}}, indent=2))


if __name__ == "__main__":
    main()
