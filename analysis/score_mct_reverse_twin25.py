"""Score native-ODE counterfactuals from target-blind MCT calibration."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from score_mct2024_reference import beats


ROOT = Path(__file__).resolve().parent / "mct_feedback_audit"
INPUT = ROOT / "reverse_grid25_score.json"
PRED = ROOT / "reverse_twin25"
FIELDS = ("APD_above_minus67_mV_ms", "Ca_peak_uM")


def read_native(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path) as f:
        return {k: f["a"][k][()].ravel() for k in
                ("t_ms", "Vm_mV", "Cai_uM", "t_mech_ms", "Lneg", "force", "gelForce", "pCaCoeff")}


def metrics(truth: np.ndarray, pred: np.ndarray) -> dict:
    ae = np.abs(truth-pred)
    return {"MAE": dict(zip(FIELDS, map(float, ae.mean(0)))),
            "median_AE": dict(zip(FIELDS, map(float, np.median(ae, axis=0)))),
            "max_AE": dict(zip(FIELDS, map(float, ae.max(0))))}


def main() -> None:
    base = json.loads(INPUT.read_text(encoding="utf-8"))
    rows = base["rows"]
    truth = np.array([[r["l10"][f] for f in FIELDS] for r in rows])
    comparisons = {
        "direct_low_ridge": np.array(base["predictions"]["low_ridge_LOO"]),
        "direct_low_mid_ridge": np.array(base["predictions"]["low_mid_ridge_LOO"]),
        "linear_extrapolation": np.array(base["predictions"]["per_variant_linear_extrapolation"]),
        "direct_low_RBF_postscore": np.array(base["predictions"]["low_RBF_LOO_postscore"]),
        "direct_low_mid_RBF_postscore": np.array(base["predictions"]["low_mid_RBF_LOO_postscore"]),
    }
    for mode in ("low", "lowmid"):
        pred = []
        for i in range(len(rows)):
            path = PRED / f"twin_{i:02d}_{mode}.mat"
            if not path.exists():
                raise FileNotFoundError(path)
            score = beats(read_native(path))[-1]
            pred.append([score[f] for f in FIELDS])
        comparisons[f"native_ODE_twin_{mode}"] = np.array(pred)
    scores = {name: metrics(truth, pred) for name, pred in comparisons.items()}
    ae_mid_twin = np.abs(truth-comparisons["native_ODE_twin_lowmid"])
    paired = {}
    for comparator in ("native_ODE_twin_low", "direct_low_mid_ridge", "linear_extrapolation",
                       "direct_low_mid_RBF_postscore"):
        ae_comp = np.abs(truth-comparisons[comparator])
        paired[comparator] = {
            f: {"twin_lowmid_better": int(np.sum(ae_mid_twin[:, k] < ae_comp[:, k])),
                "twin_lowmid_worse": int(np.sum(ae_mid_twin[:, k] > ae_comp[:, k])),
                "mean_absolute_error_difference": float(np.mean(ae_comp[:, k]-ae_mid_twin[:, k]))}
            for k, f in enumerate(FIELDS)}
    result = {"n_variants": len(rows),
              "strictness": "Each held-out variant excluded from calibration emulator; high load simulated with native ODE at inferred parameters",
              "scores": scores, "paired_comparison": paired,
              "predictions": {k: v.tolist() for k, v in comparisons.items()},
              "truth": truth.tolist(),
              "interpretation_boundary": "Two-parameter deterministic grid in the same simulator family; no experimental or out-of-family validation."}
    (ROOT / "reverse_twin25_score.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"scores": scores, "paired_comparison": paired}, indent=2))


if __name__ == "__main__":
    main()
