"""Diagnose the predeclared 40-beat convergence failures at 60 beats."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from score_mct2024_reference import beats


ROOT = Path(__file__).resolve().parent / "mct_feedback_audit" / "reverse_robust3"


def read_case(path: Path):
    with h5py.File(path) as f:
        group = f["a"]
        return {k: group[k][()].ravel() for k in
                ("t_ms", "Vm_mV", "Cai_uM", "t_mech_ms", "Lneg", "force", "gelForce", "pCaCoeff")}


def main() -> None:
    cohort = json.loads((ROOT / "cohort_score.json").read_text(encoding="utf-8"))
    lookup = {r["id"]: r for r in cohort["rows"]}
    output = []
    for failed in cohort["convergence_failures"]:
        ident, load = failed["id"], failed["load"]
        path = ROOT / "failed_60beat" / f"{ident}_{load}_60.mat"
        if not path.exists():
            raise FileNotFoundError(path)
        b = beats(read_case(path), final_beat=60)
        apd_range = float(np.ptp([x["APD_above_minus67_mV_ms"] for x in b]))
        ca_range = float(np.ptp([x["Ca_peak_uM"] for x in b])/b[-1]["Ca_peak_uM"])
        output.append({"id": ident, "load": load,
                       "APD_last4_range_ms": apd_range,
                       "Ca_last4_range_fraction": ca_range,
                       "passes_60beat_convergence": apd_range <= 5 and ca_range <= .01,
                       "APD_beat40_to60_difference_ms": float(b[-1]["APD_above_minus67_mV_ms"]-
                                                               lookup[ident][load]["APD_above_minus67_mV_ms"]),
                       "Ca_beat40_to60_difference_uM": float(b[-1]["Ca_peak_uM"]-
                                                             lookup[ident][load]["Ca_peak_uM"])})
    result = {"selected_40beat_failures": output,
              "all_failed_conditions_pass_at60": all(r["passes_60beat_convergence"] for r in output),
              "interpretation": "Selected-condition diagnostic only; not the all-condition 60-beat rerun required to pass the frozen gate."}
    (ROOT / "failed60_score.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
