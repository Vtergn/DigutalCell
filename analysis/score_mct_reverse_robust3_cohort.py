"""Score the frozen 40-beat, three-parameter MCT cohort and convergence gate."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import numpy as np

from score_mct2024_reference import beats


ROOT = Path(__file__).resolve().parent / "mct_feedback_audit" / "reverse_robust3"
LOADS = ("l02", "l06", "l10")
FIELDS = ("APD_above_minus67_mV_ms", "Ca_peak_uM")
EXTRA = ("gel_stress_peak_kPa", "pCa_multiplier_peak", "force_peak_kPa")


def read_case(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path) as f:
        group = f["a"]
        return {k: group[k][()].ravel() for k in
                ("t_ms", "Vm_mV", "Cai_uM", "t_mech_ms", "Lneg", "force", "gelForce", "pCaCoeff")}


def main() -> None:
    with (ROOT / "design.csv").open(newline="", encoding="utf-8") as f:
        design = list(csv.DictReader(f))
    rows = []
    for design_row in design:
        row = {k: design_row[k] for k in ("id", "split")}
        row.update({k: float(design_row[k]) for k in ("mct_gain", "pCa0_scale", "serca_scale")})
        for load in LOADS:
            path = ROOT / "native_cohort" / f"{row['id']}_{load}.mat"
            if not path.exists():
                raise FileNotFoundError(path)
            last_four = beats(read_case(path), final_beat=40)
            values = {k: float(last_four[-1][k]) for k in FIELDS + EXTRA}
            apd_range = float(np.ptp([b[FIELDS[0]] for b in last_four]))
            ca_range = float(np.ptp([b[FIELDS[1]] for b in last_four]))
            rel_ca_range = ca_range / max(values[FIELDS[1]], 1e-9)
            row[load] = {**values, "beat37_to40_APD_range_ms": apd_range,
                         "beat37_to40_Ca_range_fraction": rel_ca_range,
                         "convergence_pass": bool(apd_range <= 5 and rel_ca_range <= .01)}
        rows.append(row)
    failures = [{"id": r["id"], "split": r["split"], "load": load,
                 "APD_range_ms": r[load]["beat37_to40_APD_range_ms"],
                 "Ca_range_fraction": r[load]["beat37_to40_Ca_range_fraction"]}
                for r in rows for load in LOADS if not r[load]["convergence_pass"]]
    result = {"design": "48 train and 16 off-grid test variants; 3 parameters; 3 loads; 40 beats",
              "convergence_gate": "last-four-beat APD range <=5ms AND Ca peak range <=1%",
              "convergence_failures": failures,
              "all_conditions_pass_convergence": len(failures) == 0,
              "rows": rows}
    (ROOT / "cohort_score.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"n": len(rows), "convergence_fail_count": len(failures),
                      "first_failures": failures[:10]}, indent=2))


if __name__ == "__main__":
    main()
