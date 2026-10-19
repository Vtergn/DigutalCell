"""Native ODE check of selected three-parameter inverse calibrations."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from score_mct2024_reference import beats


ROOT = Path(__file__).resolve().parent / "mct_feedback_audit" / "reverse_robust3"
FIELDS = ("APD_above_minus67_mV_ms", "Ca_peak_uM")
IDS = ("test_000", "test_005", "test_010", "test_015")


def read_case(path: Path):
    with h5py.File(path) as f:
        g = f["a"]
        return {k: g[k][()].ravel() for k in
                ("t_ms", "Vm_mV", "Cai_uM", "t_mech_ms", "Lneg", "force", "gelForce", "pCaCoeff")}


def main() -> None:
    cohort = json.loads((ROOT / "cohort_score.json").read_text(encoding="utf-8"))
    lookup = {r["id"]: r for r in cohort["rows"]}
    rows = []
    for ident in IDS:
        for mode, loads in (("clean_low", ("l02",)),
                            ("clean_lowmid", ("l02", "l06"))):
            for load in loads:
                path = ROOT / "calibration_audit" / f"{ident}_{mode}_{load}.mat"
                if not path.exists():
                    raise FileNotFoundError(path)
                predicted = beats(read_case(path), final_beat=40)[-1]
                observed = lookup[ident][load]
                rows.append({"id": ident, "mode": mode, "load": load,
                             "absolute_errors": {field: abs(predicted[field]-observed[field])
                                                 for field in FIELDS}})
    summary = {}
    for mode in ("clean_low", "clean_lowmid"):
        subset = [r for r in rows if r["mode"] == mode]
        summary[mode] = {field: {"mean_AE": float(np.mean([r["absolute_errors"][field] for r in subset])),
                                 "max_AE": float(np.max([r["absolute_errors"][field] for r in subset]))}
                         for field in FIELDS}
    result = {"prespecified_ids": IDS, "rows": rows, "summary": summary,
              "scope": "Four fixed held-out test identities; descriptive native ODE calibration audit."}
    (ROOT / "calibration_audit_score.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
