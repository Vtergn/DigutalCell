"""Check whether inferred parameters reproduce observed calibration data in the native ODE."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from score_mct2024_reference import beats


ROOT = Path(__file__).resolve().parent / "mct_feedback_audit"
DATA = ROOT / "reverse_calibration_audit"
FIELDS = ("APD_above_minus67_mV_ms", "Ca_peak_uM")


def read_case(path: Path):
    with h5py.File(path) as f:
        return {k: f["a"][k][()].ravel() for k in
                ("t_ms", "Vm_mV", "Cai_uM", "t_mech_ms", "Lneg", "force", "gelForce", "pCaCoeff")}


def main() -> None:
    base = json.loads((ROOT / "reverse_grid25_score.json").read_text(encoding="utf-8"))
    rows = base["rows"]
    out = []
    for mode, indices, loads in (("low", (3, 21, 2, 15, 22), ("l02",)),
                                 ("lowmid", (0, 6, 12, 18, 24), ("l02", "l06"))):
        for i in indices:
            for load in loads:
                path = DATA / f"audit_{i:02d}_{mode}_{load}.mat"
                if not path.exists():
                    raise FileNotFoundError(path)
                score = beats(read_case(path))[-1]
                target = rows[i][load]
                out.append({"id": rows[i]["id"], "mode": mode, "load": load,
                            "absolute_errors": {f: abs(float(score[f])-target[f]) for f in FIELDS}})
    summary = {}
    for mode in ("low", "lowmid"):
        selected = [r for r in out if r["mode"] == mode]
        summary[mode] = {f: {"mean_AE": float(np.mean([r["absolute_errors"][f] for r in selected])),
                             "max_AE": float(np.max([r["absolute_errors"][f] for r in selected]))}
                         for f in FIELDS}
    result = {"selected_cases": out, "summary": summary,
              "selection": "Five largest low-only gain errors; five grid-spread low+mid variants. Diagnostic, not random sample."}
    (ROOT / "reverse_calibration_audit_score.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
