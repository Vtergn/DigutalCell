"""Compare standard and tightened ode15s settings for the candidate alternans."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from score_mct2024_reference import beats


ROOT = Path(__file__).resolve().parent / "mct_feedback_audit/reverse_robust3"


def read(path: Path):
    with h5py.File(path) as f:
        g = f["a"]
        case = {k: g[k][()].ravel() for k in
                ("t_ms", "Vm_mV", "Cai_uM", "t_mech_ms", "Lneg", "force", "gelForce", "pCaCoeff")}
    return beats(case, final_beat=100, window=10)


def summarize(rows):
    apd = np.array([r["APD_above_minus67_mV_ms"] for r in rows])
    ca = np.array([r["Ca_peak_uM"] for r in rows])
    return {"APD_last10_ms": apd.tolist(), "Ca_last10_uM": ca.tolist(),
            "APD_last10_range_ms": float(np.ptp(apd)),
            "Ca_last10_range_uM": float(np.ptp(ca)),
            "APD_even_odd_gap_ms": float(abs(np.mean(apd[1::2])-np.mean(apd[::2]))),
            "Ca_even_odd_gap_uM": float(abs(np.mean(ca[1::2])-np.mean(ca[::2])))}


def main() -> None:
    result = {
        "standard": summarize(read(ROOT / "train036_high100.mat")),
        "tight": summarize(read(ROOT / "train036_high100_tight.mat")),
        "solver_settings": {"standard": "RelTol=1e-5, MaxStep=1ms",
                            "tight": "RelTol=1e-6, MaxStep=0.5ms"},
        "boundary": "This checks solver-setting sensitivity of one trajectory only; it does not validate biological alternans or exclude all numerical artifacts.",
    }
    (ROOT / "train036_solver_sensitivity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: {"APD_even_odd_gap_ms": v["APD_even_odd_gap_ms"],
                          "Ca_even_odd_gap_uM": v["Ca_even_odd_gap_uM"]}
                      for k, v in result.items() if k in ("standard", "tight")}, indent=2))


if __name__ == "__main__":
    main()
