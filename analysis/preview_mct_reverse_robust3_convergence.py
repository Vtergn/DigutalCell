"""Read-only progress check on completed 40-beat MCT files."""
from pathlib import Path

import h5py
import numpy as np

from score_mct2024_reference import beats


folder = Path(__file__).resolve().parent / "mct_feedback_audit/reverse_robust3/native_cohort"
results = []
for path in sorted(folder.glob("*.mat")):
    with h5py.File(path) as f:
        group = f["a"]
        case = {k: group[k][()].ravel() for k in
                ("t_ms", "Vm_mV", "Cai_uM", "t_mech_ms", "Lneg", "force", "gelForce", "pCaCoeff")}
    final = beats(case, final_beat=40)
    apd_range = float(np.ptp([x["APD_above_minus67_mV_ms"] for x in final]))
    ca_range = float(np.ptp([x["Ca_peak_uM"] for x in final])/final[-1]["Ca_peak_uM"])
    results.append((path.name, apd_range, ca_range))
fail = [x for x in results if x[1] > 5 or x[2] > .01]
print("scored", len(results), "failed", len(fail))
print("max APD range ms", max((x[1] for x in results), default=None))
print("max relative Ca range", max((x[2] for x in results), default=None))
print("first failures", fail[:8])
