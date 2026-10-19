"""Choose four diverse training cells for true-ODE local sensitivity checks."""
import csv
import json
from pathlib import Path

import h5py
import numpy as np

root = Path(__file__).resolve().parents[1]
cohort = root / "analysis/virtual_cardiomyocyte_5000/merged_cohort.h5"
out = root / "analysis/virtual_cardiomyocyte_5000/jacobian_bases.csv"
with h5py.File(cohort) as f:
    theta, split = f["theta"][()], f["split_code"][()]
    lo, hi = f["lo"][()], f["hi"][()]
    lam = f["lambda"][()]
    idx = int(np.argmin(abs(lam-1.1)))
    peak = f["Ta_kPa"][:, idx, :].max(axis=1)
unit = (theta-lo)/(hi-lo)
eligible = np.flatnonzero((split == 0) & ((unit > .05) & (unit < .95)).all(axis=1))
assert len(eligible) > 20
ordered = eligible[np.argsort(peak[eligible])]
chosen = ordered[(np.array([.12,.36,.64,.88])*(len(ordered)-1)).astype(int)]
with out.open("w",newline="") as f:
    writer=csv.writer(f)
    writer.writerow(["cell_index_zero_based"]+[f"theta_{i+1}" for i in range(12)] +
                    [f"lo_{i+1}" for i in range(12)] + [f"hi_{i+1}" for i in range(12)])
    for cell in chosen:
        writer.writerow([int(cell)]+theta[cell].tolist()+lo.tolist()+hi.tolist())
out.with_suffix(".json").write_text(json.dumps({"cell_indices":chosen.tolist(),
    "Tpeak_lambda_1p1_kPa":peak[chosen].tolist(),
    "chosen_only_from_training_split":True,
    "prior_margin_each_parameter":.05},indent=2),encoding="utf-8")
print(chosen.tolist(),peak[chosen].round(2).tolist())
