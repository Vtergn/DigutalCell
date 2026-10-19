"""Apply the previously frozen C1/C2 SAC-flux slope rule to improved theta."""
from pathlib import Path
import json

import numpy as np
from sklearn.metrics import confusion_matrix

OUT = Path(__file__).resolve().parent / "human_bidirectional_output"
gate = np.loadtxt(OUT / "dynamic_inverse_gate.csv", delimiter=",")
pilot = json.loads((OUT / "dynamic_flux_pilot_metadata.json").read_text())
shared = pilot["shared_beta"]
assert shared > 0 and gate.shape[1] == 8
rows = []
for row in gate:
    g1, g2, presence = row[5], row[6], row[4]
    if presence < .05 or not np.isfinite(g1) or g1 <= 0:
        beta = 0.0
    else:
        individual = float(np.clip(np.log(max(g1, .02)/max(g2, .02))/.05,
                                   -10, 15))
        beta = (.75*shared + .25*individual
                if individual > shared/2 else 0.0)
    rows.append([*row[:4], presence, beta])
arr = np.asarray(rows)
np.savetxt(OUT / "dynamic_inverse_residual_input.csv", arr,
           delimiter=",", fmt="%.12g")
summary = []
for obs in [0, 1]:
    s = arr[arr[:, 2] == obs]
    y = s[:, 1] == 2
    pred = s[:, 5] > 0
    summary.append({"observation_kind": obs,
                    "saturation_confusion_present_only":
                        confusion_matrix(y[s[:, 1] != 3],
                                         pred[s[:, 1] != 3]).tolist(),
                    "active_fraction_by_form": [float(np.mean(
                        s[s[:, 1] == j, 5] > 0)) for j in [1, 2, 3]]})
(OUT / "dynamic_inverse_residual_preparation.json").write_text(
    json.dumps({"shared_beta": shared, "rule": pilot["beta_rule"],
                "summary": summary}, indent=2), encoding="utf-8")
print(summary)
