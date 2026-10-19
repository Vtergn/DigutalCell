"""Cell-bootstrap comparisons for locked dynamic discrepancy scores."""
import json
from pathlib import Path

import numpy as np

path = Path(__file__).resolve().parent / "human_bidirectional_output"
data = json.loads((path / "dynamic_discrepancy_scores.json").read_text())
scores = data["scores"]
rng = np.random.default_rng(20261002)
pairs = [
    ("matched_parameter_wrong_physics", "linear_transport"),
    ("matched_parameter_wrong_physics", "direct_ridge_sequence"),
    ("calibrated_prior_gain", "linear_transport"),
]
print("n", data["n_valid_all_forms"])
for protocol in range(3):
    print("TARGET", protocol+1)
    for a, b in pairs:
        # Positive means a is worse than b. Saturating-SAC truth only.
        x = np.array(scores[a][protocol]["cellwise_target_window_mae"])[:, 1, :]
        y = np.array(scores[b][protocol]["cellwise_target_window_mae"])[:, 1, :]
        d = x-y
        draw = rng.integers(0, len(d), (10000, len(d)))
        ci = np.quantile(d[draw].mean(axis=1), [.025, .975], axis=0).T
        print(a, "minus", b, "mean", np.round(d.mean(axis=0), 4),
              "95%CI", np.round(ci, 4))
