"""Fixed paired bootstrap of incremental gate and residual effects."""
import json
from pathlib import Path

import numpy as np

out = Path(__file__).resolve().parent / "human_bidirectional_output"
data = json.loads((out / "dynamic_flux_pilot_scores.json").read_text())
rows = data["results"]
rng = np.random.default_rng(20261002)


def get(form: int, target: int, method: str):
    found = [r for r in rows if r["truth_form"] == form and
             r["target"] == target and r["method"] == method and
             r["alignment"] == "baseline_aligned"]
    assert len(found) == 1
    return {i: np.array(e) for i, e in zip(found[0]["row_ids"],
                                            found[0]["cellwise_mae"])}


for form in range(3):
    for target in range(3):
        physics = get(form, target, "personalized_physics_prototype")
        gate = get(form, target, "prototype_plus_sac_gate")
        residual = get(form, target, "prototype_gate_flux_residual")
        paired = sorted(set(physics) & set(gate) & set(residual))
        if not paired:
            continue
        physics = np.stack([physics[i] for i in paired])
        gate = np.stack([gate[i] for i in paired])
        residual = np.stack([residual[i] for i in paired])
        for label, a, b in [("gate_minus_physics", gate, physics),
                            ("residual_minus_gate", residual, gate)]:
            diff = a-b
            draws = rng.integers(0, len(diff), (10000, len(diff)))
            ci = np.quantile(diff[draws].mean(axis=1), [.025, .975], axis=0).T
            print(form, target, label, "n", len(paired),
                  "mean", np.round(diff.mean(0), 4),
                  "95%CI", np.round(ci, 4))
