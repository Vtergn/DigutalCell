"""Paired cell bootstrap: physiological inverse versus prototype, C1/C2 versus baseline."""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "human_bidirectional_output"
data = json.loads((OUT / "dynamic_inverse_target_scores.json").read_text())
old = json.loads((OUT / "dynamic_flux_pilot_scores.json").read_text())
old_input = np.loadtxt(OUT / "dynamic_flux_pilot_input.csv", delimiter=",",
                       skiprows=1)
rng = np.random.default_rng(20261002)


def inverse(obs, form, target, method):
    r = next(x for x in data["results"] if x["observation_kind"] == obs
             and x["truth_form"] == form and x["target"] == target
             and x["method"] == method and x["alignment"] == "baseline_aligned")
    return dict(zip(r["cell_ids_zero_based"], np.asarray(r["cellwise_mae"])))


def prototype(form, target):
    r = next(x for x in old["results"] if x["truth_form"] == form
             and x["target"] == target and
             x["method"] == "prototype_plus_sac_gate" and
             x["alignment"] == "baseline_aligned")
    ids = old_input[np.asarray(r["row_ids"]), 0].astype(int)-1
    return dict(zip(ids,
                    np.asarray(r["cellwise_mae"])))


def compare(name, a, b):
    common = sorted(set(a) & set(b))
    if not common:
        return
    d = np.stack([a[i]-b[i] for i in common])
    draw = rng.integers(0, len(d), (10000, len(d)))
    ci = np.quantile(d[draw].mean(axis=1), [.025, .975], axis=0).T
    print(name, "n", len(d), "A-B", np.round(d.mean(0), 4),
          "CI", np.round(ci, 4))


for form in [0, 1, 2]:
    for target in [0, 1, 2]:
        p = prototype(form, target)
        b = inverse(0, form, target, "native_physics_gate")
        c = inverse(1, form, target, "native_physics_gate")
        compare(f"form{form} target{target} baseline_inverse-v-prototype", b, p)
        compare(f"form{form} target{target} C1C2_inverse-v-baseline_inverse", c, b)
