"""Paired uncertainty and robust summaries for the locked SAC decomposition."""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "human_bidirectional_output"
data = json.loads((OUT / "dynamic_model_selection_scores.json").read_text())
previous = json.loads((OUT / "dynamic_inverse_target_scores.json").read_text())
rng = np.random.default_rng(20261002)


def paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    idx = rng.integers(n, size=(10000, n))
    return {"mean_difference": float(d.mean()),
            "median_difference": float(np.median(d)),
            "mean_difference_ci95": np.quantile(d[idx].mean(axis=1), [.025, .975]).tolist(),
            "n_better_a": int((d < 0).sum()), "n": n}


def robust(e):
    e = np.asarray(e)
    return {"mean": float(e.mean()), "median": float(np.median(e)),
            "p90": float(np.quantile(e, .9)),
            "n_above_5_mV": int((e > 5).sum())}


summary = {"decomposition": [], "selection": [], "previous_gate_comparison": []}
for d in data["decomposition"]:
    if d["observation_kind"] != 1:
        continue
    arms = {k: np.asarray(v["cellwise_mae"])[:, 0] for k, v in d["arms"].items()}
    summary["decomposition"].append({
        "target": d["target"], "n": d["n_common"],
        "arms": {k: robust(v) for k, v in arms.items()},
        "correct_form_minus_wrong_form_at_theta_hat": paired(
            arms["theta_hat_saturating_SAC"], arms["theta_hat_original_SAC"]),
        "wrong_form_true_theta_minus_correct_form_theta_hat": paired(
            arms["theta_true_original_SAC"], arms["theta_hat_saturating_SAC"]),
    })

records = data["selection_records"]
for form in range(3):
    for target in range(3):
        found = {r["method"]: r for r in records
                 if r.get("observation_kind") == 1 and r.get("truth_form") == form
                 and r.get("target") == target}
        a, b = found["fixed_gain_rule"], found["correct_form_at_theta_hat"]
        assert a["cell_ids"] == b["cell_ids"]
        av = np.asarray(a["cellwise_mae"])[:, 0]
        bv = np.asarray(b["cellwise_mae"])[:, 0]
        summary["selection"].append({"truth_form": form, "target": target,
                                     "n": a["n"], "fixed": robust(av),
                                     "correct_form": robust(bv),
                                     "fixed_minus_correct": paired(av, bv)})
        if form != 1:
            continue
        old = next(x for x in previous["results"]
                   if x["observation_kind"] == 1 and x["truth_form"] == form
                   and x["target"] == target and x["method"] == "native_physics_gate"
                   and x["alignment"] == "baseline_aligned")
        id_key = "cell_ids_zero_based"
        old_by_cell = dict(zip(old[id_key], np.asarray(old["cellwise_mae"])[:, 0]))
        common = [i for i, cell in enumerate(a["cell_ids"]) if cell in old_by_cell]
        old_e = np.array([old_by_cell[a["cell_ids"][i]] for i in common])
        summary["previous_gate_comparison"].append({
            "target": target, "n_common": len(common),
            "continuous_gate": robust(old_e), "fixed_selection": robust(av[common]),
            "fixed_minus_gate": paired(av[common], old_e)})

(OUT / "dynamic_model_selection_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8")
for name, rows in summary.items():
    print(name)
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))
