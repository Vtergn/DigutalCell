"""Paired seed/family conditional uncertainty and plot for risk design benchmark."""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader((HERE/"predictive_risk_cells.csv").open(encoding="utf-8")))
data = json.loads((HERE/"predictive_risk_results.json").read_text(encoding="utf-8"))
methods = ("monotonic", "balanced_random", "static_maximin", "local_fisher", "predictive_minimax")
unseen = data["unseen_families"]
rng = np.random.default_rng(1031)


def values(seed, family, method):
    return np.array([float(r["abs_challenge_error"]) for r in rows
                     if int(r["seed"]) == seed and r["family"] == family and r["method"] == method])


paired = {}
for reference in ("balanced_random", "static_maximin", "monotonic"):
    blocks = [values(seed, family, "predictive_minimax")-values(seed, family, reference)
              for seed in data["seeds"] for family in unseen]
    bs = np.array([np.mean([np.mean(block[rng.integers(len(block), size=len(block))])
                            for block in blocks]) for _ in range(4000)])
    paired[reference] = {"mean_difference": float(np.mean(np.concatenate(blocks))),
                         "cell_bootstrap_95pct_fixed_seeds_and_families": [float(x) for x in np.quantile(bs, [.025, .975])]}

seed_summary = {str(seed): {m: {"median": float(np.median(np.concatenate([values(seed, f, m) for f in unseen]))),
                               "worst_family_mean": float(max(np.mean(values(seed, f, m)) for f in unseen))}
                            for m in methods} for seed in data["seeds"]}
out = {"paired_unseen": paired, "per_seed_unseen": seed_summary}
(HERE/"predictive_risk_uncertainty.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

labels = ("Monotonic", "Balanced\nrandom", "Static\nmaximin", "Local\nFisher", "Predictive\nminimax")
colors = ("#777f89", "#258a75", "#6275a7", "#de9634", "#b84b4b")
fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3))
medians = [data["summary"]["unseen"][m]["median_error"] for m in methods]
worst = [data["summary"]["unseen"][m]["worst_family_mean_error"] for m in methods]
for ax, y, title in zip(axes, (medians, worst),
                        ("Median seventh-challenge absolute error", "Worst unseen-family mean error")):
    ax.bar(np.arange(5), y, color=colors)
    ax.set_xticks(np.arange(5), labels, fontsize=9)
    ax.set_title(title)
    ax.set_ylabel("Normalized peak-current units")
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
fig.suptitle("Coverage-constrained predictive design: prespecified go/no-go")
fig.tight_layout()
fig.savefig(HERE/"predictive_risk_go_no_go.png", dpi=180)
print(json.dumps(out, indent=2))
