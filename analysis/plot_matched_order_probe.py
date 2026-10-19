"""Visual summary of matched-action order and anchor-interval sensitivity."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader((HERE/"matched_order_probe_cells.csv").open(encoding="utf-8")))
anchor = json.loads((HERE/"matched_order_anchor_interval.json").read_text(encoding="utf-8"))
grouped = defaultdict(list)
for row in rows:
    grouped[(row["family"], int(row["cell"]))].append(row)
families = ("no_history", "single_sigmoid", "dual_sigmoid", "force_fatigue",
            "mechanical_hysteresis", "late_gain_switch")
true_span, pred_span = [], []
for family in families:
    t, p = [], []
    for cell in range(40):
        values = grouped[(family, cell)]
        t.append(np.ptp([float(x["true_challenge_peak"]) for x in values]))
        p.append(np.ptp([float(x["pred_challenge_peak"]) for x in values]))
    true_span.append(np.median(t))
    pred_span.append(np.median(p))

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.35, 1]})
x = np.arange(len(families))
axes[0].bar(x-.18, true_span, width=.35, label="True seventh peak", color="#2f8b7a")
axes[0].bar(x+.18, pred_span, width=.35, label="Predicted seventh peak", color="#d9913a")
axes[0].set_xticks(x, [f.replace("_", "\n") for f in families], fontsize=8)
axes[0].set_ylabel("Median range across 24 orders\n(normalized peak units)")
axes[0].set_title("Same action set, different order")
axes[0].legend(frameon=False, fontsize=8)
axes[0].grid(axis="y", alpha=.2)

for family, color in (("no_history", "#858d96"), ("single_sigmoid", "#2f8b7a"),
                      ("dual_sigmoid", "#4865aa"), ("force_fatigue", "#c47935")):
    y = [100*anchor[family][str(dt)] for dt in (50, 200, 800, 5000)]
    axes[1].plot([50, 200, 800, 5000], y, "o-", label=family.replace("_", " "), color=color)
axes[1].set_xscale("log")
axes[1].set_xticks([50, 200, 800, 5000], ["50", "200", "800", "5000"])
axes[1].set_ylabel("Median true peak range across orders (%)")
axes[1].set_xlabel("Interval before common anchor (ms)")
axes[1].set_title("An anchor can hide or reveal memory")
axes[1].legend(frameon=False, fontsize=8)
axes[1].grid(alpha=.2)
fig.suptitle("Matched mechanical stimuli isolate order from stimulus diversity")
fig.tight_layout()
fig.savefig(HERE/"matched_order_probe.png", dpi=180)
