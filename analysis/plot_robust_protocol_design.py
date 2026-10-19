"""Plot the preregistered synthetic protocol comparison from JSON output."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
data = json.loads((HERE / "robust_protocol_results.json").read_text(encoding="utf-8"))
summ = data["summary"]
families = list(data["development_families"])+list(data["unseen_families"])
methods = ("duke_like", "random_diverse", "local_fisher", "robust_ensemble")
names = ("Monotonic", "Balanced random", "Local Fisher", "Ensemble")
colors = ("#9198a4", "#26876d", "#d99036", "#3a61ab")

fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.5), gridspec_kw={"width_ratios": [1.65, 1]})
xx = np.arange(len(families))
for j, (method, name, color) in enumerate(zip(methods, names, colors)):
    values = [summ["development" if f in data["development_families"] else "unseen"]
              [method]["per_family_mae"][f] for f in families]
    ax[0].bar(xx+(j-1.5)*.19, values, width=.18, label=name, color=color)
ax[0].axvline(3.5, color="black", lw=.8, ls="--")
ax[0].set_xticks(xx, [x.replace("_", "\n") for x in families], fontsize=8)
ax[0].set_ylabel("Held-out challenge peak MAE (simulator units)")
ax[0].set_title("Four development families | Four unseen families")
ax[0].legend(frameon=False, ncol=2, fontsize=8)
ax[0].grid(axis="y", alpha=.18)

width = .19
for j, (method, name, color) in enumerate(zip(methods, names, colors)):
    values = [summ[s][method]["short_interval_fraction"] for s in ("development", "unseen")]
    ax[1].bar(np.arange(2)+(j-1.5)*width, values, width=width, color=color)
ax[1].set_xticks([0, 1], ["Development", "Unseen"])
ax[1].set_ylim(0, 1)
ax[1].set_ylabel("Fraction of eligible shots at 50 ms")
ax[1].set_title("Short interval sampling")
ax[1].grid(axis="y", alpha=.18)
fig.suptitle("Sequential protocol design under mechanistic mismatch", fontsize=13)
fig.tight_layout()
fig.savefig(HERE / "robust_protocol_design.png", dpi=180)
