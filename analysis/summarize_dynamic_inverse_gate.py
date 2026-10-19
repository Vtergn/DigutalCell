"""Paired-cell bootstrap of inverse-state mechanism classification."""
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

root = Path(__file__).resolve().parent / "human_bidirectional_output"
new = np.loadtxt(root / "dynamic_inverse_gate.csv", delimiter=",")
old = np.loadtxt(root / "dynamic_flux_pilot_input.csv", delimiter=",",
                 skiprows=1)
cell_ids = np.unique(new[:, 0].astype(int))
assert len(cell_ids) == 45
rng = np.random.default_rng(20261002)


def auc_for(table, observation, sample):
    subset = table[np.isin(table[:, 0], sample)]
    if observation is not None:
        subset = subset[subset[:, 2] == observation]
    subset = subset[subset[:, 1] != 3]
    label = subset[:, 1] == 2
    if table.shape[1] == 7:
        contrast = subset[:, 4]  # previous beta estimate
    else:
        contrast = np.log(np.maximum(subset[:, 5], .02) /
                          np.maximum(subset[:, 6], .02))
    return roc_auc_score(label, contrast)


for label, table, kind in [("prototype", old, None),
                           ("inverse_baseline", new, 0),
                           ("inverse_C1C2", new, 1)]:
    vals = []
    for _ in range(5000):
        sample = rng.choice(cell_ids, size=len(cell_ids), replace=True)
        # Include multiplicities by rebuilding the sampled cell blocks.
        blocks = [table[table[:, 0] == i] for i in sample]
        stack = np.concatenate(blocks)
        vals.append(auc_for(stack, kind, sample))
    point = auc_for(table, kind, cell_ids)
    print(label, "AUC", round(point, 4), "95% cell bootstrap",
          np.quantile(vals, [.025, .975]).round(4))
