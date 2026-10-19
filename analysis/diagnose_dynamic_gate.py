"""Diagnostic: how much gate/form information survives parameter uncertainty?"""
from pathlib import Path

import h5py
import numpy as np
from sklearn.metrics import roc_auc_score

out = Path(__file__).resolve().parent / "human_bidirectional_output"
tab = np.loadtxt(out / "dynamic_flux_pilot_input.csv", delimiter=",", skiprows=1)
form = tab[:, 1].astype(int)
print("few-shot SAC-presence AUC", roc_auc_score(form != 3, tab[:, 3]))
print("few-shot saturation-form AUC", roc_auc_score(
    form[form != 3] == 2, tab[form != 3, 4]))

# This second analysis intentionally uses same-cell true-parameter source and
# no-SAC traces. It is an oracle identifiability diagnostic, not a predictor.
with h5py.File(out / "dynamic_discrepancy_cohort.mat") as f:
    w = f["waveforms"][:]
    valid = f["ok"][:].astype(bool).all(axis=1)
w = w[valid]
g = []
for mechanism in [0, 1, 2]:
    gains = []
    for j, onset in [(1, 150), (2, 600)]:
        scale = np.array([5.0, .02])[None, :, None]
        source = ((w[:, 0, j, :2, onset:onset+110] -
                   w[:, 2, j, :2, onset:onset+110]) / scale).reshape(len(w), -1)
        observed = ((w[:, mechanism, j, :2, onset:onset+110] -
                     w[:, 2, j, :2, onset:onset+110]) / scale).reshape(len(w), -1)
        gains.append((source*observed).sum(axis=1) /
                     (source*source).sum(axis=1))
    g.append(np.stack(gains, axis=1))

print("oracle same-cell median C1/C2 gains", [
    np.median(x, axis=0).round(4).tolist() for x in g])
contrast = np.r_[np.log(g[0][:, 0]/g[0][:, 1]),
                 np.log(g[1][:, 0]/g[1][:, 1])]
print("oracle same-cell saturation-form AUC", roc_auc_score(
    np.r_[np.zeros(len(w)), np.ones(len(w))], contrast))
