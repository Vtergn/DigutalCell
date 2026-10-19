"""Fixed held-out 13-parameter native ODE closure inputs, matched source/altered."""

from pathlib import Path

import numpy as np

from score_human_feedback_observability import fit_predict

ROOT = Path(__file__).resolve().parent / "human_bidirectional_output"
s = np.loadtxt(ROOT / "feedback_13param_cohort.csv", delimiter=",")
a = np.loadtxt(ROOT / "feedback_13param_discrepancy.csv", delimiter=",")
valid = (s[:, -1] == 1) & (a[:, -1] == 1)
s, a = s[valid], a[valid]
assert s.shape == a.shape and np.allclose(s[:, :13], a[:, :13])


def unpack(z):
    theta, base = z[:, :13], z[:, 13:16]
    f = z[:, 16:-1].reshape(len(z), 12, 8).transpose(0, 2, 1)
    x = np.c_[base, f[:, 2, :]]  # fixed 300-ms, lambda 1.10 calibration
    y = f[:, 5:8, :][:, :, [0, 2, 3, 6, 8, 10]].reshape(len(z), -1)
    return theta, x, y


theta, sx, sy = unpack(s)
_, ax, ay = unpack(a)
rng = np.random.default_rng(777)
order = rng.permutation(len(s))
test, train = order[:12], order[12:]
lo = [.9, .85, .85, .85, .9, .8, .4, .75, 1.8, 4.5, 1.8, -4.1, 100]
hi = [1.1, 1.15, 1.15, 1.15, 1.1, 1.2, 1.6, .86, 2.2, 5.5, 2.8, -.9, 140]
rows = []
for scenario, x, y in [(0, sx, sy), (1, ax, ay)]:
    estimated = fit_predict(sx[train], theta[train], x[test])
    estimated = np.clip(estimated, lo, hi)
    direct = fit_predict(sx[train], sy[train], x[test])
    rows.append(np.c_[test + 1, np.full(len(test), scenario),
                      theta[test], estimated, y[test], direct])
rows = np.vstack(rows)
out = ROOT / "feedback_13param_native_input.csv"
np.savetxt(out, rows, delimiter=",", fmt="%.12g")
print("saved", out, "shape", rows.shape)
