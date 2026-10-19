"""Synthetic practical-observability check under three measurement sets.

The data are synthetic, not a fit to Duke/ETH/eLife. Exact gauge invariance is
preserved even after adding output noise; calibrated tension observations can
remove the corresponding symmetry in a *same-cell* experiment.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

HERE = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260928)
F = np.linspace(0, 450, 30)
THETA = dict(T0=1.3, lam=.007, T50=2.5, k=.8, G=200.)
SD_I, SD_T = 5., .03


def predict(theta):
    local = theta["T0"] + theta["lam"] * F
    return (theta["G"] * expit((local-theta["T50"])/theta["k"]),
            theta["lam"] * F, local)


def main():
    ref_i, ref_delta, ref_abs = predict(THETA)
    obs_i = ref_i + RNG.normal(0, SD_I, F.size)
    obs_delta = ref_delta + RNG.normal(0, SD_T, F.size)
    obs_abs = ref_abs + RNG.normal(0, SD_T, F.size)
    scales = np.linspace(.65, 1.35, 71)
    offsets = np.linspace(-.7, .7, 71)
    losses = {name: np.empty((len(offsets), len(scales)))
              for name in ("F_I", "F_deltaT_I", "F_Tabs_I")}
    for row, b in enumerate(offsets):
        for col, a in enumerate(scales):
            transformed = dict(T0=a*THETA["T0"]+b, lam=a*THETA["lam"],
                               T50=a*THETA["T50"]+b, k=a*THETA["k"], G=THETA["G"])
            p_i, p_delta, p_abs = predict(transformed)
            i_loss = np.sum(((obs_i-p_i)/SD_I)**2)
            losses["F_I"][row, col] = i_loss
            losses["F_deltaT_I"][row, col] = i_loss + np.sum(((obs_delta-p_delta)/SD_T)**2)
            losses["F_Tabs_I"][row, col] = i_loss + np.sum(((obs_abs-p_abs)/SD_T)**2)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), constrained_layout=True)
    result = {"synthetic": True, "noise_sd_pA": SD_I, "tension_noise_sd_mN_m": SD_T,
              "force_nN": [float(F.min()), float(F.max())], "profiles": {}}
    for ax, (name, surface) in zip(axes, losses.items()):
        offset_surface = surface - np.min(surface)
        # Plot truncated excess chi-square so the flat directions are visible.
        im = ax.imshow(np.minimum(offset_surface, 100), origin="lower", aspect="auto",
                       extent=[scales[0], scales[-1], offsets[0], offsets[-1]],
                       cmap="viridis", vmin=0, vmax=100)
        ax.plot([1], [0], "rx", markersize=7)
        ax.set(title=name.replace("_", "+"), xlabel="tension scale a", ylabel="tension offset b")
        result["profiles"][name] = {
            "loss_range_along_a_at_b0": float(np.ptp(surface[len(offsets)//2, :])),
            "loss_range_along_b_at_a1": float(np.ptp(surface[:, len(scales)//2])),
            "grid_minimum_a_b": [float(scales[np.unravel_index(np.argmin(surface),
                                                                  surface.shape)[1]]),
                                 float(offsets[np.unravel_index(np.argmin(surface),
                                                                   surface.shape)[0]])],
        }
    fig.colorbar(im, ax=axes, shrink=.8, label="excess chi-square (capped at 100)")
    fig.savefig(HERE / "observability_noise_demo.png", dpi=180)
    (HERE / "observability_noise_demo.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
