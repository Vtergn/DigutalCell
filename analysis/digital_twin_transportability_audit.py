"""Observational equivalence versus counterfactual predictability.

Uses Duke's observed force ranges only as input support. Currents below are
synthetic; no Duke cell's T0, lambda, T50, k or G is fitted or inferred.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "Data" / "2022_pnas" / "2022_pnas" / "bychannel" / "bychannel_summary.csv"
BASE = dict(T0=1.3, lam=.007, T50=2.7, k=.7, G=200.)
SCALES = np.linspace(.5, 2., 61)
OFFSETS = np.linspace(-.5, .5, 41)
DELTA_T = .3  # mN/m; example perturbation, not measured in the Duke cells


def gauge(a, b):
    return dict(T0=a*BASE["T0"]+b, lam=a*BASE["lam"],
                T50=a*BASE["T50"]+b, k=a*BASE["k"], G=BASE["G"])


def current_force(f, p, extra_tension=0.):
    return p["G"] * expit((p["T0"] + p["lam"]*np.asarray(f) + extra_tension
                           - p["T50"]) / p["k"])


def current_clamp(t_abs, p):
    return p["G"] * expit((t_abs-p["T50"]) / p["k"])


def main():
    with DATA.open(newline="", encoding="utf-8-sig") as handle:
        cells = [r for r in csv.DictReader(handle)
                 if r["construct"] in {"mp1", "mp2"} and float(r["responds"]) == 1]
    peak_forces = np.asarray([float(r["peakf"]) for r in cells])
    force_grid = np.linspace(0, 650, 400)
    reference = current_force(force_grid, BASE)
    params = [gauge(a, b) for a in SCALES for b in OFFSETS]
    max_baseline_discrepancy = max(float(np.max(np.abs(current_force(force_grid, p)-reference)))
                                   for p in params)
    assert max_baseline_discrepancy < 1e-10

    # Two forms of proposed counterfactual require different observables:
    # an externally delivered calibrated delta tension anchors scale; an
    # absolute tension clamp also anchors the latent tension zero.
    force_challenge = 200.
    delta_values = np.asarray([current_force(force_challenge, gauge(a, 0), DELTA_T)
                               for a in SCALES], float)
    delta_at_peaks = np.asarray([[current_force(f, gauge(a, 0), DELTA_T)
                                  for a in SCALES] for f in peak_forces], float)
    clamp_tension = 3.0
    clamp_values = np.asarray([current_clamp(clamp_tension, p) for p in params])
    # A force-only intervention stays inside the observational equivalence
    # class, even if it is larger than the historical force range.
    force_step_values = np.asarray([current_force(force_challenge+50, p) for p in params])

    result = {
        "data_use": "34 Duke responding Piezo1/Piezo2 cells supply peak-force support only; all currents are synthetic",
        "reference_parameters": BASE,
        "gauge_scale_range": [float(SCALES[0]), float(SCALES[-1])],
        "gauge_offset_range_mN_m": [float(OFFSETS[0]), float(OFFSETS[-1])],
        "same_force_current_max_abs_difference_pA": max_baseline_discrepancy,
        "at_force_200nN": {
            "baseline_current_pA": float(current_force(force_challenge, BASE)),
            "force_plus_50nN_prediction_range_pA": list(map(float, [np.min(force_step_values), np.max(force_step_values)])),
            "plus_0p3mN_m_tension_prediction_range_pA": list(map(float, [np.min(delta_values), np.max(delta_values)])),
        },
        "on_34_observed_peak_force_inputs": {
            "peak_force_median_nN": float(np.median(peak_forces)),
            "delta_tension_prediction_spread_median_pA": float(np.median(np.ptp(delta_at_peaks, axis=1))),
            "delta_tension_prediction_spread_q25_q75_pA": list(map(float,
                                                                      np.quantile(np.ptp(delta_at_peaks, axis=1), [.25, .75]))),
        },
        "absolute_tension_clamp_3mN_m_prediction_range_pA": list(map(float,
                                                                         [np.min(clamp_values), np.max(clamp_values)])),
        "scope": "Exact non-identifiability is algebraic; magnitudes depend on chosen synthetic parameters and intervention. This is not a fitted biological effect.",
    }
    (HERE / "digital_twin_transportability_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.6), constrained_layout=True)
    for a in (.5, 1., 2.):
        p = gauge(a, 0)
        axes[0].plot(force_grid, current_force(force_grid, p), label=f"a={a:g}")
        axes[1].plot(force_grid, current_force(force_grid, p, DELTA_T), label=f"a={a:g}")
    axes[0].set(title="Observed F-to-I relation", xlabel="AFM force (nN)", ylabel="Synthetic current (pA)")
    axes[1].set(title="With +0.3 mN/m local tension", xlabel="AFM force (nN)",
                ylabel="Counterfactual current (pA)")
    for a in (.5, 1., 2.):
        for b in (-.5, .5):
            axes[2].plot(np.linspace(1, 5, 250),
                         current_clamp(np.linspace(1, 5, 250), gauge(a, b)),
                         label=f"a={a:g}, b={b:g}")
    axes[2].axvline(clamp_tension, color="black", ls="--", lw=.8)
    axes[2].set(title="Under absolute tension clamp", xlabel="Local tension (mN/m)",
                ylabel="Counterfactual current (pA)")
    for ax in axes:
        ax.legend(fontsize=6)
    fig.savefig(HERE / "digital_twin_transportability_audit.png", dpi=180)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
