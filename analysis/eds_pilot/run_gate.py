"""Reproducible first EDS identifiability and sparse-frequency gate."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from model import cm_factor, jacobian, response

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(20261011)
FREQ = np.geomspace(1e5, 15e6, 21)
NAMES = ["epsilon_mem", "sigma_cyto", "mu"]
NOMINAL = np.log([4.7, 0.65, 2.5])
LO = np.log([0.1, 0.1, 0.1])
HI = np.log([30.0, 3.0, 30.0])
NOISE = 0.0075  # absolute response units; deliberately explicit assumption


def svd_stats(indices, theta=NOMINAL):
    j = jacobian(FREQ[np.asarray(indices)], theta) / NOISE
    s = np.linalg.svd(j, compute_uv=False)
    return s, float(s[0] / s[-1]) if s[-1] > 1e-12 else float("inf")


def design(k):
    # Exhaustive nominal D-optimal subset; choose from training-model information only.
    best, best_score = None, -np.inf
    for idx in itertools.combinations(range(len(FREQ)), k):
        j = jacobian(FREQ[list(idx)], NOMINAL) / NOISE
        sign, score = np.linalg.slogdet(j.T @ j + 1e-10 * np.eye(3))
        if sign > 0 and score > best_score:
            best, best_score = idx, score
    return best


def fit(y, idx, initial):
    f = FREQ[list(idx)]
    result = least_squares(lambda p: (response(f, p) - y) / NOISE,
                           initial, bounds=(LO, HI), max_nfev=110)
    return result.x


def main():
    full = tuple(range(len(FREQ)))
    designs = {"full21": full}
    for k in [3, 4, 5]:
        designs[f"log{k}"] = tuple(np.round(np.linspace(0, len(FREQ)-1, k)).astype(int))
        designs[f"dopt{k}"] = design(k)

    population = np.column_stack([
        RNG.uniform(2.0, 8.0, 80),
        RNG.uniform(0.3, 1.1, 80),
        RNG.uniform(1.5, 4.0, 80),
    ])
    # Every design receives the same cells and the same frequency-wise noise.
    noise = RNG.normal(0, NOISE, (len(population), len(FREQ)))
    records = {}
    error_arrays = {}
    for label, idx in designs.items():
        errors, shifted_errors, clipped = [], [], 0
        for n, truth in enumerate(population):
            y_all = response(FREQ, np.log(truth)) + noise[n]
            y = y_all[list(idx)]
            est = np.exp(fit(y, idx, NOMINAL))
            errors.append(np.abs(np.log(est / truth)))
            # Deliberately mismatched truth: membrane conductance 10x larger.
            shifted = np.real(cm_factor(FREQ, truth[0], truth[1], sigma_mem=1e-5)) / truth[2]
            shifted_est = np.exp(fit((shifted + noise[n])[list(idx)], idx, NOMINAL))
            shifted_errors.append(np.abs(np.log(shifted_est / truth)))
            clipped += int(np.any(np.isclose(np.log(est), LO, atol=1e-3) |
                                   np.isclose(np.log(est), HI, atol=1e-3)))
        errors = np.asarray(errors)
        shifted_errors = np.asarray(shifted_errors)
        error_arrays[label + "_nominal"] = errors
        error_arrays[label + "_membrane_shift"] = shifted_errors
        s, condition = svd_stats(idx)
        records[label] = {
            "indices": [int(i) for i in idx],
            "frequencies_mhz": np.round(FREQ[list(idx)] / 1e6, 5).tolist(),
            "singular_values": s.tolist(),
            "jacobian_condition": condition,
            "median_absolute_log_error_by_parameter": dict(zip(NAMES, np.median(errors, axis=0).tolist())),
            "median_mean_absolute_log_error": float(np.median(errors.mean(axis=1))),
            "p90_mean_absolute_log_error": float(np.quantile(errors.mean(axis=1), 0.9)),
            "mismatched_membrane_conductance_median_error": float(np.median(shifted_errors.mean(axis=1))),
            "mismatched_membrane_conductance_p90_error": float(np.quantile(shifted_errors.mean(axis=1), 0.9)),
            "bound_hit_cells": clipped,
        }

    j = jacobian(FREQ, NOMINAL)
    # Perfectly calibrated field scale vs one extra unknown multiplicative scale.
    gain_column = response(FREQ, NOMINAL)
    j_aug = np.column_stack([j, gain_column])
    singular_aug = np.linalg.svd(j_aug, compute_uv=False)
    # Exact identity: log(gain) and log(mu) have opposite derivative columns.
    scale_symmetry_error = float(np.max(np.abs(j_aug[:, 2] + j_aug[:, 3])))
    payload = {
        "scope": "spherical single-shell + small-deformation reduced model, not original EDS reproduction",
        "source": "Du et al. Acta Biomater 2025 doi:10.1016/j.actbio.2024.12.012 Eq4, Eq11",
        "assumptions": {"epsilon_medium": 78, "noise_sd_dimensionless": NOISE,
                        "cell_radius_um": 3.9, "geometry": "sphere; constant field scale"},
        "nominal": dict(zip(NAMES, np.exp(NOMINAL).tolist())),
        "full_spectrum_singular_values": svd_stats(full)[0].tolist(),
        "unknown_gain_augmented_singular_values": singular_aug.tolist(),
        "mu_gain_jacobian_symmetry_max_abs": scale_symmetry_error,
        "designs": records,
    }
    (OUT / "gate.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(OUT / "per_cell_errors.npz", truth=population, noise=noise, **error_arrays)

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True)
    for col, name in zip(j.T, NAMES):
        axs[0].semilogx(FREQ/1e6, col, label=name)
    axs[0].axhline(0, color="black", lw=.6)
    axs[0].set(xlabel="Frequency (MHz)", ylabel="Sensitivity to log parameter",
               title="Reduced-model sensitivity")
    axs[0].legend(fontsize=8)
    labels = list(records)
    vals = [records[x]["median_mean_absolute_log_error"] for x in labels]
    shifted = [records[x]["mismatched_membrane_conductance_median_error"] for x in labels]
    axs[1].scatter(np.arange(len(labels))-.1, vals, color="#116466", label="Nominal")
    axs[1].scatter(np.arange(len(labels))+.1, shifted, color="#c85a30", label="10× membrane conductivity")
    axs[1].set_xticks(range(len(labels)), labels, rotation=50, ha="right")
    axs[1].set(ylabel="Median mean |log estimate / truth|", title="Synthetic sparse-frequency gate")
    axs[1].legend(fontsize=8)
    fig.savefig(OUT / "gate.png", dpi=180)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
