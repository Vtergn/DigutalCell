"""Approximate Gaussian conditional posterior on train-screened mechanical params.

This evaluates whether posterior volume shrinks *and* test coverage is credible.
It is a joint-normal approximation to the simulator sampling distribution, not
an exact mechanistic posterior or a neural flow.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from strocchi_multiload_benchmark import paired_data, screen_mechanical


OUT = Path(__file__).with_suffix(".json")
SETS = {"L1": (0,), "L1_L2": (0, 1), "L1_L3": (0, 2), "L1_L2_L3": (0, 1, 2)}
SHRINKS = (0.01, 0.05, 0.1, 0.2, 0.4)
Z90 = 1.6448536269514722


def fit_gaussian(x, theta, ids, shrink):
    data = np.column_stack([theta, x])
    mu = data[ids].mean(0)
    sd = data[ids].std(0)
    sd = np.where(sd > 0, sd, 1)
    z = (data[ids] - mu) / sd
    cov = np.cov(z, rowvar=False)
    cov = (1 - shrink) * cov + shrink * np.eye(cov.shape[0])
    return mu, sd, cov


def condition(model, x, n_theta):
    mu, sd, cov = model
    mt, mx = mu[:n_theta], mu[n_theta:]
    st, sx = sd[:n_theta], sd[n_theta:]
    ct, cx = cov[:n_theta, :n_theta], cov[n_theta:, n_theta:]
    ctx = cov[:n_theta, n_theta:]
    xz = (x - mx) / sx
    gain = np.linalg.solve(cx, ctx.T).T
    predicted_z = xz @ gain.T
    conditional_cov = ct - gain @ ctx.T
    conditional_cov = (conditional_cov + conditional_cov.T) / 2
    return mt + predicted_z * st, conditional_cov, st


def metrics(model, x, theta, ids):
    mean, cov, scale = condition(model, x[ids], theta.shape[1])
    err = (theta[ids] - mean) / scale
    variances = np.clip(np.diag(cov), 1e-12, None)
    coverage = (np.abs(err) <= Z90 * np.sqrt(variances)).mean(0)
    mahalanobis_sq = np.sum(err @ np.linalg.inv(cov) * err, axis=1)
    joint_coverage = float(np.mean(mahalanobis_sq <= chi2.ppf(0.9, theta.shape[1])))
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError("Conditional covariance not positive definite")
    return {
        "mae_standardized": float(np.abs(err).mean()),
        "coverage_90pct_mean": float(coverage.mean()),
        "joint_ellipsoid_coverage_90pct": joint_coverage,
        "coverage_90pct_each": list(map(float, coverage)),
        "width_90pct_standardized_mean": float((2 * Z90 * np.sqrt(variances)).mean()),
        "logdet_conditional_cov": float(logdet),
        "nll_per_cell": float(0.5 * (theta.shape[1] * np.log(2 * np.pi) + logdet +
                                     np.mean(mahalanobis_sq))),
    }


def main():
    names, _, all_theta, blocks = paired_data()
    rows = []
    for seed in range(10):
        rng = np.random.default_rng(20260928 + seed)
        perm = rng.permutation(len(all_theta))
        train, test = perm[:744], perm[744:]
        inner = rng.permutation(train)
        fit_ids, val_ids = inner[:595], inner[595:]
        selected = screen_mechanical(all_theta, blocks, fit_ids, names)
        theta = all_theta[:, selected]
        scores = {}
        for label, load_ids in SETS.items():
            x = np.column_stack([blocks[i] for i in load_ids])
            choices = []
            for shrink in SHRINKS:
                model = fit_gaussian(x, theta, fit_ids, shrink)
                choices.append((metrics(model, x, theta, val_ids)["nll_per_cell"], shrink))
            shrink = min(choices)[1]
            model = fit_gaussian(x, theta, train, shrink)
            scores[label] = {"shrink": shrink, **metrics(model, x, theta, test)}
        rows.append({"seed": seed, "selected": [names[i] for i in selected], "scores": scores})
    summary = {}
    for label in SETS:
        summary[label] = {
            field: float(np.mean([r["scores"][label][field] for r in rows]))
            for field in ("mae_standardized", "coverage_90pct_mean", "joint_ellipsoid_coverage_90pct", "width_90pct_standardized_mean", "logdet_conditional_cov", "nll_per_cell")
        }
    result = {
        "model": "joint-normal conditional density of 10 train-screened mechanical parameters and observed load features; covariance diagonal shrink tuned on train-inner validation",
        "n_parameter_sets": len(all_theta),
        "n_train_test": [744, 187],
        "summary": summary,
        "splits": rows,
        "cautions": [
            "Normality is an approximation; 90% marginal posterior coverage is empirically tested, not guaranteed.",
            "The parameter screen uses only training samples but selects a related 10-parameter subset, not the full 27-dimensional posterior.",
            "Log determinant measures posterior volume only inside this approximate Gaussian model.",
        ],
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
