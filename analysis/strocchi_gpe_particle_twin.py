"""Leakage-safe GP-forward, particle-calibrated held-out-load pilot.

This is a GP *emulator* closed loop, not execution of the original simulator.
The prior is an empirical Gaussian copula over training paired parameter rows.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm, rankdata

from strocchi_forward_gpe_audit import GP, load_for_split, records
from strocchi_multiload_benchmark import LOADS, paired_data, fit_one


SEED = int(os.environ.get("STROCCHI_TWIN_SEED", "20260928"))
N_PARTICLES = int(os.environ.get("STROCCHI_TWIN_PARTICLES", "16384"))
OUT = Path(__file__).with_suffix(".json") if SEED == 20260928 else (
    Path(__file__).with_name(f"strocchi_gpe_particle_twin_{SEED}.json"))


def sample_copula(theta_train, n, rng):
    n_train, n_params = theta_train.shape
    ranks = np.column_stack([rankdata(theta_train[:, j], method="average") for j in range(n_params)])
    gaussian = norm.ppf((ranks - 0.5) / n_train)
    corr = np.corrcoef(gaussian, rowvar=False)
    corr = 0.9 * corr + 0.1 * np.eye(n_params)
    z = rng.multivariate_normal(np.zeros(n_params), corr, size=n)
    u = np.clip(norm.cdf(z), 1e-6, 1 - 1e-6)
    particles = np.column_stack([
        np.quantile(theta_train[:, j], u[:, j], method="linear")
        for j in range(n_params)
    ])
    return particles


def likelihood_precision(gp, true_cal, pred_cal, var_cal):
    resid = (true_cal - pred_cal) / gp.ystd
    cov = np.cov(resid, rowvar=False)
    cov = 0.75 * cov + 0.25 * np.diag(np.diag(cov))
    # GP uncertainty and empirical residual disagreement are both retained;
    # this conservative addition is audited via held-out coverage/ESS.
    cov += np.diag(np.mean(var_cal / gp.ystd ** 2, axis=0))
    cov += 1e-4 * np.eye(cov.shape[0])
    return np.linalg.inv(cov)


def posterior_predictions(y_observed, seen, target, particle_yz, gp_models, precisions,
                          theta_particles, theta_true, y_true, particle_yvar_mean):
    n = len(y_true)
    target_gp = gp_models[target]
    output_mean = np.zeros((n, y_true.shape[1]))
    output_var = np.zeros_like(output_mean)
    theta_mean = np.zeros((n, theta_particles.shape[1]))
    ess = np.zeros(n)
    for row in range(n):
        logw = np.zeros(len(theta_particles))
        for i in seen:
            obs = (y_observed[i][row] - gp_models[i].ymean) / gp_models[i].ystd
            diff = particle_yz[i] - obs
            logw -= 0.5 * np.einsum("nd,df,nf->n", diff, precisions[i], diff, optimize=True)
        w = np.exp(logw - logsumexp(logw))
        ess[row] = 1 / np.sum(w ** 2)
        target_particle = particle_yz[target]
        mean_z = w @ target_particle
        var_z = w @ (target_particle - mean_z) ** 2
        var_z += particle_yvar_mean[target]
        output_mean[row] = mean_z * target_gp.ystd + target_gp.ymean
        output_var[row] = var_z * target_gp.ystd ** 2
        theta_mean[row] = w @ theta_particles
    return output_mean, output_var, theta_mean, ess


def main():
    _, output_names, theta, blocks = paired_data()
    dataset = records()
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(theta))
    train, cal, test = perm[:558], perm[558:744], perm[744:]
    cal_noise, cal_interval = cal[:93], cal[93:]
    excluded = {tuple(row) for row in theta[np.r_[cal, test]]}
    gp_models, precisions = [], []
    particle_yz, particle_yvar_mean = [], []
    particles = sample_copula(theta[train], N_PARTICLES, rng)
    for i in range(3):
        x_train, y_train = load_for_split(dataset, excluded, i)
        assert not any(tuple(row) in excluded for row in x_train)
        gp = GP(x_train, y_train)
        pred_noise, var_noise = gp.predict(theta[cal_noise], variance=True)
        precision = likelihood_precision(gp, blocks[i][cal_noise], pred_noise, var_noise)
        pred_particles = gp.predict(particles, batch=512)
        gp_models.append(gp)
        precisions.append(precision)
        particle_yz.append((pred_particles - gp.ymean) / gp.ystd)
        particle_yvar_mean.append(np.mean(var_noise / gp.ystd ** 2, axis=0))
    output = {
        "n_particles": N_PARTICLES,
        "split": {"train": len(train), "likelihood_cal": len(cal_noise),
                  "interval_cal": len(cal_interval), "test": len(test)},
        "gp_train_counts": [len(load_for_split(dataset, excluded, i)[0]) for i in range(3)],
        "prior": "Gaussian copula fitted only to 558 paired training parameter vectors; rank marginals with correlation shrinkage 0.1",
        "scoring": "all held-out output errors standardized using the 558 paired training parameter vectors; likelihood and conformal calibration use disjoint halves",
        "tasks": {},
    }
    for name, seen, target in [
        ("L1_to_L3", (0,), 2),
        ("L1_L2_to_L3", (0, 1), 2),
        ("L1_to_L2", (0,), 1),
        ("L1_L3_to_L2", (0, 2), 1),
    ]:
        obs_cal = {i: blocks[i][cal_interval] for i in seen}
        obs_test = {i: blocks[i][test] for i in seen}
        pred_cal, var_cal, _, ess_cal = posterior_predictions(
            obs_cal, seen, target, particle_yz, gp_models, precisions,
            particles, theta[cal_interval], blocks[target][cal_interval], particle_yvar_mean)
        pred_test, var_test, theta_hat, ess_test = posterior_predictions(
            obs_test, seen, target, particle_yz, gp_models, precisions,
            particles, theta[test], blocks[target][test], particle_yvar_mean)
        scale = blocks[target][train].std(0)
        scale = np.where(scale > 0, scale, 1)
        err_cal = np.abs((blocks[target][cal_interval] - pred_cal) / scale)
        q_idx = min(int(np.ceil((len(cal_interval) + 1) * 0.9)) - 1, len(cal_interval) - 1)
        q = np.sort(err_cal, axis=0)[q_idx]
        err_test = np.abs((blocks[target][test] - pred_test) / scale)
        x_direct = np.column_stack([blocks[i] for i in seen])
        direct = fit_one(x_direct, blocks[target], train, cal, test,
                         np.random.default_rng(SEED + 100 + target + 10 * len(seen)))
        target_param_std = theta[train].std(0)
        target_param_std = np.where(target_param_std > 0, target_param_std, 1)
        output["tasks"][name] = {
            "twin_mae_standardized": float(err_test.mean()),
            "direct_ridge_mae_standardized": float(direct["mae_each"].mean()),
            "twin_conformal_90pct_coverage": float((err_test <= q).mean()),
            "twin_conformal_90pct_width_standardized": float((2 * q).mean()),
            "twin_ess_median": float(np.median(ess_test)),
            "twin_ess_min": float(np.min(ess_test)),
            "theta_posterior_mean_mae_standardized": float(np.mean(np.abs((theta[test] - theta_hat) / target_param_std))),
            "particle_predictive_variance_z_mean": float(np.mean(var_test / scale ** 2)),
            "per_target_feature_mae_standardized": {
                f"{j}:{label}": {
                    "twin": float(err_test[:, j].mean()),
                    "direct_ridge": float(direct["mae_each"][j]),
                }
                for j, label in enumerate(output_names[target])
            },
        }
    OUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
