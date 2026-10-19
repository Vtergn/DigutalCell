"""Compare direct load-to-load prediction with a parameter-mediated surrogate.

The mediator uses an approximate Gaussian posterior over all 27 parameters and
a ridge *surrogate* from true parameters to target-load features. It does not
run ToR-ORd-Land for each posterior draw.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from strocchi_gaussian_posterior_pilot import fit_gaussian, condition
from strocchi_multiload_benchmark import paired_data, fit_one, standardized, ridge_fit, predictions


OUT = Path(__file__).with_suffix(".json")
CASES = {
    "L1_to_L3": ((0,), 2),
    "L1_L2_to_L3": ((0, 1), 2),
    "L1_to_L2": ((0,), 1),
    "L1_L3_to_L2": ((0, 2), 1),
}


def mediated(theta, x_seen, y_target, train, cal, test):
    # The same coarse shrinkage is fixed before test evaluation. A more exact
    # density model can later replace this component without changing splits.
    posterior = fit_gaussian(x_seen, theta, train, shrink=0.1)
    theta_mean, theta_cov_z, _ = condition(posterior, x_seen, theta.shape[1])
    xt, yt, ymu, ysd = standardized(theta, y_target, train)
    beta = ridge_fit(xt, yt, train, alpha=10.0)
    # Use the training parameter scale to apply the forward surrogate.
    tmu, tsd = theta[train].mean(0), theta[train].std(0)
    tsd = np.where(tsd > 0, tsd, 1)
    pred_z = predictions((theta_mean - tmu) / tsd, beta)
    true_z = (y_target - ymu) / ysd
    # Calibrated intervals absorb model error and omitted nonlinear effects.
    errors_cal = np.abs(true_z[cal] - pred_z[cal])
    q_idx = min(int(np.ceil((len(cal) + 1) * 0.9)) - 1, len(cal) - 1)
    q = np.sort(errors_cal, axis=0)[q_idx]
    error = np.abs(true_z[test] - pred_z[test])
    return {
        "mae": float(error.mean()),
        "coverage_90pct": float((error <= q).mean()),
        "interval_fullwidth": float((2 * q).mean()),
        "posterior_covariance_trace_z": float(np.trace(theta_cov_z)),
    }


def main():
    _, _, theta, blocks = paired_data()
    rows = []
    for seed in range(10):
        rng = np.random.default_rng(20260928 + seed)
        perm = rng.permutation(len(theta))
        train, cal, test = perm[:558], perm[558:744], perm[744:]
        scores = {}
        for case, (seen, target) in CASES.items():
            x = np.column_stack([blocks[i] for i in seen])
            y = blocks[target]
            direct = fit_one(x, y, train, cal, test, rng)
            via_theta = mediated(theta, x, y, train, cal, test)
            scores[case] = {
                "direct_ridge_mae": float(direct["mae_each"].mean()),
                "parameter_mediated_mae": via_theta["mae"],
                "parameter_mediated_coverage_90pct": via_theta["coverage_90pct"],
                "parameter_mediated_interval_fullwidth": via_theta["interval_fullwidth"],
                "posterior_covariance_trace_z": via_theta["posterior_covariance_trace_z"],
            }
        rows.append({"seed": seed, "scores": scores})
    summary = {}
    for case in CASES:
        summary[case] = {
            key: float(np.mean([r["scores"][case][key] for r in rows]))
            for key in rows[0]["scores"][case]
        }
        summary[case]["mediated_wins_splits"] = sum(
            r["scores"][case]["parameter_mediated_mae"] < r["scores"][case]["direct_ridge_mae"]
            for r in rows
        )
    result = {
        "design": "same 10 outer splits as direct benchmark; Gaussian conditional posterior over 27 parameters; ridge surrogate from true parameters to target-load features; calibrated prediction intervals",
        "summary": summary,
        "splits": rows,
        "limitations": [
            "The forward map is trained from paired CSV rather than the mechanistic ToR-ORd-Land simulator.",
            "The posterior is a joint-Gaussian approximation; the mapping ignores nonlinearity and may discard information.",
            "Mediated and direct models use different hyperparameter policies; these are architecture diagnostics, not final tuned model rankings.",
        ],
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
