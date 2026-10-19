"""Test an externally fixed Stepwise force/pressure exchange rate on calcium thresholds.

The exchange rate is derived from Fig. 2 FLIM cells and never tuned on Fig. 3
calcium threshold cells. Evaluation holds out one experimental repeat at a time.
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from external_physics_bridge import PRESSURE_GRID, eth_rows, fit, score


HERE = Path(__file__).resolve().parent


def features(rows, ratio):
    pressure = np.broadcast_to(PRESSURE_GRID[None, :], (len(rows), len(PRESSURE_GRID)))
    force = np.broadcast_to(np.array([r["force_nN"] for r in rows])[:, None], pressure.shape)
    threshold = np.array([r["threshold_mbar"] for r in rows])[:, None]
    target = (pressure >= threshold).astype(float)
    matrix = np.stack([np.ones_like(pressure), (pressure + ratio * force) / 100], axis=-1)
    return matrix.reshape(-1, 2), target.ravel()


def fit_fixed(rows, ratio):
    matrix, target = features(rows, ratio)
    penalty = .001

    def objective(coef):
        logit = matrix @ coef
        prob = expit(logit)
        value = np.mean(np.logaddexp(0, logit) - target * logit) + penalty * coef[1] ** 2
        gradient = matrix.T @ (prob - target) / len(target)
        gradient[1] += 2 * penalty * coef[1]
        return float(value), gradient

    result = minimize(objective, np.zeros(2), method="L-BFGS-B", jac=True)
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def score_fixed(rows, ratio, coef):
    matrix, target = features(rows, ratio)
    prob = np.clip(expit(matrix @ coef), 1e-9, 1 - 1e-9)
    return {"brier": float(np.mean((prob - target) ** 2)),
            "logloss": float(np.mean(-target * np.log(prob) - (1-target) * np.log1p(-prob)))}


def main():
    stepwise = json.loads((HERE / "eth_stepwise_results.json").read_text(encoding="utf-8"))
    force = stepwise["indentation"]["mean_slope_tension_mN_per_m_per_dose"]
    pressure = stepwise["aspiration"]["mean_slope_tension_mN_per_m_per_dose"]
    ratio = force / pressure
    report = {"fixed_mbar_per_nN": ratio, "conditions": {}}
    all_rows = eth_rows()
    for condition in ("Control", "MargAc", "CytoD"):
        group = [r for r in all_rows if r["condition"] == condition]
        heldout = []
        for repeat in (1, 2, 3):
            train = [r for r in group if r["repeat"] != repeat]
            test = [r for r in group if r["repeat"] == repeat]
            fixed = fit_fixed(train, ratio)
            baseline = fit(train, "pressure_only")
            free = fit(train, "additive")
            heldout.append({"repeat": repeat, "test_cells": len(test),
                            "pressure_only": score(test, "pressure_only", baseline),
                            "fixed_stepwise_ratio": score_fixed(test, ratio, fixed),
                            "free_additive": score(test, "additive", free)})
        total = sum(x["test_cells"] for x in heldout)
        weighted = {key: sum(x["test_cells"] * x[key]["brier"] for x in heldout) / total
                    for key in ("pressure_only", "fixed_stepwise_ratio", "free_additive")}
        report["conditions"][condition] = {"heldout": heldout, "weighted_brier": weighted}
        print(condition, "n", total, "weighted Brier", {k: round(v, 4) for k, v in weighted.items()})
        for x in heldout:
            print(" repeat",x["repeat"],"n",x["test_cells"],
                  {k:round(x[k]["brier"],4) for k in weighted})
    (HERE / "eth_stepwise_threshold_bridge_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
