"""One prespecified scalar-output forward-surrogate rescue for Tdur/Trest.

Models are chosen solely by 5-fold CV within forward-training parameter rows.
The 186 calibration and 187 test identities are excluded from every fit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist, pdist
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from strocchi_forward_gpe_audit import load_for_split, records
from strocchi_multiload_benchmark import LOADS, paired_data


OUT = Path(__file__).with_suffix(".json")
TARGETS = [(0, 5)] + [(i, j) for i in (1, 2) for j in range(6 if i == 1 else 7)]
CANDIDATES = ["ridge", "xgb", "mlp2", "gp_rbf_ard", "gp_matern_ard"]


def y_forward(y, transform):
    return np.log(y) if transform == "log" else y


def y_inverse(y, transform):
    return np.exp(np.clip(y, -20, 20)) if transform == "log" else y


class ScalarGP:
    def __init__(self, x, y, family):
        self.xmean, self.xstd = x.mean(0), x.std(0)
        self.xstd = np.where(self.xstd > 0, self.xstd, 1)
        self.ymean, self.ystd = y.mean(), y.std()
        self.ystd = self.ystd if self.ystd > 0 else 1
        # Output-specific train-only relevance produces anisotropic scales.
        rho = np.array([spearmanr(x[:, j], y).statistic for j in range(x.shape[1])])
        rho = np.nan_to_num(np.abs(rho), nan=0.0)
        relevance = 0.15 + rho
        relevance /= np.mean(relevance)
        self.weight = relevance
        self.x = ((x - self.xmean) / self.xstd) * relevance
        self.ell = np.median(pdist(self.x)) * 0.65
        self.family = family
        d = cdist(self.x, self.x) / self.ell
        k = self.kernel(d)
        k.flat[::len(k) + 1] += 0.02
        self.chol = cho_factor(k, lower=True, check_finite=False)
        self.alpha = cho_solve(self.chol, (y - self.ymean) / self.ystd, check_finite=False)

    def kernel(self, d):
        if self.family == "gp_rbf_ard":
            return np.exp(-0.5 * d * d)
        t = np.sqrt(5) * d
        return (1 + t + t * t / 3) * np.exp(-t)

    def predict(self, x):
        z = ((x - self.xmean) / self.xstd) * self.weight
        k = self.kernel(cdist(z, self.x) / self.ell)
        return (k @ self.alpha) * self.ystd + self.ymean


def model_predict(kind, x_train, y_train, x_test, transform):
    yt = y_forward(y_train, transform)
    if kind.startswith("gp_"):
        transformed_pred = ScalarGP(x_train, yt, kind).predict(x_test)
    else:
        scaler = StandardScaler().fit(x_train)
        train_z, test_z = scaler.transform(x_train), scaler.transform(x_test)
        ymean, ystd = yt.mean(), yt.std()
        ystd = ystd if ystd > 0 else 1
        target_z = (yt - ymean) / ystd
        if kind == "ridge":
            z = np.column_stack([np.ones(len(train_z)), train_z])
            penalty = np.eye(z.shape[1]) * 10
            penalty[0, 0] = 0
            coef = np.linalg.solve(z.T @ z + penalty, z.T @ target_z)
            pred_z = np.column_stack([np.ones(len(test_z)), test_z]) @ coef
        elif kind == "xgb":
            model = XGBRegressor(n_estimators=350, max_depth=3, learning_rate=0.045,
                                 subsample=0.9, colsample_bytree=0.9,
                                 reg_lambda=5, min_child_weight=4, n_jobs=2,
                                 objective="reg:squarederror", random_state=20260928)
            model.fit(train_z, target_z)
            pred_z = model.predict(test_z)
        elif kind == "mlp2":
            predictions = []
            for seed in (11, 29):
                model = MLPRegressor(hidden_layer_sizes=(64, 32), activation="tanh",
                                     alpha=0.01, learning_rate_init=0.001,
                                     max_iter=300, early_stopping=True,
                                     n_iter_no_change=20, random_state=seed)
                model.fit(train_z, target_z)
                predictions.append(model.predict(test_z))
            pred_z = np.mean(predictions, axis=0)
        else:
            raise ValueError(kind)
        transformed_pred = pred_z * ystd + ymean
    return y_inverse(transformed_pred, transform)


def scores(y, pred, standard_deviation):
    err = y - pred
    return {
        "mae_standardized": float(np.abs(err).mean() / standard_deviation),
        "r2": float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)),
        "mae_original_units": float(np.abs(err).mean()),
    }


def main():
    _, names, theta, _ = paired_data()
    dataset = records()
    rng = np.random.default_rng(20260928)
    perm = rng.permutation(len(theta))
    train_ids, cal_ids, test_ids = perm[:558], perm[558:744], perm[744:]
    excluded = {tuple(row) for row in theta[np.r_[cal_ids, test_ids]]}
    result = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {
        "protocol": "5-fold train-only model selection for each scalar output; fixed seed 20260928; 186 calibration and 187 test paired identities excluded from all forward model fits",
        "model_definitions": {
            "gp": "independent scalar GP with output-specific rank-correlation relevance-scaled anisotropic kernel; fixed global length and nugget, not marginal-likelihood-optimized ARD",
            "xgb": "fixed shallow XGBRegressor hyperparameters",
            "mlp2": "two-seed small tanh MLP ensemble",
            "ridge": "standardized ridge",
        },
        "targets": {},
    }
    for load_idx, output_idx in TARGETS:
        load = LOADS[load_idx]
        x, all_y = load_for_split(dataset, excluded, load_idx)
        y = all_y[:, output_idx]
        assert not any(tuple(row) in excluded for row in x)
        key = f"L{load_idx+1}:{output_idx}:{names[load_idx][output_idx]}"
        if key in result["targets"]:
            continue
        folds = list(KFold(n_splits=5, shuffle=True, random_state=20260928).split(x))
        variants = [(kind, trans) for kind in CANDIDATES
                    for trans in (("identity", "log") if output_idx == 5 else ("identity",))]
        cv_results = {}
        for kind, transform in variants:
            per_fold = []
            for ids_train, ids_val in folds:
                pred = model_predict(kind, x[ids_train], y[ids_train], x[ids_val], transform)
                per_fold.append(scores(y[ids_val], pred, y[ids_train].std()))
            cv_results[f"{kind}:{transform}"] = {
                "mae_standardized": float(np.mean([s["mae_standardized"] for s in per_fold])),
                "r2_mean": float(np.mean([s["r2"] for s in per_fold])),
            }
        best = min(cv_results, key=lambda z: cv_results[z]["mae_standardized"])
        kind, transform = best.split(":")
        pred_cal = model_predict(kind, x, y, theta[cal_ids], transform)
        pred_test = model_predict(kind, x, y, theta[test_ids], transform)
        y_cal = np.array([dataset[load_idx][tuple(row)][output_idx] for row in theta[cal_ids]])
        y_test = np.array([dataset[load_idx][tuple(row)][output_idx] for row in theta[test_ids]])
        result["targets"][key] = {
            "n_train": len(x),
            "selected": best,
            "cv": cv_results,
            "calibration": scores(y_cal, pred_cal, y.std()),
            "test": scores(y_test, pred_test, y.std()),
            "distribution": {
                "train_min": float(y.min()), "train_median": float(np.median(y)),
                "train_q01": float(np.quantile(y, 0.01)),
                "train_q99": float(np.quantile(y, 0.99)),
                "train_max": float(y.max()),
            },
        }
        print(key, best, "cv", round(cv_results[best]["mae_standardized"], 3),
              "cal_r2", round(result["targets"][key]["calibration"]["r2"], 3),
              "test_r2", round(result["targets"][key]["test"]["r2"], 3), flush=True)
        OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
