"""Fit fixed direct baselines and target-blind three-parameter MCT twins."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.optimize import minimize
from scipy.stats import qmc
from sklearn.kernel_ridge import KernelRidge


ROOT = Path(__file__).resolve().parent / "mct_feedback_audit" / "reverse_robust3"
RANGES = np.array([[.20, 1.30], [.90, 1.10], [.85, 1.15]])
FIELDS = ("APD_above_minus67_mV_ms", "Ca_peak_uM")
LOADS = ("l02", "l06", "l10")
NOISE_SD = np.array([2., .01, 2., .01])


def standardize(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = a.mean(0)
    sd = np.maximum(a.std(0), 1e-9)
    return (a-mean)/sd, (b-mean)/sd, mean, sd


def direct_ridge(x: np.ndarray, y: np.ndarray, xtest: np.ndarray) -> np.ndarray:
    xx, tx, mx, sx = standardize(x, xtest)
    yy, _, my, sy = standardize(y, y)
    w = np.linalg.solve(xx.T @ xx + np.eye(xx.shape[1]), xx.T @ yy)
    return my + sy * (tx @ w)


def direct_rbf(x: np.ndarray, y: np.ndarray, xtest: np.ndarray) -> np.ndarray:
    xx, tx, _, _ = standardize(x, xtest)
    yy, _, my, sy = standardize(y, y)
    model = RBFInterpolator(xx, yy, kernel="thin_plate_spline", smoothing=1e-6)
    return my + sy * model(tx)


def direct_augmented_kernel(x: np.ndarray, y: np.ndarray, xtest: np.ndarray,
                            seed: int = 20261001) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sd = NOISE_SD[-x.shape[1]:]
    augmented = np.concatenate([x] + [x + rng.normal(size=x.shape)*sd for _ in range(8)])
    targets = np.tile(y, (9, 1))
    xx, tx, _, _ = standardize(augmented, xtest)
    yy, _, my, sy = standardize(targets, targets)
    model = KernelRidge(alpha=.1, kernel="rbf", gamma=1/(2*x.shape[1]))
    model.fit(xx, yy)
    return my + sy * model.predict(tx)


def invert(theta_train: np.ndarray, feature_train: np.ndarray,
           observed: np.ndarray, probe: np.ndarray) -> tuple[np.ndarray, float]:
    scaled_theta = (theta_train-RANGES[:, 0])/(RANGES[:, 1]-RANGES[:, 0])
    mean = feature_train.mean(0)
    sd = np.maximum(feature_train.std(0), 1e-9)
    emulator = RBFInterpolator(scaled_theta, (feature_train-mean)/sd,
                               kernel="thin_plate_spline", smoothing=1e-6)
    target = (observed-mean)/sd

    def objective(u: np.ndarray) -> float:
        residual = emulator(np.asarray(u)[None, :])[0] - target
        return float(residual @ residual)

    vals = emulator(probe)-target
    initial = probe[np.argsort(np.sum(vals*vals, axis=1))[:3]]
    results = [minimize(objective, x, method="L-BFGS-B",
                        bounds=((0, 1), (0, 1), (0, 1)),
                        options={"maxiter": 150}) for x in initial]
    best = min(results, key=lambda z: z.fun)
    return RANGES[:, 0] + best.x*(RANGES[:, 1]-RANGES[:, 0]), float(best.fun)


def main() -> None:
    cohort = json.loads((ROOT / "cohort_score.json").read_text(encoding="utf-8"))
    train = [r for r in cohort["rows"] if r["split"] == "train"]
    test = [r for r in cohort["rows"] if r["split"] == "test"]
    if len(train) != 48 or len(test) != 16:
        raise ValueError("Design split changed")
    theta_train = np.array([[r[k] for k in ("mct_gain", "pCa0_scale", "serca_scale")]
                            for r in train])
    theta_test = np.array([[r[k] for k in ("mct_gain", "pCa0_scale", "serca_scale")]
                           for r in test])

    def extract(rows: list[dict], loads: tuple[str, ...]) -> np.ndarray:
        return np.array([[r[load][field] for load in loads for field in FIELDS] for r in rows])

    xtrain = extract(train, LOADS[:2])
    xtest = extract(test, LOADS[:2])
    ytrain = extract(train, LOADS[2:])
    ytest = extract(test, LOADS[2:])
    predictions = {
        "low_ridge_clean": direct_ridge(xtrain[:, :2], ytrain, xtest[:, :2]),
        "lowmid_ridge_clean": direct_ridge(xtrain, ytrain, xtest),
        "low_RBF_clean": direct_rbf(xtrain[:, :2], ytrain, xtest[:, :2]),
        "lowmid_RBF_clean": direct_rbf(xtrain, ytrain, xtest),
        "linear_extrap_clean": 2*xtest[:, 2:]-xtest[:, :2],
    }
    rng = np.random.default_rng(20261002)
    noises = rng.normal(size=(3, len(test), 4))*NOISE_SD
    for rep, noise in enumerate(noises):
        noisy = xtest+noise
        predictions[f"lowmid_ridge_aug_noise{rep}"] = direct_ridge(
            np.concatenate([xtrain] + [xtrain + np.random.default_rng(20261001+j).normal(
                size=xtrain.shape)*NOISE_SD for j in range(8)]),
            np.tile(ytrain, (9, 1)), noisy)
        predictions[f"lowmid_KRR_aug_noise{rep}"] = direct_augmented_kernel(xtrain, ytrain, noisy)
        predictions[f"linear_extrap_noise{rep}"] = 2*noisy[:, 2:]-noisy[:, :2]
    (ROOT / "direct_predictions.json").write_text(json.dumps({
        "test_ids": [r["id"] for r in test], "truth": ytest.tolist(),
        "predictions": {k: v.tolist() for k, v in predictions.items()},
        "noise_draws": noises.tolist(),
        "noise_SD": NOISE_SD.tolist(),
    }, indent=2), encoding="utf-8")

    probe = qmc.Sobol(d=3, scramble=False).random_base2(8)
    fits = []
    for i, row in enumerate(test):
        for mode, reps in (("clean_low", (0,)), ("clean_lowmid", (0,)),
                           ("noisy_lowmid", (0, 1, 2))):
            for rep in reps:
                selected = slice(None, 2) if mode == "clean_low" else slice(None, 4)
                observed = xtest[i, selected].copy()
                if mode == "noisy_lowmid":
                    observed += noises[rep, i]
                inferred, loss = invert(theta_train, xtrain[:, selected], observed, probe)
                fits.append({"id": row["id"], "mode": mode, "rep": rep,
                             "mct_gain": float(inferred[0]),
                             "pCa0_scale": float(inferred[1]),
                             "serca_scale": float(inferred[2]),
                             "fit_loss_standardized_sse": loss,
                             "true_mct_gain": float(theta_test[i, 0]),
                             "true_pCa0_scale": float(theta_test[i, 1]),
                             "true_serca_scale": float(theta_test[i, 2])})
    with (ROOT / "inferred_parameters.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=("id", "mode", "rep", "mct_gain",
                                               "pCa0_scale", "serca_scale"))
        writer.writeheader()
        writer.writerows({k: row[k] for k in writer.fieldnames} for row in fits)
    (ROOT / "inverse_diagnostics.json").write_text(json.dumps(fits, indent=2), encoding="utf-8")
    for mode in ("clean_low", "clean_lowmid", "noisy_lowmid"):
        chosen = [r for r in fits if r["mode"] == mode]
        error = np.array([[r[f"{k}"]-r[f"true_{k}"] for k in
                           ("mct_gain", "pCa0_scale", "serca_scale")] for r in chosen])
        print(mode, "parameter MAE", np.mean(abs(error), axis=0))
    print(f"Wrote {len(fits)} inferred parameter vectors")


if __name__ == "__main__":
    main()
