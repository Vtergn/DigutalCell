"""Paired synthetic pilot for information-guided mechanical stimulus design.

The simulator is intentionally minimal. It has an effective force threshold,
force-to-current peak delay, recovery time and cell gain. Parameters and units
are illustrative, not estimates for Duke or PIEZO channels. The active designer
uses only past noisy observations and a population prior, never true parameters.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit


HERE = Path(__file__).resolve().parent
SEED = 20260928
N_CELLS = 50
N_PULSES = 6
SIGMA_AMPLITUDE = .035
SIGMA_TIMING_MS = .8
PRIOR_CENTER = np.array([200., 5., np.log(300.), np.log(1.)])
PRIOR_SD = np.array([60., 5., .8, .5])
LOWER = np.array([120., 0., np.log(40.), np.log(.4)])
UPPER = np.array([280., 15., np.log(1500.), np.log(2.)])
STIMULI = [(float(force), float(speed), float(interval))
           for force in (260, 300, 370, 440)
           for speed in (5, 10, 20)
           for interval in (50, 200, 800)]


def observations(theta: np.ndarray,
                 protocol: list[tuple[float, float, float]],
                 mismatch: bool = False) -> np.ndarray:
    """Returns [peak amplitude, peak time] for each pulse.

    Force ramps linearly at speed until the requested peak. The first interval
    is ignored. A pulse depletes a readiness state proportional to activation;
    readiness recovers exponentially between pulses.
    """
    f50, delay, log_tau, log_gain = theta
    tau, gain = np.exp(log_tau), np.exp(log_gain)
    readiness = 1.
    previous_post = 1.
    out = []
    for n, (fmax, speed, interval) in enumerate(protocol):
        if n:
            remaining = (np.exp(-interval/tau) if not mismatch else
                         .7*np.exp(-interval/tau)+.3*np.exp(-interval/(4*tau)))
            readiness = 1-(1-previous_post)*remaining
        activation = expit((fmax-f50)/(35. if mismatch else 25.))
        out.append((gain*readiness*activation, f50/speed+delay))
        previous_post = readiness*(1-.70*activation)
    return np.asarray(out)


def fit_posterior_mode(protocol: list[tuple[float, float, float]],
                       measured: np.ndarray) -> np.ndarray:
    def residual(theta: np.ndarray) -> np.ndarray:
        pred = observations(theta, protocol)
        data = ((pred-measured)/[SIGMA_AMPLITUDE, SIGMA_TIMING_MS]).ravel()
        return np.r_[data, (theta-PRIOR_CENTER)/PRIOR_SD]
    return least_squares(residual, PRIOR_CENTER.copy(), bounds=(LOWER, UPPER),
                         max_nfev=75, ftol=1e-6, xtol=1e-6).x


def jacobian_last(theta: np.ndarray,
                  protocol: list[tuple[float, float, float]]) -> np.ndarray:
    delta = np.array([.1, .01, .005, .005])
    columns = []
    for j in range(4):
        plus, minus = theta.copy(), theta.copy()
        plus[j] += delta[j]
        minus[j] -= delta[j]
        columns.append((observations(plus, protocol)[-1]-
                        observations(minus, protocol)[-1])/(2*delta[j]))
    return np.stack(columns, axis=1)/np.array([[SIGMA_AMPLITUDE],
                                               [SIGMA_TIMING_MS]])


def fisher(theta: np.ndarray,
           protocol: list[tuple[float, float, float]]) -> np.ndarray:
    precision = np.diag(1/PRIOR_SD**2)
    for n in range(len(protocol)):
        j = jacobian_last(theta, protocol[:n+1])
        precision = precision+j.T@j
    return precision


def optimal_next(theta_hat: np.ndarray,
                 history: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    candidates = STIMULI if history else [p for p in STIMULI if p[2] == 800]
    scored = []
    for candidate in candidates:
        sign, logdet = np.linalg.slogdet(fisher(theta_hat, history+[candidate]))
        scored.append(logdet if sign > 0 else -np.inf)
    return candidates[int(np.argmax(scored))]


def duke_like(n: int) -> tuple[float, float, float]:
    return float(260+25*n), 10., 5000.


def run_cell(theta_true: np.ndarray, method: str, seed: int,
             noise: np.ndarray, mismatch: bool) -> tuple[dict, list[dict]]:
    rng = np.random.default_rng(seed)
    protocol, measured, records = [], [], []
    for n in range(N_PULSES):
        if method == "duke_like":
            candidate = duke_like(n)
        elif method == "random":
            candidate = STIMULI[int(rng.integers(len(STIMULI)))]
        else:
            estimate = (fit_posterior_mode(protocol, np.asarray(measured))
                        if protocol else PRIOR_CENTER)
            candidate = optimal_next(estimate, protocol)
        protocol.append(candidate)
        truth = observations(theta_true, protocol, mismatch=mismatch)[-1]
        observed = truth+noise[n]
        measured.append(observed)
        records.append({"pulse": n+1, "fmax_nN": candidate[0],
                        "loading_speed_nN_per_ms": candidate[1],
                        "interval_ms": candidate[2]})
    theta_hat = fit_posterior_mode(protocol, np.asarray(measured))
    # Future challenge is measured after the six designed stimuli. Its interval
    # probes recovery; the challenge observation was never used to fit theta.
    challenge = protocol+[(310., 7., 150.)]
    true_future = observations(theta_true, challenge, mismatch=mismatch)[-1]
    pred_future = observations(theta_hat, challenge)[-1]
    outcome = {"f50_abs_error_nN": abs(theta_hat[0]-theta_true[0]),
               "delay_abs_error_ms": abs(theta_hat[1]-theta_true[1]),
               "tau_abs_log_error": abs(theta_hat[2]-theta_true[2]),
               "gain_abs_log_error": abs(theta_hat[3]-theta_true[3]),
               "challenge_peak_abs_error": abs(pred_future[0]-true_future[0]),
               "challenge_timing_abs_error_ms": abs(pred_future[1]-true_future[1]),
               "force_span_nN": max(p[0] for p in protocol)-min(p[0] for p in protocol),
               "speed_levels": len({p[1] for p in protocol}),
               "interval_levels": len({p[2] for p in protocol[1:]})}
    return outcome, records


def main() -> None:
    rng = np.random.default_rng(SEED)
    true = np.column_stack([rng.uniform(145, 245, N_CELLS),
                            rng.uniform(1, 10, N_CELLS),
                            np.log(rng.uniform(80, 800, N_CELLS)),
                            np.log(rng.uniform(.75, 1.4, N_CELLS))])
    outcomes, protocols = [], []
    for mismatch in (False, True):
        scenario = "misspecified" if mismatch else "in_model"
        for cell, theta in enumerate(true):
            common_noise = np.random.default_rng(SEED+cell*37).normal(
                0, [SIGMA_AMPLITUDE, SIGMA_TIMING_MS], size=(N_PULSES, 2))
            for method in ("duke_like", "random", "information_optimal"):
                result, steps = run_cell(theta, method, SEED+cell*37+len(method),
                                         common_noise, mismatch)
                outcomes.append({"scenario": scenario, "cell": cell,
                                 "method": method, **result})
                protocols += [{"scenario": scenario, "cell": cell,
                               "method": method, **step} for step in steps]
    for filename, rows in (("optimal_protocol_pilot_cells.csv", outcomes),
                           ("optimal_protocol_pilot_choices.csv", protocols)):
        with (HERE / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    metrics = [key for key in outcomes[0] if key not in {"scenario", "cell", "method"}]
    scenarios = ("in_model", "misspecified")
    methods = ("duke_like", "random", "information_optimal")
    summary = {scenario: {method: {metric: float(np.median([
        row[metric] for row in outcomes if row["scenario"] == scenario and
        row["method"] == method])) for metric in metrics} for method in methods}
        for scenario in scenarios}
    # Paired cell bootstrap for optimal-versus-other median error differences.
    boot_rng = np.random.default_rng(SEED+999)
    uncertainty = {}
    for scenario in scenarios:
        uncertainty[scenario] = {}
        for reference in ("duke_like", "random"):
            uncertainty[scenario][reference] = {}
            for metric in metrics[:6]:
                by_cell = np.array([next(r[metric] for r in outcomes
                                         if r["scenario"] == scenario and r["cell"] == cell
                                         and r["method"] == "information_optimal")-
                                    next(r[metric] for r in outcomes
                                         if r["scenario"] == scenario and r["cell"] == cell
                                         and r["method"] == reference)
                                    for cell in range(N_CELLS)])
                samples = np.median(boot_rng.choice(by_cell,
                                                    size=(3000, N_CELLS), replace=True), axis=1)
                uncertainty[scenario][reference][metric] = {
                    "paired_median_difference": float(np.median(by_cell)),
                    "cell_bootstrap_95pct": [float(v) for v in
                                             np.quantile(samples, [.025, .975])]}
    output = {"seed": SEED, "synthetic_cells": N_CELLS, "pulses_per_cell": N_PULSES,
              "parameters": "effective F50, delay, recovery tau, gain",
              "summary_median": summary,
              "optimal_minus_reference_paired": uncertainty,
              "limitation": "A stylized simulator with Gaussian peak/time noise. One scenario matches the fitted model. The second generates data with a 35-nN sigmoid width and a two-timescale recovery kernel while fitting the simpler 25-nN single-exponential model. Neither scenario is calibrated to Duke or PIEZO physiology. The misspecified scenario challenges transfer but remains synthetic."}
    (HERE / "optimal_protocol_pilot_results.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
