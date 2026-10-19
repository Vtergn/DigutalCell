"""Synthetic, sequential mechanical protocol comparison under model uncertainty.

This is an illustrative methods benchmark, not a calibrated PIEZO simulator.
All designers see only past noisy peak/time observations. Evaluation families are
never used by the ensemble-robust designer. The family-informed reference is
given the true *family* but not cell parameters or future observations.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.special import expit, logsumexp

HERE = Path(__file__).resolve().parent
SEED = 20260928
N_CELLS = 20
N_SHOTS = 6
N_PARTICLES = 128
NOISE = np.array([.035, .8])
FORCES = (200., 260., 330., 420.)
SPEEDS = (5., 20.)
INTERVALS = (50., 200., 800.)
CANDIDATES = [(f, v, dt) for f in FORCES for v in SPEEDS for dt in INTERVALS]
DEVELOPMENT = ("single_sigmoid", "dual_sigmoid", "no_history", "power_single")
UNSEEN = ("threshold_drift", "rate_sensitive", "gain_drift", "power_dual")
FAMILIES = DEVELOPMENT + UNSEEN
METHODS = ("duke_like", "random_diverse", "local_fisher", "robust_ensemble", "family_informed")
CHALLENGES = ((260., 7., 50.), (330., 12., 200.), (420., 7., 800.))


def model_output(theta: np.ndarray, protocol: list[tuple[float, float, float]],
                 family: str) -> np.ndarray:
    """Return Nparticles x Npulses x [peak, time] for a protocol."""
    f50, delay, log_tau, log_gain = theta.T
    tau, gain = np.exp(log_tau), np.exp(log_gain)
    ready = np.ones(len(theta))
    post = ready.copy()
    out = []
    for n, (force, speed, interval) in enumerate(protocol):
        if n:
            if family == "no_history":
                ready = np.ones(len(theta))
            else:
                residual = np.exp(-interval/tau)
                if family in ("dual_sigmoid", "power_dual"):
                    residual = .7*residual+.3*np.exp(-interval/(4*tau))
                ready = 1-(1-post)*residual
        shift = 4*n if family == "threshold_drift" else 0.
        if family == "rate_sensitive":
            shift = -18*np.log2(speed/10.)
        effective_f50 = f50+shift
        if family.startswith("power"):
            activation = np.clip((force-effective_f50+50)/200, 0, 1)**1.5
        else:
            activation = expit((force-effective_f50)/25.)
        current_gain = gain*(1-.05*n) if family == "gain_drift" else gain
        peak = current_gain*ready*activation
        time = effective_f50/speed+delay
        out.append(np.stack((peak, time), axis=-1))
        post = ready*(1-.70*activation)
    return np.stack(out, axis=1)


def draw_particles(rng: np.random.Generator, n: int) -> np.ndarray:
    return np.column_stack((rng.uniform(150, 250, n),
                            rng.uniform(0, 12, n),
                            np.log(rng.uniform(80, 800, n)),
                            np.log(rng.uniform(.75, 1.4, n))))


def posterior(particles: np.ndarray, protocol: list, measured: list,
              family_set: tuple[str, ...], hedge: bool = True) -> dict:
    """Particle posterior with mild likelihood tempering for finite-bank stability."""
    if not protocol:
        return {fam: np.full(len(particles), 1/(len(family_set)*len(particles)))
                for fam in family_set}
    logw = []
    observed = np.asarray(measured)
    for fam in family_set:
        prediction = model_output(particles, protocol, fam)
        residual = (prediction-observed[None, :, :])/NOISE
        # Timing is unobservable for weak/non-responses.
        residual[:, observed[:, 0] < .12, 1] = 0.
        logw.append(-.5*np.sum(residual**2, axis=(1, 2))/2.0)
    logw = np.asarray(logw)
    evidence = logsumexp(logw, axis=1)-np.log(len(particles))
    model_weights = np.exp(evidence-logsumexp(evidence))
    if hedge and len(family_set) > 1:
        model_weights = .7*model_weights+.3/len(family_set)
    bank = {}
    for j, fam in enumerate(family_set):
        conditional = np.exp(logw[j]-logsumexp(logw[j]))
        bank[fam] = model_weights[j]*conditional
    return bank


def weighted_mean_cov(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.sum(weights[:, None]*values, axis=0)
    center = values-mean
    return mean, (weights[:, None]*center).T@center


def score_candidate(particles: np.ndarray, bank: dict, history: list,
                    candidate: tuple, mode: str) -> float:
    """Myopic challenge variance reduction + model separation + space filling."""
    all_now, all_future, all_w, means, within = [], [], [], [], []
    for fam, w in bank.items():
        protocol = history+[candidate]
        now = model_output(particles, protocol, fam)[:, -1, :]/NOISE
        future = np.mean(np.stack([
            model_output(particles, protocol+[challenge], fam)[:, -1, :]/NOISE
            for challenge in CHALLENGES], axis=0), axis=0)
        all_now.append(now)
        all_future.append(future)
        all_w.append(w)
        local = w/w.sum()
        model_mean, model_cov = weighted_mean_cov(now, local)
        means.append(model_mean)
        within.append(np.trace(model_cov))
    now = np.concatenate(all_now)
    future = np.concatenate(all_future)
    weights = np.concatenate(all_w)
    mean_now, cov_now = weighted_mean_cov(now, weights)
    mean_future, _ = weighted_mean_cov(future, weights)
    cross = (weights[:, None]*(future-mean_future)).T@(now-mean_now)
    try:
        reduction = np.trace(cross@np.linalg.solve(cov_now+np.eye(2), cross.T))
    except np.linalg.LinAlgError:
        reduction = 0.
    reduction = np.log1p(max(reduction, 0.))
    if len(bank) > 1:
        mass = np.array([w.sum() for w in all_w])
        model_means = np.asarray(means)
        between = np.sum(mass[:, None]*(model_means-mean_now)**2)
        discrimination = np.log1p(between/(1+np.dot(mass, within)))
    else:
        discrimination = 0.
    force, speed, interval = candidate
    if history:
        coverage = sum(value not in [p[j] for p in history] for j, value in enumerate(candidate))
    else:
        coverage = 2.  # no interval information exists before the first shot
    cost = .15*((force-200)/220)**2+.04*(speed/20)**2
    if mode == "family_informed":
        return reduction+.16*coverage-cost
    return reduction+.75*discrimination+1.50*coverage-cost


def choose_next(method: str, rng: np.random.Generator, particles: np.ndarray,
                history: list, measured: list, true_family: str) -> tuple:
    n = len(history)
    if method == "duke_like":
        return (float(200+35*n), 10., 5000.)
    if method == "random_diverse":
        candidates = CANDIDATES if n else [x for x in CANDIDATES if x[2] == 800]
        novelty = [sum(value not in [p[j] for p in history]
                       for j, value in enumerate(x) if j != 2 or n > 0)
                   for x in candidates]
        best = [x for x, score in zip(candidates, novelty) if score == max(novelty)]
        return best[int(rng.integers(len(best)))]
    if method == "local_fisher":
        # Same local D-optimal rule as the first pilot; fit one in-model family
        # by weighted particle mean. This is selection only, not evaluation.
        bank = posterior(particles, history, measured, ("single_sigmoid",), hedge=False)
        estimate = np.sum(bank["single_sigmoid"][:, None]*particles, axis=0)
        from optimal_mechanical_protocol_pilot import fisher
        scored = []
        for x in CANDIDATES:
            sign, determinant = np.linalg.slogdet(fisher(estimate, history+[x]))
            scored.append(determinant if sign > 0 else -np.inf)
        return CANDIDATES[int(np.argmax(scored))]
    families = DEVELOPMENT if method == "robust_ensemble" else (true_family,)
    bank = posterior(particles, history, measured, families)
    candidates = CANDIDATES if n else [x for x in CANDIDATES if x[2] == 800]
    scores = [score_candidate(particles, bank, history, x, method) for x in candidates]
    return candidates[int(np.argmax(scores))]


def outcome_for_cell(theta: np.ndarray, family: str, method: str,
                     particles: np.ndarray, noise: np.ndarray, rng: np.random.Generator) -> tuple[dict, list]:
    history, measured, choices = [], [], []
    for n in range(N_SHOTS):
        x = choose_next(method, rng, particles, history, measured, family)
        history.append(x)
        truth = model_output(theta[None, :], history, family)[0, -1]
        measured.append(truth+noise[n])
        choices.append({"shot": n+1, "force_nN": x[0], "speed_nN_per_ms": x[1],
                        "interval_ms": x[2]})
    # Identical blinded estimator for every protocol; only the selected shots vary.
    bank = posterior(particles, history, measured, DEVELOPMENT)
    challenge_error = []
    challenge_timing_error = []
    for challenge in CHALLENGES:
        pred = sum(np.sum(w[:, None]*model_output(particles, history+[challenge], fam)[:, -1, :], axis=0)
                   for fam, w in bank.items())
        actual = model_output(theta[None, :], history+[challenge], family)[0, -1]
        challenge_error.append(abs(pred[0]-actual[0]))
        challenge_timing_error.append(abs(pred[1]-actual[1]))
    result = {"challenge_peak_mae": float(np.mean(challenge_error)),
              "challenge_peak_max_error": float(np.max(challenge_error)),
              "challenge_timing_mae_ms": float(np.mean(challenge_timing_error)),
              "force_levels": len({x[0] for x in history}),
              "speed_levels": len({x[1] for x in history}),
              "interval_levels": len({x[2] for x in history[1:]}),
              "short_interval_count": sum(x[2] == 50 for x in history[1:]),
              "development_model_correct": int(max(bank, key=lambda f: bank[f].sum()) == family)
              if family in DEVELOPMENT else -1}
    return result, choices


def main() -> None:
    rng = np.random.default_rng(SEED)
    particles = draw_particles(np.random.default_rng(SEED+1), N_PARTICLES)
    records, choices = [], []
    for family in FAMILIES:
        for cell in range(N_CELLS):
            theta = draw_particles(rng, 1)[0]
            noise = rng.normal(0, NOISE, (N_SHOTS, 2))
            for method in METHODS:
                method_rng = np.random.default_rng(SEED+100000*FAMILIES.index(family)+
                                                   100*cell+METHODS.index(method))
                result, protocol = outcome_for_cell(theta, family, method,
                                                    particles, noise, method_rng)
                records.append({"family": family, "split": "unseen" if family in UNSEEN
                                else "development", "cell": cell, "method": method, **result})
                choices.extend({"family": family, "cell": cell, "method": method, **step}
                               for step in protocol)
        print(f"finished {family}", flush=True)
    for name, rows in (("robust_protocol_cells.csv", records),
                       ("robust_protocol_choices.csv", choices)):
        with (HERE/name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {}
    for split in ("development", "unseen"):
        summary[split] = {}
        for method in METHODS:
            data = [r for r in records if r["split"] == split and r["method"] == method]
            per_family = {fam: float(np.mean([r["challenge_peak_mae"] for r in data
                                              if r["family"] == fam]))
                          for fam in (DEVELOPMENT if split == "development" else UNSEEN)}
            summary[split][method] = {
                "mean_challenge_mae": float(np.mean([r["challenge_peak_mae"] for r in data])),
                "mean_challenge_timing_mae_ms": float(np.mean([r["challenge_timing_mae_ms"] for r in data])),
                "family_90pct_mae": float(np.quantile(list(per_family.values()), .9)),
                "worst_family_mae": max(per_family.values()),
                "per_family_mae": per_family,
                "development_model_accuracy": float(np.mean([r["development_model_correct"] for r in data]))
                if split == "development" else None,
                "short_interval_fraction": float(np.sum([r["short_interval_count"] for r in data])/(len(data)*5)),
                "mean_force_levels": float(np.mean([r["force_levels"] for r in data])),
                "mean_speed_levels": float(np.mean([r["speed_levels"] for r in data])),
                "mean_interval_levels": float(np.mean([r["interval_levels"] for r in data]))}
    # Paired bootstrap over cells within each generator. Generator count is only
    # four per split, so these intervals are conditional on the chosen zoo.
    boot = {}
    for split in ("development", "unseen"):
        method_a = np.array([r["challenge_peak_mae"] for r in records
                             if r["split"] == split and r["method"] == "robust_ensemble"])
        method_b = np.array([r["challenge_peak_mae"] for r in records
                             if r["split"] == split and r["method"] == "random_diverse"])
        differences = (method_a-method_b).reshape(4, N_CELLS)
        bs = np.stack([np.mean(differences[np.arange(4)[:, None],
                                          rng.integers(N_CELLS, size=(4, N_CELLS))])
                       for _ in range(3000)])
        boot[split] = {"robust_minus_random_mean": float(np.mean(differences)),
                       "cell_bootstrap_95pct_conditional_on_families": [float(v) for v in np.quantile(bs, [.025, .975])]}
    output = {"seed": SEED, "cells_per_family": N_CELLS,
              "shots_per_cell": N_SHOTS, "particles": N_PARTICLES,
              "development_families": DEVELOPMENT, "unseen_families": UNSEEN,
              "challenge_panel": CHALLENGES, "summary": summary,
              "paired_comparison": boot,
              "limitations": ["All results are synthetic and parameter ranges are illustrative, not estimated from Duke.",
                              "Family-informed design knows the true generator class and is a reference, not a feasible prospective method.",
                              "Generator-family holdout tests four chosen discrepancies; it is not evidence of universal out-of-model robustness.",
                              "The score weights and candidate grid were fixed before this run; engineering cost is illustrative, not a damage model."]}
    (HERE/"robust_protocol_results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "paired_comparison": boot}, indent=2))


if __name__ == "__main__":
    main()
