"""Coverage-constrained sequential design for a fixed seventh challenge.

Exploratory synthetic go/no-go benchmark. The action rule is fixed before
evaluation on four new generator classes. No parameter is calibrated to cells.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.special import erf, expit, logsumexp

import robust_protocol_design as prior

HERE = Path(__file__).resolve().parent
DEVELOPMENT = prior.FAMILIES
UNSEEN = ("force_fatigue", "mechanical_hysteresis", "nonmonotone_rate", "late_gain_switch")
MODELS = prior.DEVELOPMENT
METHODS = ("monotonic", "balanced_random", "static_maximin", "local_fisher", "predictive_minimax")
GRID = prior.CANDIDATES
ANCHOR = (330., 10., 800.)  # sixth pulse, identical across protocols
CHALLENGE = (310., 10., 200.)  # seventh pulse, blind and identical
NOISE = prior.NOISE
N_ADAPTIVE = 5
N_PARTICLES = 192
N_CELLS_PER_FAMILY_SEED = 10
SEEDS = (17, 29, 43)
DELTA_SD = np.array([.025, .025, .025, .025, .025, .018])


def discrepancy_basis(action: tuple, pulse_index: int) -> np.ndarray:
    force, speed, interval = action
    force_knots = np.maximum(0., 1.-np.abs(force-np.array([200., 310., 420.]))/110.)
    return np.r_[force_knots, np.log2(speed/10.),
                 np.exp(-interval/200.), (pulse_index-2.5)/2.5]


def model_bank_output(theta: np.ndarray, delta: np.ndarray,
                      protocol: list, family: str) -> np.ndarray:
    output = prior.model_output(theta, protocol, family).copy()
    for n, action in enumerate(protocol):
        output[:, n, 0] = np.maximum(0., output[:, n, 0] +
                                  delta@discrepancy_basis(action, n))
    return output


def new_truth(theta: np.ndarray, protocol: list, family: str) -> np.ndarray:
    """Generators excluded entirely from model development and model bank."""
    f50, delay, log_tau, log_gain = theta
    tau, gain = np.exp(log_tau), np.exp(log_gain)
    ready, post, prior_activation = 1., 1., 0.
    out = []
    for n, (force, speed, interval) in enumerate(protocol):
        if n:
            tau_here = tau*(1+.7*prior_activation) if family == "force_fatigue" else tau
            ready = 1-(1-post)*np.exp(-interval/tau_here)
        local_f50 = f50
        if family == "mechanical_hysteresis" and n:
            local_f50 += 32*prior_activation*np.exp(-interval/400.)
        if family == "nonmonotone_rate":
            local_f50 += 17*abs(np.log2(speed/10.))
        activation = expit((force-local_f50)/25.)
        if family == "late_gain_switch":
            current_gain = gain*(.72 if n >= 3 else 1.)
        else:
            current_gain = gain
        out.append((current_gain*ready*activation, local_f50/speed+delay))
        depletion = .90*activation**2 if family == "force_fatigue" else .70*activation
        post = ready*(1-depletion)
        prior_activation = activation
    return np.asarray(out)


def truth_output(theta: np.ndarray, protocol: list, family: str) -> np.ndarray:
    if family in UNSEEN:
        return new_truth(theta, protocol, family)
    return prior.model_output(theta[None, :], protocol, family)[0]


def particle_bank(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    theta = prior.draw_particles(rng, N_PARTICLES)
    delta = rng.normal(0, DELTA_SD, size=(N_PARTICLES, len(DELTA_SD)))
    return theta, delta


def posterior(theta: np.ndarray, delta: np.ndarray, history: list,
              observed: list) -> dict:
    if not history:
        return {fam: np.full(len(theta), 1/(len(MODELS)*len(theta))) for fam in MODELS}
    seen = np.asarray(observed)
    likelihood = []
    for fam in MODELS:
        pred = model_bank_output(theta, delta, history, fam)
        res = (pred-seen[None, :, :])/NOISE
        res[:, seen[:, 0] < .12, 1] = 0.
        likelihood.append(-np.sum(res**2, axis=(1, 2))/5.)
    logw = np.asarray(likelihood)
    evidence = logsumexp(logw, axis=1)-np.log(len(theta))
    model_w = np.exp(evidence-logsumexp(evidence))
    model_w = .85*model_w+.15/len(MODELS)
    return {fam: model_w[j]*np.exp(logw[j]-logsumexp(logw[j]))
            for j, fam in enumerate(MODELS)}


def coverage_feasible(history: list, action: tuple) -> bool:
    if not history and action[2] != 800.:
        return False
    trial = history+[action]
    if len(trial) > N_ADAPTIVE:
        return False
    intervals = [x[2] for x in trial[1:]]
    if any(intervals.count(x) > 2 for x in prior.INTERVALS):
        return False
    left = N_ADAPTIVE-len(trial)
    need_force = len({200., 420.}-{x[0] for x in trial})
    need_rate = len({5., 20.}-{x[1] for x in trial})
    need_interval = len(set(prior.INTERVALS)-set(intervals))
    return max(need_force, need_rate, need_interval) <= left


def candidates(history: list) -> list:
    return [x for x in GRID if coverage_feasible(history, x)]


def normalized_distance(a: tuple, b: tuple, first_a: bool, first_b: bool) -> float:
    force = (a[0]-b[0])/220.
    speed = (a[1]-b[1])/15.
    interval = 0. if first_a or first_b else (np.log(a[2])-np.log(b[2]))/np.log(16.)
    return float(np.sqrt(force**2+speed**2+interval**2))


def min_distance(protocol: list) -> float:
    return min(normalized_distance(a, b, i == 0, j == 0)
               for i, a in enumerate(protocol) for j, b in enumerate(protocol) if j > i)


def complete_maximin(history: list) -> list:
    out = history.copy()
    while len(out) < N_ADAPTIVE:
        options = candidates(out)
        if not out:
            x = (200., 5., 800.)
        elif len(out) == 1:
            x = max(options, key=lambda a: normalized_distance(a, out[0], False, True))
        else:
            x = max(options, key=lambda a: min_distance(out+[a]))
        out.append(x)
    return out


def static_maximin(seed: int) -> list:
    """Random-restart maximin over feasible complete protocols, then freeze."""
    rng = np.random.default_rng(seed)
    best, best_score = None, -np.inf
    for _ in range(500):
        out = []
        for _ in range(N_ADAPTIVE):
            options = candidates(out)
            out.append(options[int(rng.integers(len(options)))])
        score = min_distance(out)
        if score > best_score:
            best, best_score = out, score
    return best


def mean_cov(x: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.sum(w[:, None]*x, axis=0)
    center = x-mean
    return mean, (center*w[:, None]).T@center


def expected_abs_normal(bias: float, sd: float) -> float:
    if sd < 1e-12:
        return abs(bias)
    return float(sd*np.sqrt(2/np.pi)*np.exp(-.5*(bias/sd)**2)+
                 abs(bias)*erf(abs(bias)/(np.sqrt(2)*sd)))


def minimax_risk(theta: np.ndarray, delta: np.ndarray, bank: dict,
                 history: list, action: tuple) -> float:
    """Worst model-class Gaussian posterior predictive L1 risk approximation.

    Hypothetical candidate observation updates a common linear Bayes predictor
    of the *fixed* seventh challenge. Future design shots are filled with a
    deterministic maximin policy and not hypothetically observed here.
    """
    full = complete_maximin(history+[action])+[ANCHOR, CHALLENGE]
    current_index = len(history)
    x_by_family, z_by_family = {}, {}
    all_x, all_z, all_w = [], [], []
    for fam, w in bank.items():
        pred = model_bank_output(theta, delta, full, fam)
        x, z = pred[:, current_index, :], pred[:, -1, 0]
        x_by_family[fam], z_by_family[fam] = x, z
        all_x.append(x)
        all_z.append(z)
        all_w.append(w)
    x_all, z_all, w_all = np.concatenate(all_x), np.concatenate(all_z), np.concatenate(all_w)
    mean_x, cov_x = mean_cov(x_all, w_all)
    mean_z = np.dot(w_all, z_all)
    cross = ((x_all-mean_x)*w_all[:, None]).T@(z_all-mean_z)
    k = np.linalg.solve(cov_x+np.diag(NOISE**2), cross)
    risk = []
    for fam, w in bank.items():
        local = w/w.sum()
        x, z = x_by_family[fam], z_by_family[fam]
        mx, cx = mean_cov(x, local)
        mz = np.dot(local, z)
        centered_z = z-mz
        var_z = np.dot(local, centered_z**2)
        cross_m = ((x-mx)*local[:, None]).T@centered_z
        bias = mz-mean_z-np.dot(k, mx-mean_x)
        variance = var_z+np.dot(k, (cx+np.diag(NOISE**2))@k)-2*np.dot(k, cross_m)
        risk.append(expected_abs_normal(bias, np.sqrt(max(variance, 0.))))
    return max(risk)


def next_action(method: str, rng: np.random.Generator, theta: np.ndarray,
                delta: np.ndarray, history: list, observed: list,
                fixed_static: list) -> tuple:
    n = len(history)
    if method == "monotonic":
        # A feasible monotonic force baseline; speed and intervals are fixed
        # except where coverage forces variation. This is synthetic, not Duke.
        preferred = [200., 260., 330., 420., 420.][n]
        options = candidates(history)
        return min(options, key=lambda x: (abs(x[0]-preferred),
                                           abs(x[1]-10.), abs(x[2]-800.)))
    if method == "static_maximin":
        return fixed_static[n]
    if method == "balanced_random":
        options = candidates(history)
        scores = [sum(value not in [p[j] for p in history]
                      for j, value in enumerate(x) if j != 2 or n > 0)
                  for x in options]
        best = [x for x, value in zip(options, scores) if value == max(scores)]
        return best[int(rng.integers(len(best)))]
    if method == "local_fisher":
        from optimal_mechanical_protocol_pilot import fisher
        bank = posterior(theta, delta, history, observed)
        weight = bank["single_sigmoid"]/bank["single_sigmoid"].sum()
        estimate = np.sum(weight[:, None]*theta, axis=0)
        options = candidates(history)
        values = [np.linalg.slogdet(fisher(estimate, history+[x]))[1] for x in options]
        return options[int(np.argmax(values))]
    bank = posterior(theta, delta, history, observed)
    options = candidates(history)
    risks = [minimax_risk(theta, delta, bank, history, x) for x in options]
    return options[int(np.argmin(risks))]


def run_cell(real_theta: np.ndarray, family: str, method: str,
             particles: np.ndarray, delta: np.ndarray, noise: np.ndarray,
             rng: np.random.Generator, static: list) -> tuple[dict, list]:
    protocol, observed = [], []
    for _ in range(N_ADAPTIVE):
        action = next_action(method, rng, particles, delta, protocol, observed, static)
        protocol.append(action)
        y = truth_output(real_theta, protocol, family)[-1]
        observed.append(y+noise[len(observed)])
    protocol.append(ANCHOR)
    observed.append(truth_output(real_theta, protocol, family)[-1]+noise[-1])
    bank = posterior(particles, delta, protocol, observed)
    prediction = sum(np.dot(w, model_bank_output(particles, delta, protocol+[CHALLENGE], fam)[:, -1, 0])
                     for fam, w in bank.items())
    actual = truth_output(real_theta, protocol+[CHALLENGE], family)[-1, 0]
    intervals = [x[2] for x in protocol[1:5]]
    metrics = {"abs_challenge_error": abs(float(prediction-actual)),
               "true_challenge": float(actual), "pred_challenge": float(prediction),
               "short_count": intervals.count(50.),
               "force_span": max(x[0] for x in protocol[:5])-min(x[0] for x in protocol[:5]),
               "speed_levels": len({x[1] for x in protocol[:5]}),
               "interval_levels": len(set(intervals))}
    return metrics, protocol


def summarize(rows: list[dict]) -> dict:
    output = {}
    for split in ("development", "unseen"):
        subset = [r for r in rows if r["split"] == split]
        families = DEVELOPMENT if split == "development" else UNSEEN
        output[split] = {}
        for method in METHODS:
            chosen = [r for r in subset if r["method"] == method]
            family_means = {fam: float(np.mean([r["abs_challenge_error"] for r in chosen
                                                 if r["family"] == fam])) for fam in families}
            values = np.array([r["abs_challenge_error"] for r in chosen])
            output[split][method] = {
                "median_error": float(np.median(values)),
                "mean_error": float(np.mean(values)),
                "worst_family_mean_error": max(family_means.values()),
                "family_mean_errors": family_means,
                "mean_short_count": float(np.mean([r["short_count"] for r in chosen]))}
    return output


def main() -> None:
    records, choices = [], []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        particles, delta = particle_bank(seed+10000)
        fixed_static = static_maximin(seed+20000)
        for family in DEVELOPMENT+UNSEEN:
            for cell in range(N_CELLS_PER_FAMILY_SEED):
                truth = prior.draw_particles(rng, 1)[0]
                # Freeze three plausible parameter regimes before running any
                # held-out family. The designer's particle prior stays fixed.
                if seed == 29:
                    truth[0] += 15.
                    truth[2] += np.log(.7)
                elif seed == 43:
                    truth[0] -= 15.
                    truth[2] += np.log(1.5)
                noise = rng.normal(0, NOISE, size=(6, 2))
                for method in METHODS:
                    method_rng = np.random.default_rng(seed*100000+cell*100+
                                                       (DEVELOPMENT+UNSEEN).index(family)*10+
                                                       METHODS.index(method))
                    result, protocol = run_cell(truth, family, method, particles,
                                                delta, noise, method_rng, fixed_static)
                    records.append({"seed": seed, "family": family,
                                    "split": "unseen" if family in UNSEEN else "development",
                                    "cell": cell, "method": method, **result})
                    choices.extend({"seed": seed, "family": family, "cell": cell,
                                    "method": method, "shot": shot+1,
                                    "force": x[0], "speed": x[1], "interval": x[2]}
                                   for shot, x in enumerate(protocol))
            print(f"seed {seed}: {family}", flush=True)
    for filename, values in (("predictive_risk_cells.csv", records),
                             ("predictive_risk_choices.csv", choices)):
        with (HERE/filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    summary = summarize(records)
    output = {"seeds": SEEDS, "cells_per_family_seed": N_CELLS_PER_FAMILY_SEED,
              "true_parameter_regimes": {"17": "base prior", "29": "F50 +15 nN; tau times 0.7",
                                         "43": "F50 -15 nN; tau times 1.5"},
              "development_families": DEVELOPMENT, "unseen_families": UNSEEN,
              "models_available_to_designer": MODELS,
              "sixth_anchor": ANCHOR, "seventh_blind_challenge": CHALLENGE,
              "summary": summary,
              "go_no_go": {"median_below_random": summary["unseen"]["predictive_minimax"]["median_error"] < summary["unseen"]["balanced_random"]["median_error"],
                            "worst_family_below_random": summary["unseen"]["predictive_minimax"]["worst_family_mean_error"] < summary["unseen"]["balanced_random"]["worst_family_mean_error"],
                            "median_below_maximin": summary["unseen"]["predictive_minimax"]["median_error"] < summary["unseen"]["static_maximin"]["median_error"],
                            "worst_family_below_maximin": summary["unseen"]["predictive_minimax"]["worst_family_mean_error"] < summary["unseen"]["static_maximin"]["worst_family_mean_error"]}}
    (HERE/"predictive_risk_results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary["unseen"], "go_no_go": output["go_no_go"]}, indent=2))


if __name__ == "__main__":
    main()
