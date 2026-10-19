"""Counterfactual order audit with identical mechanical action multiset.

This is a synthetic prospective-design diagnostic. Real Duke and wedge records
do not randomize order sufficiently to estimate this effect directly.
"""
from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import numpy as np

import predictive_risk_protocol as p

HERE = Path(__file__).resolve().parent
SEED = 20260928
N_CELLS = 40
FAMILIES = ("no_history", "single_sigmoid", "dual_sigmoid",
            "force_fatigue", "mechanical_hysteresis", "late_gain_switch")
FIRST = (200., 5., 800.)
MOVABLE = ((260., 5., 800.), (330., 5., 200.),
           (420., 20., 50.), (420., 20., 200.))
ORDERS = list(itertools.permutations(range(4)))
LOW_TO_HIGH = (0, 1, 2, 3)
HIGH_TO_LOW = (3, 2, 1, 0)


def protocol(order: tuple[int, ...]) -> list:
    return [FIRST]+[MOVABLE[j] for j in order]+[p.ANCHOR]


def run_cell(theta: np.ndarray, family: str, noise: np.ndarray,
             particles: np.ndarray, delta: np.ndarray) -> list[dict]:
    out = []
    for order in ORDERS:
        actions = protocol(order)
        true_series = p.truth_output(theta, actions+[p.CHALLENGE], family)
        observed = true_series[:6]+noise
        bank = p.posterior(particles, delta, actions, observed)
        future = actions+[p.CHALLENGE]
        pred = sum(np.dot(w, p.model_bank_output(particles, delta, future, fam)[:, -1, 0])
                   for fam, w in bank.items())
        out.append({"order": "".join(map(str, order)),
                    "last_adaptive_force": actions[4][0],
                    "true_challenge_peak": float(true_series[-1, 0]),
                    "pred_challenge_peak": float(pred),
                    "abs_error": float(abs(pred-true_series[-1, 0]))})
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    particles, delta = p.particle_bank(SEED+10)
    records = []
    for family in FAMILIES:
        for cell in range(N_CELLS):
            theta = p.prior.draw_particles(rng, 1)[0]
            noise = rng.normal(0, p.NOISE, (6, 2))
            records.extend({"family": family, "cell": cell, **r}
                           for r in run_cell(theta, family, noise, particles, delta))
        print(f"finished {family}", flush=True)
    with (HERE/"matched_order_probe_cells.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {}
    for family in FAMILIES:
        grouped = [[r for r in records if r["family"] == family and r["cell"] == cell]
                   for cell in range(N_CELLS)]
        relative_spans = []
        physical_delta = []
        error_delta = []
        for rows in grouped:
            truth = np.array([r["true_challenge_peak"] for r in rows])
            relative_spans.append((truth.max()-truth.min())/max(truth.mean(), 1e-9))
            lo = next(r for r in rows if r["order"] == "0123")
            hi = next(r for r in rows if r["order"] == "3210")
            physical_delta.append(lo["true_challenge_peak"]-hi["true_challenge_peak"])
            error_delta.append(lo["abs_error"]-hi["abs_error"])
        summary[family] = {"median_relative_true_challenge_span": float(np.median(relative_spans)),
                           "median_true_peak_low_high_minus_high_low": float(np.median(physical_delta)),
                           "median_error_low_high_minus_high_low": float(np.median(error_delta)),
                           "cells_low_high_more_accurate": int(np.sum(np.array(error_delta) < 0)),
                           "median_low_high_error": float(np.median([next(r["abs_error"] for r in rows if r["order"] == "0123") for rows in grouped])),
                           "median_high_low_error": float(np.median([next(r["abs_error"] for r in rows if r["order"] == "3210") for rows in grouped]))}
    output = {"seed": SEED, "cells_per_family": N_CELLS,
              "same_action_multiset": [FIRST, *MOVABLE],
              "fixed_sixth_anchor": p.ANCHOR, "fixed_seventh_challenge": p.CHALLENGE,
              "permutations_per_cell": len(ORDERS), "summary": summary,
              "limitations": "Synthetic counterfactual. Order is not randomized in Duke/wedge; no direct biological order effect is established."}
    (HERE/"matched_order_probe_results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
