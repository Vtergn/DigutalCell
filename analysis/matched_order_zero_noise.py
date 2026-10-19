"""Separate measurement noise from protocol-order inference variation."""
import json
from pathlib import Path

import numpy as np

import matched_order_probe as order
import predictive_risk_protocol as p

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(604)
particles, delta = p.particle_bank(20260938)
result = {}
for family in ("no_history", "single_sigmoid", "force_fatigue"):
    true_spans, pred_spans = [], []
    for _ in range(20):
        theta = p.prior.draw_particles(rng, 1)[0]
        rows = order.run_cell(theta, family, np.zeros((6, 2)), particles, delta)
        true_spans.append(np.ptp([r["true_challenge_peak"] for r in rows]))
        pred_spans.append(np.ptp([r["pred_challenge_peak"] for r in rows]))
    result[family] = {"median_true_span": float(np.median(true_spans)),
                      "median_prediction_span": float(np.median(pred_spans))}
(HERE/"matched_order_zero_noise.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
