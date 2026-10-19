"""Physical order contrast versus interval before a common anchor pulse."""
import json
from pathlib import Path

import numpy as np

import matched_order_probe as order
import predictive_risk_protocol as p

HERE = Path(__file__).resolve().parent
FAMILIES = ("no_history", "single_sigmoid", "dual_sigmoid", "force_fatigue")
ANCHOR_INTERVALS = (50., 200., 800., 5000.)
rng = np.random.default_rng(20260939)
result = {}
for family in FAMILIES:
    cells = [p.prior.draw_particles(rng, 1)[0] for _ in range(100)]
    result[family] = {}
    for dt in ANCHOR_INTERVALS:
        spans = []
        for theta in cells:
            peaks = []
            for perm in order.ORDERS:
                seq = order.protocol(perm)[:5]+[(330., 10., dt), p.CHALLENGE]
                peaks.append(p.truth_output(theta, seq, family)[-1, 0])
            spans.append((max(peaks)-min(peaks))/max(np.mean(peaks), 1e-9))
        result[family][str(int(dt))] = float(np.median(spans))
(HERE/"matched_order_anchor_interval.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
