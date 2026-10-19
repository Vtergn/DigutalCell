"""Target-blind two-parameter MCT calibration for the 25-variant grid.

Only low/mid observations enter the inverse fit. A native MATLAB ODE run at
the inferred parameter values supplies the high-load twin prediction.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parent / "mct_feedback_audit"
SCORE = ROOT / "reverse_grid25_score.json"
OUT = ROOT / "reverse_grid25_inferred_params.csv"
DETAIL = ROOT / "reverse_grid25_inverse_diagnostics.json"
GAINS = (0.25, 1.25)
SCALES = (0.90, 1.10)


def main() -> None:
    rows = json.loads(SCORE.read_text(encoding="utf-8"))["rows"]
    theta = np.array([[r["mct_gain"], r["pCa0_scale"]] for r in rows])
    inputs = (theta - [GAINS[0], SCALES[0]]) / [GAINS[1]-GAINS[0], SCALES[1]-SCALES[0]]
    gridline = np.linspace(0, 1, 41)
    grid = np.array(np.meshgrid(gridline, gridline)).reshape(2, -1).T
    output_rows, diagnostics = [], []
    for i, row in enumerate(rows):
        training = np.arange(len(rows)) != i
        for mode, loads in (("low", ("l02",)), ("lowmid", ("l02", "l06"))):
            observed = np.array([row[load][field] for load in loads
                                 for field in ("APD_above_minus67_mV_ms", "Ca_peak_uM")])
            forward = np.array([[r[load][field] for load in loads
                                 for field in ("APD_above_minus67_mV_ms", "Ca_peak_uM")]
                                for r in rows])
            mean = forward[training].mean(0)
            sd = np.maximum(forward[training].std(0), 1e-8)
            z = (forward[training] - mean) / sd
            target = (observed - mean) / sd
            emulator = RBFInterpolator(inputs[training], z,
                                       kernel="thin_plate_spline", smoothing=1e-6)

            def loss(u: np.ndarray) -> float:
                difference = emulator(np.asarray(u)[None, :])[0] - target
                return float(difference @ difference)

            grid_loss = np.sum((emulator(grid) - target[None, :])**2, axis=1)
            starters = grid[np.argsort(grid_loss)[:3]]
            opts = [minimize(loss, start, method="L-BFGS-B", bounds=((0, 1), (0, 1)))
                    for start in starters]
            best = min(opts, key=lambda opt: opt.fun)
            inferred = np.array([GAINS[0], SCALES[0]]) + np.array(best.x) * [
                GAINS[1]-GAINS[0], SCALES[1]-SCALES[0]]
            output_rows.append({"index": i, "mode": mode, "mct_gain": inferred[0],
                                "pCa0_scale": inferred[1]})
            diagnostics.append({"id": row["id"], "mode": mode,
                                "fit_loss_standardized_sse": float(best.fun),
                                "true_mct_gain": row["mct_gain"],
                                "inferred_mct_gain": float(inferred[0]),
                                "true_pCa0_scale": row["pCa0_scale"],
                                "inferred_pCa0_scale": float(inferred[1])})
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=("index", "mode", "mct_gain", "pCa0_scale"))
        writer.writeheader()
        writer.writerows(output_rows)
    DETAIL.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    for mode in ("low", "lowmid"):
        dx = np.array([[r["inferred_mct_gain"]-r["true_mct_gain"],
                        r["inferred_pCa0_scale"]-r["true_pCa0_scale"]]
                       for r in diagnostics if r["mode"] == mode])
        print(mode, "parameter MAE", np.mean(np.abs(dx), axis=0))
    print(OUT)


if __name__ == "__main__":
    main()
