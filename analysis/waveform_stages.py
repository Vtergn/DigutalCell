"""Describe Piezo onset, peak, and identifiable postpeak decay segments."""

from __future__ import annotations

import csv
from collections import defaultdict

import numpy as np
from scipy.optimize import curve_fit

from dynamic_input_screen import HERE, load_traces, onset_ms
from one_shot_work import read_metadata, first_calibration


def exponential(time: np.ndarray, baseline: float, amplitude: float, tau_ms: float) -> np.ndarray:
    return baseline + amplitude*np.exp(-time/tau_ms)


def fit_decay(current: np.ndarray, peak: int, threshold: float) -> dict:
    end = min(250, len(current))
    if peak >= end-20 or float(current[peak]) < threshold:
        return {"tau_decay_ms": float("nan"), "decay_fit_r2": float("nan"), "decay_fit_eligible": 0}
    y = np.maximum(current[peak:end], 0)
    time = np.arange(len(y), dtype=float)
    tail = float(np.median(y[-10:]))
    if tail > .9*float(np.max(y)):
        return {"tau_decay_ms": float("nan"), "decay_fit_r2": float("nan"), "decay_fit_eligible": 0}
    try:
        parameters, _ = curve_fit(exponential, time, y,
                                  p0=(tail, max(float(y[0])-tail, 1.0), 30.0),
                                  bounds=([0, 0, 1], [max(float(np.max(y)), 1), 5*max(float(np.max(y)), 1), 500]),
                                  maxfev=5000)
    except (RuntimeError, ValueError):
        return {"tau_decay_ms": float("nan"), "decay_fit_r2": float("nan"), "decay_fit_eligible": 0}
    prediction = exponential(time, *parameters)
    variance = float(np.sum((y-np.mean(y))**2))
    r2 = 1-float(np.sum((y-prediction)**2))/max(variance, 1e-12)
    eligible = bool(r2 >= .7 and parameters[1] >= .2*float(np.max(y)) and parameters[2] < 490)
    return {"tau_decay_ms": float(parameters[2]) if eligible else float("nan"),
            "decay_fit_r2": r2, "decay_fit_eligible": int(eligible)}


def main() -> None:
    by_cell = read_metadata()
    calibration = {cell: first_calibration(rows) for cell, rows in by_cell.items()}
    traces, _ = load_traces(by_cell, calibration)
    output = []
    for cell, row in calibration.items():
        if row is None or row["channel"] not in {"mp1", "mp2"}:
            continue
        cal_sweep = int(row["sweep"])
        voltage = traces[(cell, cal_sweep)]["voltage_mV"]
        threshold = max(10.0, 5*float(row["stdev_pA"]))
        for stimulus in by_cell[cell]:
            sweep = int(stimulus["sweep"])
            if sweep < cal_sweep:
                continue
            trace = traces[(cell, sweep)]
            if abs(trace["voltage_mV"]-voltage) > 5:
                continue
            current = trace["current"]
            force = trace["F"]
            onset = onset_ms(current, threshold, trace["contact_ms"]-5)
            peak_start = max(trace["contact_ms"]-5, 0)
            peak = peak_start + int(np.argmax(current[peak_start:]))
            force_peak = int(np.argmax(force[:200]))
            force_decay = fit_decay(force, force_peak, max(float(np.max(force[:200]))*.01, 1.0))
            has_event = np.isfinite(onset)
            output.append({
                "cell_id": cell, "channel": row["channel"], "date": row["date"],
                "sweep": sweep, "is_calibration": int(sweep == cal_sweep),
                "voltage_mV": trace["voltage_mV"], "event_threshold_pA": threshold,
                "has_event": int(has_event), "onset_ms": onset,
                "peak_time_ms": float(peak) if has_event else float("nan"),
                "force_peak_time_ms": float(force_peak),
                "current_minus_force_peak_ms": float(peak-force_peak) if has_event else float("nan"),
                "peak_pA": float(np.max(current)),
                "force_at_250_fraction": float(force[249]/max(float(np.max(force[:200])), 1.0)),
                "current_retention_20ms": float(np.median(current[peak+20:peak+25])/max(float(current[peak]), 1.0))
                if has_event and peak < 225 else float("nan"),
                "force_retention_20ms": float(np.median(force[force_peak+20:force_peak+25])/max(float(force[force_peak]), 1.0))
                if force_peak < 225 else float("nan"),
                "force_decay_tau_ms": force_decay["tau_decay_ms"],
                "force_decay_fit_r2": force_decay["decay_fit_r2"],
                "peak_to_250ms_fraction": float((current[peak]-np.median(current[240:250])) / max(current[peak], 1))
                if has_event and peak < 230 else float("nan"),
                **(fit_decay(current, peak, threshold) if has_event else
                   {"tau_decay_ms": float("nan"), "decay_fit_r2": float("nan"), "decay_fit_eligible": 0}),
            })
    with (HERE / "waveform_stage_observations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    for channel in ("mp1", "mp2"):
        future = [r for r in output if r["channel"] == channel and not r["is_calibration"]]
        decays = [r for r in future if r["decay_fit_eligible"]]
        print(channel, "future", len(future), "events", sum(r["has_event"] for r in future),
              "decay_fit", len(decays), "decay_cells", len({r["cell_id"] for r in decays}),
              "median_tau_ms", round(float(np.median([r["tau_decay_ms"] for r in decays])), 1) if decays else float("nan"))


if __name__ == "__main__":
    main()
