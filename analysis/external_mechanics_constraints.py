"""Read-only, protocol-scoped constraints from independent Piezo experiments.

This module deliberately does not fit Duke data. The source experiments differ
in cell type, stimulus geometry and measurement domain. Recovery values are
returned only for measured intervals; AFM speed interpolation concerns force
relaxation, never current. Caller must retain that distinction.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def _csv(name: str) -> list[dict]:
    with (HERE / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pressure_pulse_observation(interval_ms: int, pulse_number: int) -> dict:
    """WT PIEZO1 peak / first peak, N2a cell-attached, -60 mmHg, 250 ms pulses.

    No interpolation: interval series were ordered, and these ratios do not
    identify a unique recovery function. Valid gaps: 50/100/200/500/1000 ms.
    """
    rows = _csv("wedge_recovery_interval_summary.csv")
    match = [r for r in rows if int(r["interval_ms"]) == interval_ms and
             int(r["pulse_number"]) == pulse_number]
    if len(match) != 1:
        raise ValueError("Only measured intervals 50/100/200/500/1000 ms and pulses 1-6 are available")
    r = match[0]
    return {"median_peak_ratio": float(r["median_ratio"]),
            "q25_peak_ratio": float(r["q25_ratio"]),
            "q75_peak_ratio": float(r["q75_ratio"]),
            "independent_patch_count_within_condition": int(r["observed"]),
            "protocol": "N2a PIEZO1 WT; cell-attached; -60 mmHg x 250 ms; +5 mmHg between pulses",
            "cross_interval_order": "1000 to 50 ms in same patch; patch identities not encoded in source table"}


def afm_force_relaxation(speed_um_per_ms: float) -> dict:
    """Log-speed interpolation of cell-median mechanical relaxation only.

    Based on tipless AFM in N2a cells, 0.0025-0.1 um/ms. Interpolation gives
    a descriptive mechanical reference, not a PIEZO current or Duke prior.
    """
    rows = sorted(_csv("wedge_afm_velocity_summary.csv"),
                  key=lambda r: float(r["speed_um_per_ms"]))
    speeds = np.array([float(r["speed_um_per_ms"]) for r in rows])
    if not speeds[0] <= speed_um_per_ms <= speeds[-1]:
        raise ValueError(f"AFM speed must be within measured range {speeds[0]}-{speeds[-1]} um/ms")
    x = np.log(float(speed_um_per_ms))
    return {"force_relaxation_tau_ms": float(np.interp(
                x, np.log(speeds), [float(r["median_tau_ms"]) for r in rows])),
            "force_remaining_pct": float(np.interp(
                x, np.log(speeds), [float(r["median_force_remaining_pct"]) for r in rows])),
            "protocol": "N2a tipless AFM ramp-and-hold; cell medians, not current",
            "interpolated": not any(np.isclose(speed_um_per_ms, speeds, rtol=0, atol=1e-12))}


def flnb_indentation_effect(channel: str, endpoint: str) -> dict:
    """Independent FLNB-KO effect in HEK293 glass-probe indentation domain."""
    if channel not in ("Piezo1", "Piezo2"):
        raise ValueError("channel must be Piezo1 or Piezo2")
    if endpoint not in ("indentation_threshold_um", "peak_current_density_pA_per_pF"):
        raise ValueError("unknown endpoint")
    results = json.loads((HERE / "piezo2_flnb_results.json").read_text(encoding="utf-8"))
    return {**results["comparisons"][f"{channel}_{endpoint}"],
            "protocol": "SWELL1-KO HEK293; blunt glass-probe indentation; independent FLNB-KO and control cells",
            "transfer_warning": "Effect ratio is not a Duke AFM force-transfer coefficient"}
