"""ETH/eLife-constrained static tension-gating model on frozen Duke signed cohort.

This is a test of an externally constrained mechanical representation, not a
claim that local Duke membrane tension or single-channel open probability was
measured. Test cells provide exactly one responsive calibration sweep. Shared
effective offset and ETH force-transfer multiplier are selected only from
other-date cells' future sweeps. eLife Fig. 5 medians are fixed externally.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import expit

from adaptation_nested import force_threshold, postpeak_score
from causal_cnn_benchmark import build_records, metrics
from dynamic_input_screen import HERE, load_traces
from one_shot_work import first_calibration, read_metadata


T0_GRID_MN_PER_M = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
TRANSFER_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0)


def physical_parameters() -> dict[str, float]:
    eth = json.loads((HERE / "eth_stepwise_results.json").read_text(encoding="utf-8"))
    elife = json.loads((HERE / "external_physics_bridge_results.json").read_text(encoding="utf-8"))
    return {
        "eth_alpha_force_mN_per_m_per_nN": eth["indentation"]["mean_slope_tension_mN_per_m_per_dose"],
        "elife_piezo1_T50_mN_per_m": elife["elife"]["t50_mN_per_m"]["median"],
        "elife_piezo1_k_mN_per_m": elife["elife"]["slope_mN_per_m"]["median"],
    }


def gate(force: np.ndarray | float, threshold: float, t0: float, alpha: float,
         t50: float, k: float) -> np.ndarray:
    """Increase in a sigmoid readout above the one-shot detection force.

    Subtracting the value at Fth operationally represents baseline/current
    detection threshold. It is not an observed channel Popen at Fth.
    """
    force = np.asarray(force, dtype=float)
    above = expit((t0 + alpha * force - t50) / k)
    at_threshold = expit((t0 + alpha * threshold - t50) / k)
    return np.maximum(above - at_threshold, 0.0)


def predict(row: dict, t0: float, multiplier: float, physics: dict) -> np.ndarray:
    alpha = physics["eth_alpha_force_mN_per_m_per_nN"] * multiplier
    t50 = physics["elife_piezo1_T50_mN_per_m"]
    k = physics["elife_piezo1_k_mN_per_m"]
    calibration_drive = float(gate(row["cal_force_peak"], row["threshold"],
                                   t0, alpha, t50, k))
    if calibration_drive < 1e-12:
        return np.zeros_like(row["force"])
    response = gate(row["force"], row["threshold"], t0, alpha, t50, k)
    return row["cal_peak"] * response / calibration_drive


def assemble_records() -> list[dict]:
    """Reuse the fixed CNN prospective QC and signed current, add AFM force."""
    records = build_records()
    by_cell = read_metadata()
    calibration = {cell: first_calibration(rows) for cell, rows in by_cell.items()}
    traces, _ = load_traces(by_cell, calibration)
    for row in records:
        cal_meta = calibration[row["cell_id"]]
        cal_trace = traces[(row["cell_id"], int(cal_meta["sweep"]))]
        trace = traces[(row["cell_id"], row["sweep"])]
        row["force"] = trace["F"]
        row["cal_force"] = cal_trace["F"]
        row["cal_contact"] = int(cal_trace["contact_ms"])
        row["threshold"] = force_threshold(cal_meta, cal_trace)
        row["cal_force_peak"] = float(np.max(cal_trace["F"][:150]))
        if row["cal_force_peak"] <= row["threshold"]:
            raise ValueError("Calibration peak force must exceed threshold")
    return records


def observed_peak(row: dict) -> float:
    return max(float(np.max(row["target"][max(row["contact"]-5, 0):])), 0.0)


def select_shared(training: list[dict], physics: dict) -> dict:
    grouped = defaultdict(list)
    for row in training:
        grouped[row["cell_id"]].append(row)
    if not grouped:
        raise ValueError("No training cells")
    candidates = []
    for t0 in T0_GRID_MN_PER_M:
        for multiplier in TRANSFER_MULTIPLIERS:
            losses = []
            for sweeps in grouped.values():
                losses.append(np.mean([
                    abs(np.log1p(float(np.max(predict(row, t0, multiplier, physics))))
                        - np.log1p(observed_peak(row)))
                    for row in sweeps
                ]))
            candidates.append((float(np.mean(losses)), t0, multiplier))
    loss, t0, multiplier = min(candidates)
    return {"train_cell_balanced_log1p_peak_loss": loss,
            "selected_shared_T0_mN_per_m": t0,
            "selected_ETH_transfer_multiplier": multiplier,
            "selected_alpha_mN_per_m_per_nN":
                physics["eth_alpha_force_mN_per_m_per_nN"] * multiplier}


def main() -> None:
    physics = physical_parameters()
    records = assemble_records()
    fits, scores = [], []
    for channel in ("mp1", "mp2"):
        channel_records = [r for r in records if r["channel"] == channel]
        for date in sorted({r["date"] for r in channel_records}):
            training = [r for r in channel_records if r["date"] != date]
            test = [r for r in channel_records if r["date"] == date]
            selected = select_shared(training, physics)
            fit_record = {"channel": channel, "heldout_date": date,
                          "train_cells": len({r["cell_id"] for r in training}),
                          "test_cells": len({r["cell_id"] for r in test}),
                          **selected}
            fits.append(fit_record)
            for row in test:
                waveform = predict(row, selected["selected_shared_T0_mN_per_m"],
                                   selected["selected_ETH_transfer_multiplier"], physics)
                scores.append({"cell_id": row["cell_id"], "channel": channel,
                               "heldout_date": date, "sweep": row["sweep"],
                               "model": "ETH_eLife_static_sigmoid",
                               "mechanistic_prior_validity": "Piezo1" if channel == "mp1" else "Piezo1_curve_transfer_only",
                               "shared_T0_mN_per_m": selected["selected_shared_T0_mN_per_m"],
                               "shared_alpha_mN_per_m_per_nN": selected["selected_alpha_mN_per_m_per_nN"],
                               "calibration_force_threshold_nN": row["threshold"],
                               "calibration_force_peak_nN": row["cal_force_peak"],
                               **metrics(row, waveform),
                               **postpeak_score(row["target"], waveform,
                                                max(10.0, 5.0*row["noise"]), row["contact"])})
            print("fold",channel,date,"train cells",fit_record["train_cells"],
                  "test cells",fit_record["test_cells"],
                  "T0",selected["selected_shared_T0_mN_per_m"],
                  "lambda multiplier",selected["selected_ETH_transfer_multiplier"],flush=True)
    for name, rows in (("mechanistic_tension_simulator_fits.csv", fits),
                       ("mechanistic_tension_simulator_scores.csv", scores)):
        with (HERE / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (HERE / "mechanistic_tension_simulator_physics.json").write_text(
        json.dumps({"physics": physics, "T0_grid": T0_GRID_MN_PER_M,
                    "transfer_multipliers": TRANSFER_MULTIPLIERS,
                    "notes": "T0 is a shared effective offset; G and V-Erev are not identified by one voltage."},
                   indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("saved",len(scores),"future sweep scores",len(fits),"date folds")


if __name__ == "__main__":
    main()
