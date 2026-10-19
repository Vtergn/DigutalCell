"""Can one calibration waveform identify a cell-specific tension offset?

Uses the exact signed-current cohort and whole-date shared fits from
mechanistic_tension_simulator.py. The test cell's calibration current is used
only to choose its own effective T0. Future current is used only for scoring.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

from adaptation_nested import postpeak_score
from causal_cnn_benchmark import inward, metrics, window
from dynamic_input_screen import HERE, ROOT
from mechanistic_tension_simulator import (assemble_records, gate, physical_parameters,
                                           predict)
from one_shot_work import first_calibration, read_metadata


T0_CELL_GRID = np.arange(0, 2.5001, .25)
LAG_MS = {"mp1": 8, "mp2": 1}  # from Young et al.; fixed before scoring


def lagged(force: np.ndarray, lag_ms: int) -> np.ndarray:
    if lag_ms == 0:
        return force.copy()
    return np.r_[np.repeat(force[0], lag_ms), force[:-lag_ms]]


def calibration_currents(cells: set[str]) -> dict[str, np.ndarray]:
    by_cell = read_metadata()
    calibration = {cell: first_calibration(rows) for cell, rows in by_cell.items()}
    with (ROOT / "bychannel_summary.csv").open(newline="", encoding="utf-8-sig") as handle:
        selected = {r["uniqueID"]: r for r in csv.DictReader(handle)}
    paths = {p.name: p for p in ROOT.rglob("*_preprocessed.feather")}
    currents = {}
    for cell in cells:
        meta = calibration[cell]
        path = paths[Path(selected[cell]["path"]).name]
        table = feather.read_table(path, columns=["sweep", "ti", "i_blsub"])
        data = {key: table[key].to_numpy() for key in table.column_names}
        currents[cell] = inward(data, window(data, int(meta["sweep"]),
                                              float(meta["compression_end_ms"])))
    return currents


def fit_t0(row: dict, current: np.ndarray, multiplier: float, physics: dict) -> dict:
    lag = LAG_MS[row["channel"]]
    shifted_force = lagged(row["cal_force"], lag)
    alpha = physics["eth_alpha_force_mN_per_m_per_nN"] * multiplier
    t50 = physics["elife_piezo1_T50_mN_per_m"]
    k = physics["elife_piezo1_k_mN_per_m"]
    start = max(int(row["cal_contact"])-5, 0)
    peak = start + int(np.argmax(current[start:]))
    end = min(peak+1, 170)
    if end-start < 10:
        raise ValueError("Calibration rise window too short")
    target = current[start:end] / row["cal_peak"]
    profile = []
    for t0 in T0_CELL_GRID:
        shape = gate(shifted_force, row["threshold"], t0, alpha, t50, k)
        scale = float(np.max(shape[start:end]))
        if scale < 1e-12:
            loss = float("inf")
        else:
            loss = float(np.mean(np.abs(target - shape[start:end] / scale)))
        profile.append((loss, float(t0)))
    best_loss, best_t0 = min(profile)
    finite = [loss for loss, _ in profile if np.isfinite(loss)]
    near = [t0 for loss, t0 in profile if loss <= best_loss + .02]
    return {"cell_T0_mN_per_m": best_t0, "calibration_rise_shape_mae": best_loss,
            "calibration_rise_start_ms": start, "calibration_rise_end_ms": end,
            "T0_within_0p02_loss_min": min(near), "T0_within_0p02_loss_max": max(near),
            "calibration_loss_range": max(finite)-min(finite)}


def main() -> None:
    physics = physical_parameters()
    records = assemble_records()
    currents = calibration_currents({r["cell_id"] for r in records})
    with (HERE / "mechanistic_tension_simulator_fits.csv").open(newline="", encoding="utf-8") as handle:
        shared = {(r["channel"], r["heldout_date"]): r for r in csv.DictReader(handle)}
    cell_fits, scores = [], []
    cache = {}
    for row in records:
        params = shared[(row["channel"], row["date"])]
        multiplier = float(params["selected_ETH_transfer_multiplier"])
        common_t0 = float(params["selected_shared_T0_mN_per_m"])
        cell = row["cell_id"]
        if cell not in cache:
            fit = fit_t0(row, currents[cell], multiplier, physics)
            cache[cell] = fit
            cell_fits.append({"cell_id": cell, "channel": row["channel"],
                              "heldout_date": row["date"], "lag_ms": LAG_MS[row["channel"]],
                              "shared_T0_mN_per_m": common_t0,
                              "ETH_multiplier": multiplier, **fit})
        lag_row = {**row, "force": lagged(row["force"], LAG_MS[row["channel"]])}
        for model, t0 in (("lagged_threshold_force", float("nan")),
                          ("shared_T0_with_fixed_lag", common_t0),
                          ("one_shot_shape_T0_with_fixed_lag", cache[cell]["cell_T0_mN_per_m"])):
            waveform = (lagged(row["force_baseline"], LAG_MS[row["channel"]])
                        if model == "lagged_threshold_force" else
                        predict(lag_row, t0, multiplier, physics))
            scores.append({"cell_id": cell, "channel": row["channel"],
                           "heldout_date": row["date"], "sweep": row["sweep"],
                           "model": model, "effective_T0_mN_per_m": t0,
                           "lag_ms": LAG_MS[row["channel"]], **metrics(row, waveform),
                           **postpeak_score(row["target"], waveform,
                                            max(10.0, 5.0*row["noise"]), row["contact"])})
    for name, rows in (("mechanistic_tension_cell_fits.csv", cell_fits),
                       ("mechanistic_tension_personalization_scores.csv", scores)):
        with (HERE / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print("cells",len(cell_fits),"future scores",len(scores))
    for channel in ("mp1", "mp2"):
        group = [x for x in cell_fits if x["channel"] == channel]
        print(channel,"selected T0",[x["cell_T0_mN_per_m"] for x in group],
              "median near-optimal width",np.median([x["T0_within_0p02_loss_max"]-
                                                       x["T0_within_0p02_loss_min"] for x in group]))


if __name__ == "__main__":
    main()
