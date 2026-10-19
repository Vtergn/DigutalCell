"""Feasibility audit for mechanical-event-aligned Piezo waveform geometry.

This is retrospective and descriptive. Shape normalization uses the observed
peak and must never be used to evaluate prospective waveform forecasts.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import spearmanr

from dynamic_input_screen import HERE, load_traces, onset_ms
from one_shot_work import first_calibration, read_metadata


LEFT_MS = 10
RIGHT_MS = 200
PEAK_LEFT_MS = 60
PEAK_RIGHT_MS = 140


def pca_equal_cell(waveforms: np.ndarray, cells: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = Counter(cells)
    weights = np.asarray([1.0/counts[cell] for cell in cells])
    weights /= np.sum(weights)
    center = np.average(waveforms, axis=0, weights=weights)
    centered = waveforms-center
    _, singular, vectors = np.linalg.svd(centered*np.sqrt(weights[:, None]), full_matrices=False)
    variance = singular**2/np.sum(singular**2)
    scores = centered@vectors.T
    return variance, scores, vectors


def cell_spearman(values: np.ndarray, feature: np.ndarray, cells: list[str]) -> float:
    unique = sorted(set(cells))
    x = np.array([np.mean(values[np.array(cells) == cell]) for cell in unique])
    y = np.array([np.mean(feature[np.array(cells) == cell]) for cell in unique])
    return float(spearmanr(x, y).statistic)


def date_eta_squared(values: np.ndarray, dates: list[str], cells: list[str]) -> float:
    unique_cells = sorted(set(cells))
    cell_means = np.array([np.mean(values[np.array(cells) == cell]) for cell in unique_cells])
    cell_dates = [dates[cells.index(cell)] for cell in unique_cells]
    overall = np.mean(cell_means)
    total = np.sum((cell_means-overall)**2)
    between = sum(np.sum(np.array(cell_dates) == date)*
                  (np.mean(cell_means[np.array(cell_dates) == date])-overall)**2
                  for date in set(cell_dates))
    return float(between/total) if total > 0 else np.nan


def date_permutation_p(values: np.ndarray, dates: list[str], cells: list[str]) -> float:
    """Shuffle date labels across cells, retaining the observed date counts."""
    unique_cells = sorted(set(cells))
    cell_means = np.array([np.mean(values[np.array(cells) == cell]) for cell in unique_cells])
    cell_dates = np.array([dates[cells.index(cell)] for cell in unique_cells])
    def eta(labels: np.ndarray) -> float:
        overall = np.mean(cell_means)
        total = np.sum((cell_means-overall)**2)
        between = sum(np.sum(labels == date)*(np.mean(cell_means[labels == date])-overall)**2
                      for date in set(labels))
        return float(between/total) if total > 0 else np.nan
    observed = eta(cell_dates)
    rng = np.random.default_rng(20260926)
    null = np.array([eta(rng.permutation(cell_dates)) for _ in range(5000)])
    return float((1+np.sum(null >= observed))/(len(null)+1))


def main(min_snr: float = 0.0, suffix: str = "") -> None:
    by_cell = read_metadata()
    calibration = {cell: first_calibration(rows) for cell, rows in by_cell.items()}
    traces, audits = load_traces(by_cell, calibration)
    audit_by_sweep = {(item["cell_id"], int(item["sweep"])): item for item in audits}
    observations = []
    for cell, cal in calibration.items():
        if cal is None or cal["channel"] not in {"mp1", "mp2"}:
            continue
        cal_voltage = traces[(cell, int(cal["sweep"]))]["voltage_mV"]
        for row in by_cell[cell]:
            sweep = int(row["sweep"])
            if sweep < int(cal["sweep"]):
                continue
            trace = traces[(cell, sweep)]
            if abs(trace["voltage_mV"]-cal_voltage) > 5:
                continue
            contact = int(trace["contact_ms"])
            current = trace["current"]
            threshold = max(10.0, 5.0*float(cal["stdev_pA"]))
            if not np.isfinite(onset_ms(current, threshold, contact-5)):
                continue
            peak_start = max(contact-5, 0)
            current_peak_time = peak_start+int(np.argmax(current[peak_start:]))
            current_peak = float(current[current_peak_time])
            audit = audit_by_sweep[(cell, sweep)]
            precontact_noise = max(float(audit["precontact_noise_pA"]), 1e-9)
            peak_snr = current_peak/precontact_noise
            if peak_snr < min_snr:
                continue
            force_peak_time = int(np.argmax(trace["F"][:150]))
            segment = current[contact-LEFT_MS:contact+RIGHT_MS+1]
            peak_segment = current[force_peak_time-PEAK_LEFT_MS:force_peak_time+PEAK_RIGHT_MS+1]
            if len(segment) != LEFT_MS+RIGHT_MS+1 or current_peak <= 0:
                raise ValueError(f"Unexpected event window for {cell}, {sweep}")
            if len(peak_segment) != PEAK_LEFT_MS+PEAK_RIGHT_MS+1:
                raise ValueError(f"Unexpected force-peak window for {cell}, {sweep}")
            observations.append({
                "cell_id": cell, "channel": cal["channel"], "date": cal["date"],
                "sweep": sweep, "is_calibration": int(sweep == int(cal["sweep"])),
                "peak_current_pA": current_peak,
                "precontact_noise_pA": precontact_noise,
                "precontact_baseline_pA": float(audit["precontact_baseline_pA"]),
                "peak_snr": peak_snr,
                "force_peak_nN": float(np.max(trace["F"][:150])),
                "work_peak_fJ": float(np.max(trace["W"][:150])),
                "force_postpeak_20ms_retention": float(trace["F"][min(force_peak_time+20, 349)]/
                                                        max(trace["F"][force_peak_time], 1e-9)),
                "peak_delay_ms": current_peak_time-force_peak_time,
                "postpeak_20ms_retention": float(current[min(current_peak_time+20, 349)]/current_peak),
                "waveform": segment,
                "force_peak_waveform": peak_segment,
            })
    rows, summary = [], []
    for channel in ("mp1", "mp2"):
        group = [item for item in observations if item["channel"] == channel]
        cells = [item["cell_id"] for item in group]
        dates = [item["date"] for item in group]
        for alignment, waveform_key in (("mechanical_contact", "waveform"),
                                        ("afm_force_peak", "force_peak_waveform")):
          raw = np.stack([item[waveform_key] for item in group])
          shape = raw/np.array([item["peak_current_pA"] for item in group])[:, None]
          for representation, matrix in (("raw_pA", raw), ("observed_peak_normalized", shape)):
            variance, scores, _ = pca_equal_cell(matrix, cells)
            cumulative = np.cumsum(variance)
            peak = np.asarray([item["peak_current_pA"] for item in group])
            largest_peak_cell = max(set(cells), key=lambda cell: np.mean(peak[np.array(cells) == cell]))
            others = np.array(cells) != largest_peak_cell
            variance_without_largest, _, _ = pca_equal_cell(matrix[others], list(np.array(cells)[others]))
            leave_one_cell_pc1 = []
            for omitted_cell in sorted(set(cells)):
                keep = np.array(cells) != omitted_cell
                variance_loo, _, _ = pca_equal_cell(matrix[keep], list(np.array(cells)[keep]))
                leave_one_cell_pc1.append((float(variance_loo[0]), omitted_cell))
            delay = np.asarray([item["peak_delay_ms"] for item in group])
            retention = np.asarray([item["postpeak_20ms_retention"] for item in group])
            force_peak = np.asarray([item["force_peak_nN"] for item in group])
            force_retention = np.asarray([item["force_postpeak_20ms_retention"] for item in group])
            summary.append({
                "channel": channel, "representation": representation,
                "alignment": alignment,
                "cells": len(set(cells)), "dates": len(set(dates)), "sweeps": len(group),
                "pc1_variance_fraction": float(variance[0]),
                "pc1_variance_without_largest_peak_cell": float(variance_without_largest[0]),
                "pc1_variance_loo_min": min(value for value, _ in leave_one_cell_pc1),
                "pc1_variance_loo_min_omitted_cell": min(leave_one_cell_pc1)[1],
                "pc1_variance_loo_max": max(value for value, _ in leave_one_cell_pc1),
                "pc1_pc2_variance_fraction": float(np.sum(variance[:2])),
                "components_for_90pct_variance": int(np.searchsorted(cumulative, .9)+1),
                "cell_spearman_pc1_peak": cell_spearman(scores[:, 0], peak, cells),
                "cell_spearman_pc1_delay": cell_spearman(scores[:, 0], delay, cells),
                "cell_spearman_pc1_retention": cell_spearman(scores[:, 0], retention, cells),
                "cell_spearman_pc1_force_peak": cell_spearman(scores[:, 0], force_peak, cells),
                "cell_spearman_pc1_force_retention": cell_spearman(scores[:, 0], force_retention, cells),
                "date_eta_squared_pc1": date_eta_squared(scores[:, 0], dates, cells),
                "date_eta_squared_pc2": date_eta_squared(scores[:, 1], dates, cells),
                "date_permutation_p_pc1": date_permutation_p(scores[:, 0], dates, cells),
                "date_permutation_p_pc2": date_permutation_p(scores[:, 1], dates, cells),
            })
            for item, coordinates in zip(group, scores):
                rows.append({k: v for k, v in item.items() if k not in {"waveform", "force_peak_waveform"}} |
                            {"representation": representation, "alignment": alignment,
                             "pc1": float(coordinates[0]),
                             "pc2": float(coordinates[1]), "pc3": float(coordinates[2])})
    for filename, content in ((f"waveform_geometry_pilot_scores{suffix}.csv", rows),
                              (f"waveform_geometry_pilot_summary{suffix}.csv", summary)):
        with (HERE / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(content[0]))
            writer.writeheader()
            writer.writerows(content)
    for item in summary:
        print("min_snr", min_snr, item)


if __name__ == "__main__":
    main()
    main(min_snr=5.0, suffix="_snr5")
