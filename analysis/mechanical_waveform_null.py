"""Mechanically matched force-proportional null for signed WT Piezo waveforms.

The null is retrospective and generous: it is given each sweep's observed peak
amplitude and force-to-current peak delay. It tests shape beyond a simple AFM
force proxy, not prospective current prediction or molecular inactivation.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

from adaptation_nested import force_threshold
from dynamic_input_screen import HERE, ROOT, load_traces
from one_shot_work import first_calibration, read_metadata


LEFT, RIGHT = 60, 140
REPEATS = 200
SEED = 20260926


def pca_stats(matrix: np.ndarray, cells: list[str]) -> tuple[float, int, float]:
    weights = np.array([1.0/Counter(cells)[cell] for cell in cells])
    weights /= weights.sum()
    center = np.average(matrix, axis=0, weights=weights)
    x = (matrix-center)*np.sqrt(weights[:, None])
    eigenvalues = np.linalg.eigvalsh(x@x.T)[::-1]
    eigenvalues = np.maximum(eigenvalues, 0)
    variance = eigenvalues/eigenvalues.sum()
    return float(variance[0]), int(np.searchsorted(np.cumsum(variance), .9)+1), float(variance[:2].sum())


def sample_noise(precontact: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    # Three-ms circular blocks retain short-range noise correlation measured
    # in the same sweep. The 10-ms reference limits longer-noise modeling.
    clean = precontact-np.mean(precontact)
    count = (length+2)//3
    starts = rng.integers(0, len(clean), size=count)
    return clean[(starts[:, None]+np.arange(3)) % len(clean)].reshape(-1)[:length]


def translate(values: np.ndarray, lag: int) -> np.ndarray:
    result = np.zeros_like(values)
    if lag >= 0:
        if lag < len(values):
            result[lag:] = values[:len(values)-lag]
    else:
        if -lag < len(values):
            result[:lag] = values[-lag:]
    return result


def main() -> None:
    with (HERE / "signed_quality_audit_sweeps.csv").open(newline="", encoding="utf-8") as handle:
        audit = [row for row in csv.DictReader(handle)
                 if row["geometry_qc"] == "1" and abs(float(row["voltage_mV"])+80) <= 5]
    chosen = {(row["cell_id"], int(row["sweep"])) for row in audit}
    by_cell = read_metadata()
    calibration = {cell: first_calibration(rows) for cell, rows in by_cell.items()}
    traces, _ = load_traces(by_cell, calibration)
    with (ROOT / "bychannel_summary.csv").open(newline="", encoding="utf-8-sig") as handle:
        selected = list(csv.DictReader(handle))
    paths = {path.name: path for path in ROOT.rglob("*_preprocessed.feather")}
    records: list[dict] = []
    for item in selected:
        cell = item["uniqueID"]
        sweeps = sorted(sweep for name, sweep in chosen if name == cell)
        if not sweeps:
            continue
        path = paths[Path(item["path"]).name]
        table = feather.read_table(path, columns=["sweep", "ti", "i_blsub"])
        data = {key: table[key].to_numpy() for key in table.column_names}
        cal_trace = traces[cell, int(calibration[cell]["sweep"])]
        threshold = force_threshold(calibration[cell], cal_trace)
        for sweep in sweeps:
            row = next(row for row in by_cell[cell] if int(row["sweep"]) == sweep)
            start = float(row["compression_end_ms"])-150
            mask = ((data["sweep"] == sweep) & (data["ti"] >= start-1e-6) &
                    (data["ti"] <= start+350+1e-6))
            if np.count_nonzero(mask) != 8751:
                raise ValueError((cell, sweep))
            signal = -data["i_blsub"][mask][:8750].reshape(350, 25).mean(axis=1)
            signal -= np.median(signal[:10])
            trace = traces[cell, sweep]
            force = trace["F"]
            f_peak_index = int(np.argmax(force[:150]))
            peak_start = max(int(trace["contact_ms"])-5, 0)
            i_peak_index = peak_start+int(np.argmax(signal[peak_start:]))
            amplitude = float(signal[i_peak_index])
            drive = np.maximum(force-threshold, 0)
            if np.max(drive) <= 0:
                raise ValueError((cell, sweep, "no mechanical drive"))
            drive /= np.max(drive)
            shifted = translate(drive, i_peak_index-f_peak_index)
            real_segment = signal[f_peak_index-LEFT:f_peak_index+RIGHT+1]/amplitude
            if len(real_segment) != LEFT+RIGHT+1:
                raise ValueError((cell, sweep, "short segment"))
            records.append({
                "cell_id": cell, "channel": calibration[cell]["channel"],
                "date": calibration[cell]["date"], "sweep": sweep,
                "amplitude_pA": amplitude, "delay_ms": i_peak_index-f_peak_index,
                "noise_pA": float(np.std(signal[:10])),
                "force_20ms_retention": float(force[min(f_peak_index+20, 349)]/max(force[f_peak_index], 1e-9)),
                "real_20ms_retention": float(signal[min(i_peak_index+20, 349)]/amplitude),
                "real_segment": real_segment, "drive": shifted,
                "precontact": signal[:10].copy(), "force_peak_index": f_peak_index,
                "current_peak_index": i_peak_index,
            })
    rng = np.random.default_rng(SEED)
    summaries = []
    null_rows = []
    sweep_rows = []
    for channel in ("mp1", "mp2"):
        group = [row for row in records if row["channel"] == channel]
        cells = [row["cell_id"] for row in group]
        real = np.stack([row["real_segment"] for row in group])
        observed_pc1, observed_dim90, observed_pc12 = pca_stats(real, cells)
        observed_retention = float(np.median([row["real_20ms_retention"] for row in group]))
        null_stats = []
        for repeat in range(REPEATS):
            shapes, retentions, delay_errors = [], [], []
            for row in group:
                noisy = row["amplitude_pA"]*row["drive"] + sample_noise(row["precontact"], 350, rng)
                peak_start = max(row["current_peak_index"]-40, 0)
                realized_peak = peak_start+int(np.argmax(noisy[peak_start:]))
                delay_errors.append(realized_peak-row["current_peak_index"])
                norm = float(noisy[realized_peak])
                shapes.append(noisy[row["force_peak_index"]-LEFT:row["force_peak_index"]+RIGHT+1]/norm)
                retentions.append(float(noisy[min(realized_peak+20, 349)]/norm))
            pc1, dim90, pc12 = pca_stats(np.stack(shapes), cells)
            null_stats.append((pc1, dim90, pc12, float(np.median(retentions)),
                               float(np.median(np.abs(delay_errors)))))
            null_rows.append({"channel": channel, "repeat": repeat, "pc1": pc1,
                              "dimension_90pct": dim90, "pc1_pc2": pc12,
                              "median_postpeak_20ms_retention": null_stats[-1][3],
                              "median_abs_peak_delay_error_ms": null_stats[-1][4]})
        values = np.array(null_stats)
        for row in group:
            index = row["current_peak_index"]
            expected = float(row["drive"][min(index+20, 349)]/max(row["drive"][index], 1e-9))
            sweep_rows.append({"cell_id": row["cell_id"], "channel": channel,
                               "date": row["date"], "sweep": row["sweep"],
                               "real_postpeak_20ms_retention": row["real_20ms_retention"],
                               "noise_free_null_postpeak_20ms_retention": expected,
                               "real_minus_null": row["real_20ms_retention"]-expected,
                               "force_postpeak_20ms_retention": row["force_20ms_retention"],
                               "peak_delay_ms": row["delay_ms"],
                               "peak_snr": row["amplitude_pA"]/max(row["noise_pA"], 1e-9)})
        per_cell = np.array([np.mean([r["real_minus_null"] for r in sweep_rows
                                      if r["channel"] == channel and r["cell_id"] == cell])
                             for cell in sorted(set(cells))])
        boot = np.median(rng.choice(per_cell, size=(5000, len(per_cell)), replace=True), axis=1)
        print(channel, "cell-median real-minus-noise-free-null 20ms retention",
              round(float(np.median(per_cell)), 3), "cell bootstrap 95%",
              np.quantile(boot, [.025, .975]).round(3),
              "negative cells", int(np.sum(per_cell < 0)), "/", len(per_cell))
        stats = [("pc1", observed_pc1, 0), ("dimension_90pct", observed_dim90, 1),
                 ("pc1_pc2", observed_pc12, 2),
                 ("median_postpeak_20ms_retention", observed_retention, 3)]
        for metric, observed, col in stats:
            summaries.append({"channel": channel, "metric": metric,
                              "cells": len(set(cells)), "sweeps": len(group),
                              "dates": len(set(row["date"] for row in group)),
                              "observed": observed,
                              "null_median": float(np.median(values[:, col])),
                              "null_q025": float(np.quantile(values[:, col], .025)),
                              "null_q975": float(np.quantile(values[:, col], .975)),
                              "empirical_lower_tail": float((1+np.sum(values[:, col] <= observed))/(REPEATS+1)),
                              "empirical_upper_tail": float((1+np.sum(values[:, col] >= observed))/(REPEATS+1))})
        print(channel, "cells", len(set(cells)), "sweeps", len(group),
              "force 20ms retention", round(float(np.median([r["force_20ms_retention"] for r in group])), 3),
              "null median abs peak jitter", round(float(np.median(values[:, 4])), 2))
        for row in summaries[-4:]:
            print(row)
    for name, rows in (("mechanical_waveform_null_summary.csv", summaries),
                       ("mechanical_waveform_null_repeats.csv", null_rows),
                       ("mechanical_waveform_null_sweeps.csv", sweep_rows)):
        with (HERE / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
