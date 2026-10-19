"""Read-only audit of the downloaded rabbit load-free/in-gel example traces.

The archive supplies one trace per signal and condition, with no cell IDs or
replicate table. Metrics are descriptive and are not paired-cell statistics.
"""
from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "Data/doi_10_25338_B8T935__v20220607/Experimental_Data.zip"
OUT = ROOT / "analysis/mct_feedback_audit"


def read_trace(archive: zipfile.ZipFile, condition: str, signal: str, smooth: bool):
    suffix = "smoothed" if smooth else "raw"
    path = next(name for name in archive.namelist() if condition in name
                and ("FURA2/" if signal == "Ca" else "Sarcomere Length/") in name
                and name.endswith(f"_{suffix}.csv"))
    rows = list(csv.reader(io.StringIO(archive.read(path).decode("utf-8-sig"))))
    return np.asarray(rows[1:], dtype=float), rows[0], path


def peaks_and_amplitudes(values: np.ndarray, is_length: bool):
    # One stimulation cycle is about one second in these example records.
    candidates, _ = find_peaks(-values if is_length else values,
                               distance=175, prominence=.04 if is_length else .2)
    # Ignore the first/last incomplete cycle when summarizing amplitudes.
    centers = candidates[(candidates >= 250) & (candidates <= len(values)-125)]
    return centers.tolist(), values[centers].tolist()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    data = {}
    with zipfile.ZipFile(ZIP) as archive:
        for condition in ["Load_free Data", "In_Gel Data"]:
            for signal in ["Ca", "SL"]:
                smooth, header, path = read_trace(archive, condition, signal, True)
                raw, raw_header, _ = read_trace(archive, condition, signal, False)
                if not np.array_equal(smooth[:, 0], raw[:, 0]):
                    raise ValueError(f"Raw/smoothed time mismatch: {path}")
                peaks, values = peaks_and_amplitudes(smooth[:, 1], signal == "SL")
                data[(condition, signal)] = (smooth, raw)
                data[f"{condition}/{signal}"] = {
                    "archive_path": path, "header": header,
                    "n_samples": len(smooth), "sampling_ms": 1000*np.median(np.diff(smooth[:,0])),
                    "window_s": [float(smooth[0,0]), float(smooth[-1,0])],
                    "smoothed_min": float(smooth[:,1].min()),
                    "smoothed_max": float(smooth[:,1].max()),
                    "interior_peak_times_s": smooth[peaks,0].tolist(),
                    "interior_extrema_values": values,
                }
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, constrained_layout=True)
    for condition, color in [("Load_free Data", "#4e79a7"), ("In_Gel Data", "#e15759")]:
        for ax, signal in zip(axes, ["Ca", "SL"]):
            smoothed, raw = data[(condition, signal)]
            ax.plot(raw[:,0], raw[:,1], color=color, alpha=.18, lw=.55)
            ax.plot(smoothed[:,0], smoothed[:,1], color=color, lw=1.5,
                    label="Load free" if "Load_free" in condition else "In gel")
    axes[0].set_ylabel("Fura-2 ratio (relative Ca signal)")
    axes[1].set_ylabel("Sarcomere length (µm)")
    axes[1].set_xlabel("Time (s)")
    axes[0].legend(frameon=False)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle("Published rabbit cardiomyocyte example traces: load free and in gel")
    fig.savefig(OUT / "rabbit_load_feedback_traces.png", dpi=170)
    plt.close(fig)
    report = {k: v for k,v in data.items() if isinstance(k,str)}
    free_ca = data[("Load_free Data", "Ca")][0][:,1]
    gel_ca = data[("In_Gel Data", "Ca")][0][:,1]
    free_sl = data[("Load_free Data", "SL")][0][:,1]
    gel_sl = data[("In_Gel Data", "SL")][0][:,1]
    report["descriptive_condition_contrast"] = {
        "Fura2_peak_load_free_vs_in_gel": [float(free_ca.max()), float(gel_ca.max())],
        "Fura2_peak_direction": "higher in gel example",
        "fractional_SL_excursion_load_free_vs_in_gel": [
            float(np.ptp(free_sl)/free_sl.max()), float(np.ptp(gel_sl)/gel_sl.max())],
        "SL_excursion_direction": "smaller in gel example",
    }
    report["interpretation_boundary"] = (
        "One example trace per signal and condition; no cell IDs, replicate counts, "
        "or known same-cell pairing. Fura-2 ratio is not absolute Ca concentration. "
        "No inferential or causal effect size is estimated from these CSVs.")
    (OUT / "archive_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
