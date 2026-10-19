"""Paired ETH Fig. 2 source-data analysis plus local TIFF quality audit.

Published source data provide manually selected membrane ROI lifetimes. TIFF
channel numbers are not labelled with physical units in exported XML, so TIFF
values are used for quality review, not independent tension calibration.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openpyxl
from PIL import Image
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data"
IMAGES = DATA / "StepwiseFlipperTrImaging/StepwiseFlipperTrImaging/ExtractedLifetimes"
OUT = ROOT / "analysis"
LIFETIME_TO_TENSION = 1.325  # mN/m per ns, HFF-1 calibration in the paper


def source_rows(sheet_name: str) -> list[dict]:
    wb = openpyxl.load_workbook(DATA / "ETH_SourceData_Fig2.xlsx", read_only=True, data_only=True)
    try:
        rows = list(wb[sheet_name].values)
    finally:
        wb.close()
    return [dict(zip(rows[0], row)) for row in rows[1:] if row[0] is not None]


def mean_ci(values: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(values))
    half = float(t.ppf(.975, len(values) - 1) * np.std(values, ddof=1) / np.sqrt(len(values)))
    return mean - half, mean + half


def paired_analysis(rows: list[dict], dose_key: str, active_cells: list[int]) -> dict:
    cell_curves = []
    slopes = []
    endpoint_deltas = []
    for cell in active_cells:
        data = sorted((r for r in rows if r["Cell"] == cell), key=lambda r: r[dose_key])
        doses = np.array([r[dose_key] for r in data], dtype=float)
        lifetimes = np.array([r["Mean Flipper-TR Lifetime (ns)"] for r in data], dtype=float)
        if len(data) < 2:
            raise ValueError(f"Insufficient dose steps for cell {cell}")
        slope = float(np.polyfit(doses, lifetimes, 1)[0])
        delta = float(lifetimes[-1] - lifetimes[0])
        slopes.append(slope)
        endpoint_deltas.append(delta)
        cell_curves.append({"cell": cell, "dose": doses.tolist(),
                            "lifetime_ns": lifetimes.tolist(),
                            "slope_ns_per_dose": slope,
                            "endpoint_delta_ns": delta})
    slopes = np.array(slopes)
    endpoint_deltas = np.array(endpoint_deltas)
    return {
        "n_cells": len(active_cells),
        "cell_curves": cell_curves,
        "mean_cell_slope_ns_per_dose": float(slopes.mean()),
        "slope_95pct_t_ci_ns_per_dose": mean_ci(slopes),
        "positive_cell_slopes": int(np.sum(slopes > 0)),
        "mean_paired_endpoint_delta_ns": float(endpoint_deltas.mean()),
        "endpoint_delta_95pct_t_ci_ns": mean_ci(endpoint_deltas),
        "positive_endpoint_deltas": int(np.sum(endpoint_deltas > 0)),
        "mean_endpoint_delta_tension_mN_per_m": float(endpoint_deltas.mean() * LIFETIME_TO_TENSION),
        "mean_slope_tension_mN_per_m_per_dose": float(slopes.mean() * LIFETIME_TO_TENSION),
    }


def audit_tiffs() -> list[dict]:
    records = []
    for group in ("Indentation", "Aspiration"):
        folder = IMAGES / group
        for path in sorted(folder.glob("Cell*_t??_ch2.tif")):
            name, frame = path.stem.split("_t")
            frame = int(frame[:2])
            with Image.open(path) as image:
                channel2 = np.asarray(image)
            with Image.open(folder / f"{name}_t{frame:02d}_ch3.tif") as image:
                channel3 = np.asarray(image)
            records.append({
                "group": group, "cell": name, "frame": frame,
                "channel2_median": float(np.median(channel2)),
                "channel2_pixels_ge_1000_frac": float(np.mean(channel2 >= 1000)),
                "channel3_median_div_1000": float(np.median(channel3) / 1000),
            })
    with (OUT / "eth_stepwise_tiff_qc.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    return records


def plot(results: dict, qc: list[dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for col, key in enumerate(("indentation", "aspiration")):
        r = results[key]
        for curve in r["cell_curves"]:
            dose = np.array(curve["dose"])
            lifetime = np.array(curve["lifetime_ns"])
            axes[0, col].plot(dose, lifetime - lifetime[0], marker="o", alpha=.75,
                              label=f"cell {curve['cell']}")
        axes[0, col].axhline(0, color="gray", lw=.7)
        axes[0, col].set(xlabel="Force (nN)" if col == 0 else "Aspiration (mbar)",
                         ylabel="Lifetime change vs first step (ns)",
                         title=f"{key.capitalize()}: paired cells n={r['n_cells']}")
        axes[0, col].legend(fontsize=8)

        group = key.capitalize()
        for cell in sorted({item["cell"] for item in qc if item["group"] == group}):
            items = sorted((item for item in qc if item["group"] == group and item["cell"] == cell),
                           key=lambda item: item["frame"])
            axes[1, col].plot([item["frame"] for item in items],
                              [item["channel2_pixels_ge_1000_frac"] for item in items],
                              marker=".", label=cell)
        axes[1, col].set(xlabel="Exported frame index (not dose step)",
                         ylabel="Fraction of channel 2 pixels ≥ 1000",
                         title="TIFF signal availability across frames", ylim=(0, 1))
        axes[1, col].legend(fontsize=8)
    fig.savefig(OUT / "eth_stepwise_analysis.png", dpi=160)
    plt.close(fig)


def main() -> None:
    force_rows = source_rows("SourceData_Fig2f")
    aspiration_rows = source_rows("SourceData_Fig2g")
    results = {
        "source": "ETH Nature Methods 2024 Source Data Fig. 2",
        "conversion_mN_per_m_per_ns": LIFETIME_TO_TENSION,
        "indentation": paired_analysis(force_rows, "Force", [1, 2, 3, 4, 5]),
        "aspiration": paired_analysis(aspiration_rows, "Pressure", [1, 2, 3, 4]),
        "indentation_unpaired_zero_force_cells": [6, 7, 8, 9, 10],
    }
    qc = audit_tiffs()
    results["tiff_qc"] = {
        "frames": len(qc),
        "cells_by_group": {group: sorted({r["cell"] for r in qc if r["group"] == group})
                           for group in ("Indentation", "Aspiration")},
        "note": "Channel identities and physical units are not encoded in exported XML; TIFF review is QC only.",
    }
    (OUT / "eth_stepwise_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot(results, qc)
    for key in ("indentation", "aspiration"):
        r = results[key]
        print(key, "n", r["n_cells"], "endpoint Δτ ns", round(r["mean_paired_endpoint_delta_ns"], 3),
              "95% t CI", tuple(round(x, 3) for x in r["endpoint_delta_95pct_t_ci_ns"]),
              "slope", round(r["mean_cell_slope_ns_per_dose"], 6),
              "positive endpoints", r["positive_endpoint_deltas"])
    print("TIFF frames audited", len(qc))


if __name__ == "__main__":
    main()
