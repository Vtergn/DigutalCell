"""Audit model outputs against published biomarker windows and local data.

The human windows are literature summaries, not patient-level validation.
Rabbit data are descriptive and are never pooled with human model outputs.
"""
from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "Data"

# BPSLand 2022 Table 4, in-vitro range column, 1-Hz human ventricular.
HUMAN_WINDOWS = {
    "APD90_ms": [180.0, 440.0],
    "active_tension_peak_kPa": [15.0, 25.0],
    "active_tension_time_to_peak_ms": [147.0, 172.0],
}


def _crossing_after_peak(t: np.ndarray, y: np.ndarray, fraction: float) -> float | None:
    peak_i = int(np.argmax(y))
    baseline = float(y[0])
    threshold = baseline + fraction * (float(y[peak_i]) - baseline)
    hits = np.flatnonzero(y[peak_i:] <= threshold)
    return None if not hits.size else float(t[peak_i + hits[0]] - t[peak_i])


def beat_metrics(code: str) -> dict[str, float | None]:
    d = np.load(HERE / f"full_beat_validation_lambda{code}.npz")
    t = d["time"]
    v = d["predicted_voltage"]
    ca = d["predicted_cai"]
    tension = d["predicted_active_tension"]
    v_peak_i = int(np.argmax(v))
    tension_peak_i = int(np.argmax(tension))
    return {
        "resting_voltage_mV": float(v[0]),
        "voltage_peak_mV": float(v[v_peak_i]),
        "APD90_ms": _crossing_after_peak(t, v, 0.1),
        "diastolic_cai_uM": float(ca[0]),
        "peak_cai_uM": float(np.max(ca)),
        "active_tension_peak_kPa": float(tension[tension_peak_i]),
        "active_tension_time_to_peak_ms": float(t[tension_peak_i] - t[v_peak_i]),
    }


def archive_peak(keyword: str) -> float:
    archive = DATA / "doi_10_25338_B8T935__v20220607" / "Experimental_Data.zip"
    with zipfile.ZipFile(archive) as z:
        name = next(
            n for n in z.namelist()
            if keyword in n and "FURA2" in n and n.endswith("_smoothed.csv")
        )
        with io.TextIOWrapper(z.open(name), encoding="utf-8-sig", newline="") as f:
            rows = csv.DictReader(f)
            values = [float(row["Ca2+ Fura-2 ratio"]) for row in rows]
    return max(values)


def main() -> None:
    baseline = beat_metrics("100")
    stretched = beat_metrics("110")
    checks = {
        key: {
            "value": baseline[key],
            "reference_window": bounds,
            "inside": bounds[0] <= baseline[key] <= bounds[1],
        }
        for key, bounds in HUMAN_WINDOWS.items()
    }
    rabbit_free = archive_peak("Load-Free")
    rabbit_gel = archive_peak("In-Gel")
    report = {
        "scope": "Descriptive external biomarker check; no individual experimental validation",
        "baseline_lambda_1_00": baseline,
        "sustained_lambda_1_10": stretched,
        "human_in_vitro_window_checks_at_baseline_only": checks,
        "model_stretch_tension_peak_ratio_1_10_over_1_00": (
            stretched["active_tension_peak_kPa"] / baseline["active_tension_peak_kPa"]
        ),
        "model_stretch_cai_peak_ratio_1_10_over_1_00": (
            stretched["peak_cai_uM"] / baseline["peak_cai_uM"]
        ),
        "human_hpsc_Ballan_ratio_at_40_percent_stretch": 1.24 / 0.51,
        "rabbit_example_fura2_peak_load_free": rabbit_free,
        "rabbit_example_fura2_peak_in_gel": rabbit_gel,
        "rabbit_example_fura2_peak_ratio_gel_over_free": rabbit_gel / rabbit_free,
        "sources": {
            "human_biomarker_windows": "https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2022.906146/full#T4",
            "human_hpsc_stretch": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7486198/",
            "rabbit_archive": "https://datadryad.org/dataset/doi:10.25338/B8T935",
        },
        "interpretation": [
            "The human windows apply to resting 1-Hz adult ventricular conditions, not to sustained stretch.",
            "Ballan uses hPSC-CM and 40% stretch; the model ratio is at 10% and is not a matched effect-size comparison.",
            "Rabbit Fura-2 is a fluorescence ratio, not absolute intracellular calcium; specimens and loading differ.",
        ],
    }
    output = HERE / "external_physiology_check.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
