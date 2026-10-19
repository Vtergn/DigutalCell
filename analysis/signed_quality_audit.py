"""Audit signed current and recording metadata for the frozen WT Piezo cohort.

This is descriptive QC: no threshold is chosen by downstream PCA or model scores.
The original 350 ms window, 1 kHz binning, and mechanical contact are unchanged.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

from dynamic_input_screen import HERE, ROOT, load_traces, onset_ms
from one_shot_work import first_calibration, read_metadata


def number(row: dict, key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    by_cell = read_metadata()
    calibration = {cell: first_calibration(rows) for cell, rows in by_cell.items()}
    traces, audits = load_traces(by_cell, calibration)
    trace_audits = {(row["cell_id"], int(row["sweep"])): row for row in audits}
    with (ROOT / "bychannel_summary.csv").open(newline="", encoding="utf-8-sig") as handle:
        selected = list(csv.DictReader(handle))
    paths = {path.name: path for path in ROOT.rglob("*_preprocessed.feather")}
    results: list[dict] = []
    for chosen in selected:
        cell = chosen["uniqueID"]
        cal = calibration[cell]
        if cal is None or chosen["construct"] not in {"mp1", "mp2"}:
            continue
        path = paths[Path(chosen["path"]).name]
        summary_path = path.with_name(path.name.replace("_preprocessed.feather", "_summary.csv"))
        with summary_path.open(newline="", encoding="utf-8-sig") as handle:
            sweep_meta = {int(row["sweep"]): row for row in csv.DictReader(handle)}
        table = feather.read_table(path, columns=["sweep", "ti", "i", "i_blsub", "v"])
        data = {key: table[key].to_numpy() for key in table.column_names}
        cal_voltage = traces[cell, int(cal["sweep"])]["voltage_mV"]
        for row in by_cell[cell]:
            sweep = int(row["sweep"])
            if sweep < int(cal["sweep"]):
                continue
            trace = traces[cell, sweep]
            contact = int(trace["contact_ms"])
            start = float(row["compression_end_ms"]) - 150
            mask = ((data["sweep"] == sweep) &
                    (data["ti"] >= start-1e-6) &
                    (data["ti"] <= start+350+1e-6))
            if np.count_nonzero(mask) != 8751:
                raise ValueError(f"Unexpected window: {cell}, {sweep}")
            raw = data["i"][mask][:8750].reshape(350, 25).mean(axis=1)
            signed = data["i_blsub"][mask][:8750].reshape(350, 25).mean(axis=1)
            baseline = float(np.median(signed[:10]))
            signed -= baseline
            voltage = float(np.median(data["v"][mask]))
            signed_noise = float(np.std(signed[:10]))
            peak_start = max(contact-5, 0)
            negative_peak = float(-np.min(signed[peak_start:]))
            positive_peak = float(np.max(signed[peak_start:]))
            polarity = -1 if negative_peak >= positive_peak else 1
            oriented = polarity*signed
            peak_index = peak_start+int(np.argmax(oriented[peak_start:]))
            rectified = trace["current"]
            rect_peak = float(np.max(rectified[peak_start:]))
            old_noise = float(trace_audits[cell, sweep]["precontact_noise_pA"])
            threshold = max(10.0, 5.0*float(cal["stdev_pA"]))
            existing_event = np.isfinite(onset_ms(rectified, threshold, contact-5))
            meta = sweep_meta[sweep]
            results.append({
                "cell_id": cell, "date": cal["date"], "channel": cal["channel"],
                "sweep": sweep, "is_calibration": int(sweep == int(cal["sweep"])),
                "voltage_mV": voltage, "voltage_delta_from_cal_mV": voltage-cal_voltage,
                "rs_Mohm_reported": number(meta, "Rs"),
                "rs_comp_percent_reported": number(meta, "Rscomp"),
                "cm_pF_reported": number(meta, "Cm"),
                "seal_reported": number(meta, "seal"),
                "leak_pA_reported": number(meta, "leak"),
                "offset_pA_reported": number(meta, "offset"),
                "stdev_pA_reported": number(meta, "stdev"),
                "raw_current_precontact_pA": float(np.median(raw[:10])),
                "signed_blsub_precontact_pA": baseline,
                "signed_precontact_noise_pA": signed_noise,
                "rectified_precontact_noise_pA": old_noise,
                "signed_negative_peak_pA": negative_peak,
                "signed_positive_peak_pA": positive_peak,
                "dominant_polarity": polarity,
                "signed_peak_pA": float(oriented[peak_index]),
                "signed_peak_index_ms": peak_index,
                "rectified_peak_pA": rect_peak,
                "signed_peak_snr": float(oriented[peak_index]/max(signed_noise, 1e-9)),
                "rectified_peak_snr": float(rect_peak/max(old_noise, 1e-9)),
                "existing_event": int(existing_event),
            })
    by_key = {(r["cell_id"], r["sweep"]): r for r in results}
    for result in results:
        cal = by_key[result["cell_id"], int(calibration[result["cell_id"]]["sweep"])]
        result["calibration_polarity"] = cal["dominant_polarity"]
        result["polarity_disagrees_with_cal"] = int(result["dominant_polarity"] != cal["dominant_polarity"])
        result["rs_ratio_to_cal"] = result["rs_Mohm_reported"]/cal["rs_Mohm_reported"]
        result["signed_noise_ratio_to_cal"] = result["signed_precontact_noise_pA"]/max(cal["signed_precontact_noise_pA"], 1e-9)
        # These flags use only recording conditions and the precontact period.
        result["prospective_voltage_qc"] = int(abs(result["voltage_delta_from_cal_mV"]) <= 5)
        result["prospective_noise_qc"] = int(result["signed_noise_ratio_to_cal"] <= 5)
        result["prospective_rs_qc"] = int(np.isfinite(result["rs_ratio_to_cal"]) and
                                         0.5 <= result["rs_ratio_to_cal"] <= 2)
        result["prospective_qc"] = int(result["prospective_voltage_qc"] and
                                       result["prospective_noise_qc"] and
                                       result["prospective_rs_qc"])
        # Response SNR is an additional retrospective requirement for geometry only.
        result["geometry_qc"] = int(result["prospective_qc"] and result["existing_event"] and
                                    result["signed_peak_snr"] >= 5 and
                                    result["dominant_polarity"] == cal["dominant_polarity"])
    output = HERE / "signed_quality_audit_sweeps.csv"
    write_csv(output, results)
    for channel in ("mp1", "mp2"):
        group = [r for r in results if r["channel"] == channel and r["existing_event"] and
                 r["prospective_voltage_qc"]]
        print(channel, "existing event sweeps", len(group), "cells", len(set(r["cell_id"] for r in group)))
        print("voltage", Counter(round(r["voltage_mV"]/20)*20 for r in group))
        print("polarity", Counter(r["dominant_polarity"] for r in group))
        print("QC exclusions", Counter(k for r in group for k in
              ("noise" if not r["prospective_noise_qc"] else "",
               "rs" if not r["prospective_rs_qc"] else "",
               "polarity" if r["polarity_disagrees_with_cal"] else "",
               "snr" if r["signed_peak_snr"] < 5 else "") if k))
        print("geometry QC", sum(r["geometry_qc"] for r in group), "cells",
              len(set(r["cell_id"] for r in group if r["geometry_qc"])))
        for key in ("signed_precontact_noise_pA", "rs_ratio_to_cal", "signed_noise_ratio_to_cal",
                    "signed_peak_snr", "signed_blsub_precontact_pA"):
            values = np.array([r[key] for r in group])
            print(key, np.quantile(values, [0, .5, .9, 1]).round(3))
    print("wrote", output)


if __name__ == "__main__":
    main()
