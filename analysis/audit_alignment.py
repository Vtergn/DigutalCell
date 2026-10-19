"""Cross-check compression CSV against preprocessed Feather for four channels."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.feather as feather


ROOT = Path(__file__).resolve().parents[1] / "Data" / "2022_pnas" / "2022_pnas" / "bychannel"


def main() -> None:
    summary = {}
    with (ROOT / "bychannel_summary.csv").open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            summary.setdefault(row["construct"], row)
    approaches = defaultdict(lambda: defaultdict(list))
    selected = {(row["uniqueID"], row["sweep"]) for row in summary.values()}
    with (ROOT / "bychannel_approaches.csv").open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            key = (row["uniqueID"], row["sweep"])
            if key in selected:
                for col in ("force", "absi_blsub", "absi_blsub_mech"):
                    approaches[key][col].append(float(row[col]))
    paths = {path.name: path for path in ROOT.rglob("*_preprocessed.feather")}
    for channel, row in summary.items():
        key = (row["uniqueID"], row["sweep"])
        path = paths[Path(row["path"]).name]
        table = feather.read_table(path, columns=["sweep", "ti", "force", "i_blsub", "v"])
        data = {col: table[col].to_numpy() for col in table.column_names}
        selected_sweep = data["sweep"] == int(row["sweep"])
        time = data["ti"][selected_sweep]
        force = data["force"][selected_sweep]
        absolute_current = np.abs(data["i_blsub"][selected_sweep])
        # Identify the compression slice by its first 10 paired force/current samples.
        target_force = np.asarray(approaches[key]["force"])
        target_current = np.asarray(approaches[key]["absi_blsub"])
        matches = np.flatnonzero(np.isclose(absolute_current, target_current[0], rtol=0, atol=1e-8))
        matches = [i for i in matches if i + len(target_force) <= len(time)]
        start = min(matches, key=lambda i: float(np.mean(np.abs(force[i:i+10] - target_force[:10]))))
        end = start + len(target_force)
        current_residual = target_current - absolute_current[start:end]
        mech_residual = np.asarray(approaches[key]["absi_blsub_mech"]) - (target_current - float(row["offset"]))
        full_window = (time >= time[start] - 1e-6) & (time <= time[start] + 350 + 1e-6)
        voltage = data["v"][selected_sweep][full_window]
        print({
            "channel": channel,
            "cell": row["uniqueID"],
            "sweep": row["sweep"],
            "compression_time_ms": [round(float(time[start]), 3), round(float(time[end - 1]), 3)],
            "peak_force_time_ms": float(row["tpeakf"]),
            "peak_current_time_ms": float(row["tpeaki"]),
            "max_force_difference_nN": round(float(np.max(np.abs(target_force - force[start:end]))), 9),
            "max_current_difference_pA": round(float(np.max(np.abs(current_residual))), 9),
            "max_mech_current_formula_difference_pA": round(float(np.max(np.abs(mech_residual))), 9),
            "full_window_voltage_p01_p99_mV": [
                round(float(np.quantile(voltage, 0.01)), 3),
                round(float(np.quantile(voltage, 0.99)), 3),
            ],
        })
        if channel == "mp1":
            raw_path = path.with_name(path.name.replace("_preprocessed.feather", ".asc"))
            raw_sweep = None
            raw_sample = None
            with raw_path.open(encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    if line.startswith("Sweep_"):
                        raw_sweep = int(line.split(",", 1)[0].rsplit("_", 1)[-1])
                    elif raw_sweep == int(row["sweep"]) and re.match(r"^\s*\d+\s*,", line):
                        fields = line.split(",")
                        if abs(float(fields[1]) - 0.5) < 1e-9:
                            raw_sample = (float(fields[2]) * 1e12, float(fields[4]) * 1e3)
                            break
            if raw_sample is None:
                raise ValueError("Expected raw 500 ms sample not found")
            raw_index = np.flatnonzero(selected_sweep & np.isclose(data["ti"], 500, atol=1e-8))
            if len(raw_index) != 1:
                raise ValueError("Expected Feather 500 ms sample not found")
            full = feather.read_table(path, columns=["i", "v"])
            feather_sample = (float(full["i"][raw_index[0]].as_py()), float(full["v"][raw_index[0]].as_py()))
            print({"raw_asc_vs_feather_500ms": {"raw_current_pA_voltage_mV": raw_sample, "feather_current_pA_voltage_mV": feather_sample}})


if __name__ == "__main__":
    main()
