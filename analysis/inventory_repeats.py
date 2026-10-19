"""Inventory all stimulus sweeps in the 100 selected bychannel cell files."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.feather as feather


ROOT = Path(__file__).resolve().parents[1] / "Data" / "2022_pnas" / "2022_pnas" / "bychannel"
OUT = Path(__file__).resolve().parent / "repeat_sweeps.csv"


def main() -> None:
    with (ROOT / "bychannel_summary.csv").open(newline="", encoding="utf-8-sig") as f:
        summary = list(csv.DictReader(f))
    paths = {p.name: p for p in ROOT.rglob("*_preprocessed.feather")}
    rows = []
    for meta in summary:
        path = paths[Path(meta["path"]).name]
        table = feather.read_table(path, columns=["sweep", "ti", "force", "work", "i_blsub", "v"])
        data = {col: table[col].to_numpy() for col in table.column_names}
        start = round((float(meta["tpeakf"]) - 150) / 250) * 250
        for sweep in np.unique(data["sweep"]):
            full = (data["sweep"] == sweep) & (data["ti"] >= start - 1e-6) & (data["ti"] <= start + 350 + 1e-6)
            compression = full & (data["ti"] <= start + 150 + 1e-6)
            if np.count_nonzero(full) != 8751 or np.count_nonzero(compression) != 3751:
                raise ValueError(f"Unexpected sweep window for {meta['uniqueID']}, sweep {sweep}")
            abs_current = np.abs(data["i_blsub"][full])
            baseline = float(np.median(abs_current[:100]))
            delta = abs_current - baseline
            force = data["force"][full]
            work = data["work"][full]
            voltage = data["v"][full]
            rows.append({
                "cell_id": meta["uniqueID"], "channel": meta["construct"], "date": meta["date"],
                "sweep": int(sweep), "selected_sweep": int(sweep) == int(meta["sweep"]),
                "window_start_ms": start, "peak_force_nN": float(np.max(data["force"][compression])),
                "peak_work_fJ": float(np.max(data["work"][compression])),
                "current_baseline_pA": baseline,
                "peak_delta_abs_current_pA": float(np.max(delta)),
                "current_noise_pA": float(np.std(delta[:500])),
                "voltage_median_mV": float(np.median(voltage)),
                "voltage_span_p01_p99_mV": float(np.quantile(voltage, .99) - np.quantile(voltage, .01)),
            })
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    by_cell = defaultdict(list)
    for row in rows:
        by_cell[row["cell_id"]].append(row)
    for channel in sorted({r["channel"] for r in rows}):
        cells = [v for v in by_cell.values() if v[0]["channel"] == channel]
        counts = [len(v) for v in cells]
        varying_force = sum(np.max([x["peak_force_nN"] for x in v]) > 1.5 * max(np.min([x["peak_force_nN"] for x in v]), 1) for v in cells if len(v) >= 2)
        print({
            "channel": channel, "cells": len(cells), "sweeps": sum(counts),
            "cells_with_2plus_sweeps": sum(n >= 2 for n in counts),
            "median_sweeps_per_cell": float(np.median(counts)),
            "force_varies_1p5x_cells": varying_force,
            "voltage_span_over_5mV_sweeps": sum(x["voltage_span_p01_p99_mV"] > 5 for v in cells for x in v),
        })
    print("sweep_count_distribution", dict(sorted(Counter(len(v) for v in by_cell.values()).items())))


if __name__ == "__main__":
    main()
