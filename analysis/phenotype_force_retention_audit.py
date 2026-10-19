"""Compare current persistence with observed global AFM-force persistence.

Uses only sweeps already admitted by the frozen signed-current waveform QC.
Global cantilever force is a proxy for local mechanical load, not membrane
tension. This audit does not identify an intrinsic inactivation mechanism.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

from dynamic_input_screen import ROOT, causal_mean
from one_shot_work import read_metadata


HERE = Path(__file__).resolve().parent
CHIM = HERE.parent / "Data" / "2022_pnas" / "2022_pnas" / "chimera_experiments"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    projections = [row for row in read_csv(HERE / "frozen_wt_projection_scores.csv")
                   if row["group"] in {"WT_mp1", "WT_mp2", "chimA"}]
    wt_sweeps = {(cell, int(row["sweep"])): row for cell, rows in read_metadata().items()
                 for row in rows}
    wt_selected = {row["uniqueID"]: row for row in
                   read_csv(ROOT / "bychannel_summary.csv") if row["construct"] in {"mp1", "mp2"}}
    chim_selected = {row["uniqueID"]: row for row in
                     read_csv(CHIM / "capchims_summary.csv") if row["construct"] == "chimA"}
    paths = {path.name: path for folder in (ROOT, CHIM)
             for path in folder.rglob("*_preprocessed.feather")}
    by_cell = defaultdict(list)
    for row in projections:
        by_cell[row["cell_id"]].append(row)
    records = []
    for cell, rows in by_cell.items():
        is_chim = rows[0]["group"] == "chimA"
        selected = chim_selected[cell] if is_chim else wt_selected[cell]
        path = paths[Path(selected["path"]).name]
        table = feather.read_table(path, columns=["sweep", "ti", "force"])
        sweep_data = table["sweep"].to_numpy()
        time = table["ti"].to_numpy()
        force_data = table["force"].to_numpy()
        for row in rows:
            sweep = int(row["sweep"])
            start = (round(float(selected["tpeakf"]))-150 if is_chim else
                     float(wt_sweeps[cell, sweep]["compression_end_ms"])-150)
            mask = ((sweep_data == sweep) & (time >= start-1e-6) &
                    (time <= start+350+1e-6))
            if np.count_nonzero(mask) != 8751:
                raise ValueError(f"Unexpected force window: {cell}, sweep {sweep}")
            force = np.maximum(causal_mean(force_data[mask][:8750].reshape(350, 25).mean(axis=1)), 0)
            peak_force_time = int(np.argmax(force[:150]))
            current_peak_time = peak_force_time+int(float(row["peak_delay_ms"]))
            if not 0 <= current_peak_time < 330:
                raise ValueError(f"Current peak outside force window: {cell}")
            force_retention = float(force[current_peak_time+20]/
                                    max(force[current_peak_time], 1e-9))
            current_retention = float(row["postpeak_20ms_retention"])
            records.append({"cell_id": cell, "date": row["date"], "group": row["group"],
                            "sweep": sweep, "force_peak_nN": float(force[peak_force_time]),
                            "force20_from_current_peak_ratio": force_retention,
                            "current20_from_current_peak_ratio": current_retention,
                            "current_minus_force_retention": current_retention-force_retention})
    with (HERE / "phenotype_force_retention_sweeps.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summaries = {}
    for group in ("WT_mp1", "WT_mp2", "chimA"):
        group_rows = [row for row in records if row["group"] == group]
        cells = sorted({row["cell_id"] for row in group_rows})
        cell_values = {key: np.asarray([np.median([row[key] for row in group_rows
                                                   if row["cell_id"] == cell]) for cell in cells])
                       for key in ("force_peak_nN", "force20_from_current_peak_ratio",
                                   "current20_from_current_peak_ratio", "current_minus_force_retention")}
        summaries[group] = {"cells": len(cells), "sweeps": len(group_rows),
                            **{f"median_{key}": float(np.median(value))
                               for key, value in cell_values.items()},
                            "cells_current_faster_than_force": int(np.sum(
                                cell_values["current_minus_force_retention"] < 0))}
    (HERE / "phenotype_force_retention_results.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
