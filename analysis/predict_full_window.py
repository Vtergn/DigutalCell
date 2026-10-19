"""Repeat held-out-cell baselines on full 350 ms stimulus windows from Feather."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

from predict_baselines import DATA, build_cell, run


def load_full() -> list[dict]:
    with (DATA / "bychannel_summary.csv").open(newline="", encoding="utf-8-sig") as f:
        summary = list(csv.DictReader(f))
    files = {path.name: path for path in DATA.rglob("*_preprocessed.feather")}
    cells = []
    for meta in summary:
        name = Path(meta["path"]).name
        table = feather.read_table(files[name], columns=["sweep", "ti", "force", "work", "i_blsub", "v"])
        data = {column: table[column].to_numpy() for column in table.column_names}
        start_ms = round((float(meta["tpeakf"]) - 150) / 250) * 250
        mask = (
            (data["sweep"] == int(meta["sweep"]))
            & (data["ti"] >= start_ms - 1e-6)
            & (data["ti"] <= start_ms + 350 + 1e-6)
        )
        time = data["ti"][mask]
        if len(time) != 8751 or not np.isclose(time[0], start_ms, atol=1e-4):
            raise ValueError(f"Unexpected time window for {meta['uniqueID']}: {len(time)} points")
        current = np.abs(data["i_blsub"][mask]) - float(meta["offset"])
        cell = build_cell(meta, data["force"][mask], data["work"][mask], current)
        voltage = data["v"][mask]
        cell["v_span_mV"] = float(np.quantile(voltage, .99) - np.quantile(voltage, .01))
        cells.append(cell)
    return cells


if __name__ == "__main__":
    calibrated = load_full()
    strict = [
        {**cell, "y": cell["y"] + cell["baseline_pA"], "baseline_pA": 0.0}
        for cell in calibrated
    ]
    # Strict here means no additional test-cell baseline subtraction beyond the
    # offset already present in the authors' preprocessed current definition.
    run(strict, "full_strict")
    run(calibrated, "full")
