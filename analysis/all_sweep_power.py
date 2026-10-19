"""Fit paper-style approach-phase power laws across all sweeps of selected cells."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

from within_cell_power import fit_one


ROOT = Path(__file__).resolve().parents[1] / "Data" / "2022_pnas" / "2022_pnas" / "bychannel"
OUT = Path(__file__).resolve().parent / "all_sweep_power.csv"


def main() -> None:
    with (ROOT / "bychannel_summary.csv").open(newline="", encoding="utf-8-sig") as f:
        selected = list(csv.DictReader(f))
    paths = {p.name: p for p in ROOT.rglob("*_preprocessed.feather")}
    rows = []
    for chosen in selected:
        filename = Path(chosen["path"]).name
        path = paths[filename]
        summary_file = path.with_name(filename.replace("_preprocessed.feather", "_summary.csv"))
        with summary_file.open(newline="", encoding="utf-8-sig") as f:
            sweeps = list(csv.DictReader(f))
        table = feather.read_table(path, columns=["sweep", "ti", "force", "work", "i_blsub"])
        data = {col: table[col].to_numpy() for col in table.column_names}
        for meta in sweeps:
            start = round((float(meta["tpeakf"]) - 150) / 250) * 250
            mask = (
                (data["sweep"] == int(meta["sweep"]))
                & (data["ti"] >= start - 1e-6)
                & (data["ti"] <= start + 150 + 1e-6)
            )
            if np.count_nonzero(mask) != 3751:
                raise ValueError(f"Unexpected approach length for {meta['uniqueID']} sweep {meta['sweep']}")
            current = np.abs(data["i_blsub"][mask]) - float(meta["offset"])
            full_mask = (
                (data["sweep"] == int(meta["sweep"]))
                & (data["ti"] >= start - 1e-6)
                & (data["ti"] <= start + 350 + 1e-6)
            )
            if np.count_nonzero(full_mask) != 8751:
                raise ValueError(f"Unexpected full-window length for {meta['uniqueID']} sweep {meta['sweep']}")
            full_current = np.maximum(np.abs(data["i_blsub"][full_mask]) - float(meta["offset"]), 0)
            work = fit_one(data["work"][mask], current, float(meta["wthresh"]))
            force = fit_one(data["force"][mask], current, float(meta["fthresh"]))
            rows.append({
                "cell_id": meta["uniqueID"], "channel": meta["construct"], "date": meta["date"],
                "sweep": int(meta["sweep"]), "selected_sweep": int(meta["sweep"]) == int(chosen["sweep"]),
                "threshold_time_ms": float(meta["thresht"]), "compression_end_ms": start + 150,
                "wthresh_fJ": float(meta["wthresh"]),
                "peak_force_nN": float(meta["peakf"]), "peak_work_fJ": float(meta["peakw"]),
                "current_peak_minus_offset_pA": float(meta["peaki"]) - float(meta["offset"]),
                "rectified_charge_350ms_fC": float(np.sum(full_current) * 0.04),
                "stdev_pA": float(meta["stdev"]),
                "work_alpha": work["alpha"], "work_log_r2": work["log_r2"],
                "work_n": work["n"], "work_log_span": work["log_span"],
                "force_alpha": force["alpha"], "force_log_r2": force["log_r2"],
                "force_n": force["n"], "force_log_span": force["log_span"],
            })
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for channel in sorted({r["channel"] for r in rows}):
        subset = [r for r in rows if r["channel"] == channel]
        valid = [r for r in subset if np.isfinite(r["work_alpha"])]
        before_end = [r for r in valid if r["threshold_time_ms"] < r["compression_end_ms"]]
        print({
            "channel": channel, "sweeps": len(subset), "valid_work_fits": len(valid),
            "valid_with_threshold_during_compression": len(before_end),
            "median_work_alpha_valid": round(float(np.median([r["work_alpha"] for r in valid])), 3),
            "median_work_alpha_before_end": round(float(np.median([r["work_alpha"] for r in before_end])), 3),
        })


if __name__ == "__main__":
    main()
