"""Project external Piezo perturbations into a frozen, signed WT PCA space.

This is a retrospective phenotype check. PCA, alignment, normalization, QC,
and component count are fixed before reading perturbation records. Different
collection dates prevent a clean causal chimera-versus-WT comparison.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

from dynamic_input_screen import HERE, ROOT, extract, load_traces
from one_shot_work import first_calibration, read_metadata
from waveform_shape_transfer import fit_pca


DATA = HERE.parent / "Data" / "2022_pnas" / "2022_pnas"
LEFT, RIGHT = 60, 140
PCS = 3


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def external(rows: list[dict], root: Path, labels: dict[str, str]) -> list[dict]:
    paths = {path.name: path for path in root.rglob("*_preprocessed.feather")}
    result = []
    for row in rows:
        construct = row["construct"]
        if construct not in labels:
            continue
        path = paths.get(Path(row["path"]).name)
        if path is None:
            continue
        table = feather.read_table(path, columns=["sweep", "ti", "force", "position",
                                                  "deflection", "work", "i_blsub", "v"])
        data = {key: table[key].to_numpy() for key in table.column_names}
        sweep = int(row["sweep"])
        # Author peak times carry sub-ms offsets; match the existing analysis
        # convention of a whole-ms 350-ms window on the 0.04-ms sample grid.
        start = round(float(row["tpeakf"]))-150
        mask = ((data["sweep"] == sweep) & (data["ti"] >= start-1e-6) &
                (data["ti"] <= start+350+1e-6))
        if np.count_nonzero(mask) != 8751:
            continue
        trace, _ = extract(data, mask, float(row["offset"]))
        voltage = float(np.median(data["v"][mask]))
        if abs(voltage+80) > 5:
            continue
        current = -data["i_blsub"][mask][:8750].reshape(350, 25).mean(axis=1)
        current -= np.median(current[:10])
        noise = max(float(np.std(current[:10])), 1e-9)
        contact = max(int(trace["contact_ms"])-5, 0)
        peak_index = contact+int(np.argmax(current[contact:]))
        amplitude = float(current[peak_index])
        if amplitude/noise < 5:
            continue
        force_peak = int(np.argmax(trace["F"][:150]))
        segment = current[force_peak-LEFT:force_peak+RIGHT+1]
        if len(segment) != LEFT+RIGHT+1:
            continue
        mechanical = np.stack([trace[key][force_peak-LEFT:force_peak+RIGHT+1]
                               for key in ("F", "W")])
        result.append({"cell_id": row["uniqueID"], "date": row["date"],
                       "construct": construct, "group": labels[construct],
                       "sweep": sweep, "peak_delay_ms": peak_index-force_peak,
                       "postpeak_20ms_retention": float(current[min(peak_index+20, 349)]/amplitude),
                       "peak_snr": amplitude/noise, "shape": segment/amplitude,
                       "mechanical": mechanical,
                       "work_threshold_fJ": float(row["wthresh"])})
    return result


def main() -> None:
    audit = [r for r in read_csv(HERE / "signed_quality_audit_sweeps.csv")
             if r["geometry_qc"] == "1" and abs(float(r["voltage_mV"])+80) <= 5]
    by_cell = read_metadata()
    calibration = {cell: first_calibration(rows) for cell, rows in by_cell.items()}
    traces, _ = load_traces(by_cell, calibration)
    with (ROOT / "bychannel_summary.csv").open(newline="", encoding="utf-8-sig") as handle:
        selected = list(csv.DictReader(handle))
    paths = {path.name: path for path in ROOT.rglob("*_preprocessed.feather")}
    wt = []
    by_selected = defaultdict(list)
    for row in audit:
        by_selected[row["cell_id"]].append(row)
    for chosen in selected:
        cell = chosen["uniqueID"]
        if cell not in by_selected:
            continue
        path = paths[Path(chosen["path"]).name]
        table = feather.read_table(path, columns=["sweep", "ti", "i_blsub"])
        data = {key: table[key].to_numpy() for key in table.column_names}
        for row in by_selected[cell]:
            sweep = int(row["sweep"])
            source = next(item for item in by_cell[cell] if int(item["sweep"]) == sweep)
            start = float(source["compression_end_ms"])-150
            mask = ((data["sweep"] == sweep) & (data["ti"] >= start-1e-6) &
                    (data["ti"] <= start+350+1e-6))
            current = -data["i_blsub"][mask][:8750].reshape(350, 25).mean(axis=1)
            current -= np.median(current[:10])
            trace = traces[cell, sweep]
            force_peak = int(np.argmax(trace["F"][:150]))
            contact = max(int(trace["contact_ms"])-5, 0)
            current_peak = contact+int(np.argmax(current[contact:]))
            amplitude = float(current[current_peak])
            segment = current[force_peak-LEFT:force_peak+RIGHT+1]
            mechanical = np.stack([trace[key][force_peak-LEFT:force_peak+RIGHT+1]
                                   for key in ("F", "W")])
            wt.append({"cell_id": cell, "date": row["date"],
                       "construct": row["channel"], "group": "WT_"+row["channel"],
                       "sweep": sweep, "peak_delay_ms": current_peak-force_peak,
                       "postpeak_20ms_retention": float(current[min(current_peak+20, 349)]/amplitude),
                       "peak_snr": amplitude/max(float(np.std(current[:10])), 1e-9),
                       "shape": segment/amplitude, "mechanical": mechanical,
                       "work_threshold_fJ": float(chosen["wthresh"])})
    # Freeze WT PCA before perturbation records are opened. First 3 components
    # are fixed, not selected by chimera/CytoD separation.
    center, axes = fit_pca(np.stack([r["shape"] for r in wt]), [r["cell_id"] for r in wt])
    axes = axes[:PCS]
    chim = external(read_csv(DATA / "chimera_experiments" / "capchims_summary.csv"),
                    DATA / "chimera_experiments", {"chimA": "chimA"})
    cyto = external(read_csv(DATA / "cytod" / "mp1-mp2-cytod" / "piezocd_summary.csv"),
                    DATA / "cytod" / "mp1-mp2-cytod",
                    {"mp2-dmso": "Piezo2_DMSO", "mp2-cytod": "Piezo2_CytoD",
                     "mp1": "Piezo1_control", "mp1cytod": "Piezo1_CytoD"})
    all_rows = wt+chim+cyto
    np.savez_compressed(HERE / "frozen_wt_waveform_arrays.npz",
                        current_shape=np.stack([r["shape"] for r in all_rows]),
                        mechanical=np.stack([r["mechanical"] for r in all_rows]),
                        cell_id=np.array([r["cell_id"] for r in all_rows]),
                        date=np.array([r["date"] for r in all_rows]),
                        group=np.array([r["group"] for r in all_rows]),
                        sweep=np.array([r["sweep"] for r in all_rows]),
                        peak_delay_ms=np.array([r["peak_delay_ms"] for r in all_rows]),
                        postpeak_20ms_retention=np.array([
                            r["postpeak_20ms_retention"] for r in all_rows]),
                        work_threshold_fJ=np.array([
                            r["work_threshold_fJ"] for r in all_rows]))
    wt_coordinates = np.stack([(row["shape"]-center)@axes.T for row in wt])
    def equal_cell_centroid(group: str) -> np.ndarray:
        cells = sorted({row["cell_id"] for row in wt if row["group"] == group})
        return np.mean([np.mean(wt_coordinates[np.array([row["cell_id"] == cell for row in wt])],
                                   axis=0) for cell in cells], axis=0)

    p1 = equal_cell_centroid("WT_mp1")
    p2 = equal_cell_centroid("WT_mp2")
    direction = p2-p1
    denom = float(direction@direction)
    scores = []
    for row in all_rows:
        coordinate = (row["shape"]-center)@axes.T
        along = float((coordinate-p1)@direction/denom)
        scores.append({key: value for key, value in row.items()
                       if key not in {"shape", "mechanical"}} |
                      {"pc1": float(coordinate[0]), "pc2": float(coordinate[1]),
                       "pc3": float(coordinate[2]), "p1_to_p2_axis": along,
                       "distance_to_p1_centroid": float(np.linalg.norm(coordinate-p1)),
                       "distance_to_p2_centroid": float(np.linalg.norm(coordinate-p2))})
    with (HERE / "frozen_wt_projection_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scores[0]))
        writer.writeheader()
        writer.writerows(scores)
    for group in sorted(set(row["group"] for row in scores)):
        items = [row for row in scores if row["group"] == group]
        per_cell = defaultdict(list)
        for row in items:
            per_cell[row["cell_id"]].append(row)
        axis = np.array([np.mean([r["p1_to_p2_axis"] for r in cell_rows])
                         for cell_rows in per_cell.values()])
        delay = np.array([np.mean([r["peak_delay_ms"] for r in cell_rows])
                          for cell_rows in per_cell.values()])
        retention = np.array([np.mean([r["postpeak_20ms_retention"] for r in cell_rows])
                              for cell_rows in per_cell.values()])
        print(group, "cells", len(per_cell), "dates", len(set(r["date"] for r in items)),
              "sweeps", len(items), "axis median", round(float(np.median(axis)), 3),
              "delay median", round(float(np.median(delay)), 2),
              "retention median", round(float(np.median(retention)), 3))


if __name__ == "__main__":
    main()
