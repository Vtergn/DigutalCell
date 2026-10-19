"""Post-hoc simplicity audit; not part of the frozen primary comparison."""
from __future__ import annotations

import csv
from pathlib import Path

import h5py
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

OUT = Path(__file__).resolve().parent / "human_bidirectional_output"
for label, stem in [("confirmation", "dynamic_event_margin"),
                    ("replication", "dynamic_event_margin_replication")]:
    cells_file = OUT / f"{stem}_{label}_cells.csv"
    if label == "replication":
        cells_file = OUT / "dynamic_event_margin_replication_cells.csv"
    with cells_file.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with h5py.File(OUT / f"{stem}_targets.mat") as f:
        tab = f["tab"][:].T.astype(float)
        pred = f["pred"][:].astype(float)
    row_by_cell = {int(c): j for j, c in enumerate(tab[:, 0])}
    ids = [int(r["cell"]) for r in rows]
    targets = np.array([int(r["tail_gt5"]) for r in rows])
    baseline = {
        "inferred_g_ionic": np.array([tab[row_by_cell[i], 7] for i in ids]),
        "predicted_peak_voltage": np.array([
            np.max(pred[row_by_cell[i], 1, 0, 300:602]) for i in ids]),
        "predicted_duration": np.array([
            float(r["pred_duration_above_minus50_ms"]) for r in rows]),
        "local_sensitivity": np.array([
            float(r["local_waveform_sensitivity_mV"]) for r in rows]),
    }
    print(label, "n", len(rows), "events", int(targets.sum()))
    for name, score in baseline.items():
        n_top = int(np.ceil(.1*len(rows)))
        top = np.argsort(-score)[:n_top]
        print(name, "AUROC", round(roc_auc_score(targets, score), 3),
              "AP", round(average_precision_score(targets, score), 3),
              "top10_recall", f"{targets[top].sum()}/{targets.sum()}")
    rng = np.random.default_rng(20261025 if label == "replication" else 20261026)
    auc_delta, ap_delta, recall_delta = [], [], []
    for _ in range(3000):
        take = rng.integers(len(rows), size=len(rows))
        if len(set(targets[take])) < 2:
            continue
        yy = targets[take]
        local = baseline["local_sensitivity"][take]
        peak = baseline["predicted_peak_voltage"][take]
        auc_delta.append(roc_auc_score(yy, local)-roc_auc_score(yy, peak))
        ap_delta.append(average_precision_score(yy, local)-
                        average_precision_score(yy, peak))
        k = int(np.ceil(.1*len(yy)))
        recall_delta.append((yy[np.argsort(-local)[:k]].sum()-
                             yy[np.argsort(-peak)[:k]].sum())/yy.sum())
    for metric, values in [("AUROC", auc_delta), ("AP", ap_delta),
                           ("top10_recall", recall_delta)]:
        print("local-minus-peak", metric, "bootstrap_ci95",
              np.round(np.quantile(values, [.025, .975]), 3).tolist())
