"""Small causal CNN versus a calibrated force law on future Piezo sweeps.

All future sweeps from a held-out date are kept together. Calibration current
from the same test cell is allowed; its future current never enters training.
The primary cohort is signed inward current at -80 mV with pre-stimulus QC.
"""

from __future__ import annotations

import csv
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import torch
from torch import nn
from torch.nn import functional as F

from adaptation_nested import force_threshold
from dynamic_input_screen import HERE, ROOT, load_traces
from one_shot_work import first_calibration, read_metadata


SEED = int(os.environ.get("CNN_BENCHMARK_SEED", "20260926"))
EPOCHS = 120
LEARNING_RATE = .005
WEIGHT_DECAY = .01
FORCE_GATE = os.environ.get("CNN_FORCE_GATE", "0") == "1"
torch.set_num_threads(2)


class CausalBlock(nn.Module):
    def __init__(self, inputs: int, outputs: int, dilation: int) -> None:
        super().__init__()
        self.left = 4*dilation
        self.conv = nn.Conv1d(inputs, outputs, kernel_size=5, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.conv(F.pad(x, (self.left, 0))))


class SmallCausalCNN(nn.Module):
    def __init__(self, inputs: int, force_gate: bool = False) -> None:
        super().__init__()
        self.force_gate = force_gate
        self.layers = nn.Sequential(CausalBlock(inputs, 6, 1),
                                    CausalBlock(6, 6, 4),
                                    CausalBlock(6, 6, 16))
        self.head = nn.Conv1d(6, 1, kernel_size=1)
        nn.init.constant_(self.head.bias, -2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        response = F.softplus(self.head(self.layers(x))).squeeze(1)
        return response*(x[:, 5, :] > 0).float() if self.force_gate else response


def window(data: dict[str, np.ndarray], sweep: int, end_ms: float) -> np.ndarray:
    start = end_ms-150
    mask = ((data["sweep"] == sweep) & (data["ti"] >= start-1e-6) &
            (data["ti"] <= start+350+1e-6))
    if np.count_nonzero(mask) != 8751:
        raise ValueError((sweep, end_ms, int(mask.sum())))
    return mask


def inward(data: dict[str, np.ndarray], mask: np.ndarray) -> np.ndarray:
    value = -data["i_blsub"][mask][:8750].reshape(350, 25).mean(axis=1)
    return value-np.median(value[:10])


def features(trace: dict, cal_trace: dict, threshold: float, peak_pA: float,
             cm_pF: float, channel: str) -> np.ndarray:
    fscale = max(float(np.max(cal_trace["F"][:150])), 1.0)
    dscale = max(float(np.max(cal_trace["d"][:150])), 1.0)
    wscale = max(float(np.max(cal_trace["W"][:150])), 1.0)
    pscale = max(float(np.max(cal_trace["P"][:150])), .001)
    values = [trace["F"]/fscale, trace["d"]/dscale,
              trace["W"]/wscale, trace["P"]/pscale,
              trace["Fdot"]/(fscale/10),
              np.maximum(trace["F"]-threshold, 0)/max(fscale-threshold, 1),
              np.full(350, threshold/fscale),
              np.full(350, np.log1p(peak_pA)/6),
              np.full(350, cm_pF/30),
              np.full(350, float(channel == "mp2"))]
    return np.clip(np.stack(values), -10, 10).astype(np.float32)


def build_records() -> list[dict]:
    audit = [r for r in csv.DictReader((HERE / "signed_quality_audit_sweeps.csv").open(encoding="utf-8"))
             if r["prospective_qc"] == "1" and abs(float(r["voltage_mV"])+80) <= 5]
    by_cell = read_metadata()
    calibration = {cell: first_calibration(rows) for cell, rows in by_cell.items()}
    traces, _ = load_traces(by_cell, calibration)
    with (ROOT / "bychannel_summary.csv").open(newline="", encoding="utf-8-sig") as handle:
        selected = list(csv.DictReader(handle))
    paths = {path.name: path for path in ROOT.rglob("*_preprocessed.feather")}
    grouped = {}
    for row in audit:
        grouped.setdefault(row["cell_id"], []).append(row)
    records = []
    for meta in selected:
        cell = meta["uniqueID"]
        if cell not in grouped:
            continue
        cal = calibration[cell]
        if cal is None or cal["channel"] not in {"mp1", "mp2"}:
            continue
        cal_sweep = int(cal["sweep"])
        future = [row for row in grouped[cell] if int(row["sweep"]) > cal_sweep]
        if not future:
            continue
        path = paths[Path(meta["path"]).name]
        table = feather.read_table(path, columns=["sweep", "ti", "i_blsub"])
        data = {key: table[key].to_numpy() for key in table.column_names}
        cal_trace = traces[cell, cal_sweep]
        cal_current = inward(data, window(data, cal_sweep, float(cal["compression_end_ms"])))
        cal_start = max(int(cal_trace["contact_ms"])-5, 0)
        cal_peak = float(np.max(cal_current[cal_start:]))
        if cal_peak <= 10:
            continue
        threshold = force_threshold(cal, cal_trace)
        span = max(float(np.max(cal_trace["F"][:150]))-threshold, 1e-9)
        cm = float(meta["Cm"])
        for row in future:
            sweep = int(row["sweep"])
            info = next(item for item in by_cell[cell] if int(item["sweep"]) == sweep)
            trace = traces[cell, sweep]
            target = inward(data, window(data, sweep, float(info["compression_end_ms"])))
            records.append({"cell_id": cell, "date": cal["date"], "channel": cal["channel"],
                            "sweep": sweep, "x": features(trace, cal_trace, threshold,
                                                           cal_peak, cm, cal["channel"]),
                            "target": target.astype(np.float32), "cal_peak": cal_peak,
                            "force_baseline": cal_peak*np.maximum(trace["F"]-threshold, 0)/span,
                            "contact": int(trace["contact_ms"]),
                            "noise": float(np.std(target[:10]))})
    return records


def train(records: list[dict], heldout: str) -> tuple[SmallCausalCNN, float]:
    torch.manual_seed(SEED)
    model = SmallCausalCNN(records[0]["x"].shape[0], force_gate=FORCE_GATE)
    training = [r for r in records if r["date"] != heldout]
    x = torch.tensor(np.stack([r["x"] for r in training]))
    y = torch.tensor(np.stack([r["target"]/r["cal_peak"] for r in training]))
    counts = Counter(r["cell_id"] for r in training)
    cell_weight = torch.tensor([1/counts[r["cell_id"]] for r in training], dtype=torch.float32)
    cell_weight /= cell_weight.sum()
    stimulus_weight = 1+2*(y > .1).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    model.train()
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        prediction = model(x)
        loss_by_time = F.smooth_l1_loss(prediction, y, beta=.5, reduction="none")
        loss = ((loss_by_time*stimulus_weight).mean(dim=1)*cell_weight).sum()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
    return model, float(loss.detach())


def metrics(row: dict, predicted: np.ndarray) -> dict:
    target = row["target"]
    contact = max(row["contact"]-5, 0)
    true_peak_index = contact+int(np.argmax(target[contact:]))
    pred_peak_index = contact+int(np.argmax(predicted[contact:]))
    true_peak = float(target[true_peak_index])
    pred_peak = float(predicted[pred_peak_index])
    end = min(true_peak_index+51, 350)
    return {"mae_pA": float(np.mean(np.abs(target-predicted))),
            "postpeak_0_50ms_mae_pA": float(np.mean(np.abs(target[true_peak_index:end]-
                                                       predicted[true_peak_index:end]))),
            "peak_abs_error_pA": abs(true_peak-pred_peak),
            "peak_time_abs_error_ms": abs(true_peak_index-pred_peak_index),
            "true_peak_pA": true_peak, "predicted_peak_pA": pred_peak,
            "false_positive_charge_fC": float(np.maximum(predicted[target <= max(10, 5*row["noise"])], 0).sum())}


def main() -> None:
    records = build_records()
    result = []
    for date in sorted(set(r["date"] for r in records)):
        training = [r for r in records if r["date"] != date]
        test = [r for r in records if r["date"] == date]
        model, train_loss = train(records, date)
        model.eval()
        with torch.no_grad():
            prediction = model(torch.tensor(np.stack([r["x"] for r in test]))).numpy()
        for row, normalized in zip(test, prediction):
            predictions = {"zero": np.zeros(350),
                           "threshold_force": row["force_baseline"],
                           "causal_cnn": normalized*row["cal_peak"]}
            for name, waveform in predictions.items():
                result.append({"cell_id": row["cell_id"], "channel": row["channel"],
                               "heldout_date": date, "sweep": row["sweep"],
                               "model": name, "train_cells": len(set(r["cell_id"] for r in training)),
                               "train_sweeps": len(training), "train_loss_normalized": train_loss,
                               **metrics(row, waveform)})
        print("fold", date, "train cells", len(set(r["cell_id"] for r in training)),
              "test cells", len(set(r["cell_id"] for r in test)), "loss", round(train_loss, 3), flush=True)
    suffix = "_gated" if FORCE_GATE else ""
    filenames = [f"causal_cnn_benchmark_scores{suffix}_seed{SEED}.csv"]
    if not FORCE_GATE:
        filenames.append("causal_cnn_benchmark_scores.csv")
    for filename in filenames:
        with (HERE / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result[0]))
            writer.writeheader()
            writer.writerows(result)
    print("saved", len(result), "scores", "parameters",
          sum(parameter.numel() for parameter in SmallCausalCNN(records[0]["x"].shape[0],
                                                                 force_gate=FORCE_GATE).parameters()),
          "gated", FORCE_GATE)


if __name__ == "__main__":
    main()
