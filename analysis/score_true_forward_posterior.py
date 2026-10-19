"""Score posterior draws propagated through the actual ODEs at sealed load."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from train_direct_load_predictor import DirectLoadNetwork, get_data


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cohort", type=Path)
    p.add_argument("forward_parts", nargs="+", type=Path)
    a = p.parse_args()
    blocks = []
    for path in a.forward_parts:
        with h5py.File(path) as f:
            blocks.append({
                "ta": f["Ta_kPa"][()].T,
                "cell": f["cell_index_zero_based"][()].ravel().astype(int),
                "mode": f["mode"][()].ravel().astype(int),
                "draw": f["draw_index"][()].ravel().astype(int),
                "ok": f["ok"][()].ravel().astype(bool),
            })
    concat = {key: np.concatenate([block[key] for block in blocks]) for key in blocks[0]}
    if not concat["ok"].all():
        raise ValueError(f"{np.sum(~concat['ok'])} true-forward simulations failed")
    with h5py.File(a.cohort) as f:
        lam = f["lambda"][()]
        actual = f["Ta_kPa"][()]
        split = f["split_code"][()]
    ix = [int(np.argmin(abs(lam - x))) for x in [1.0, 1.1, 1.125]]
    assert np.allclose(lam[ix], [1., 1.1, 1.125])
    cells = sorted(set(concat["cell"]))
    assert np.all(split[cells] == 2)
    direct_data = get_data(a.cohort)
    ap, ca, tension, _, _ = direct_data
    torch.set_num_threads(4)
    result = {"target_lambda": 1.125, "n_test_cells": len(cells),
              "draws_per_cell_per_mode": int(sum((concat["cell"] == cells[0]) &
                                             (concat["mode"] == 2))), "modes": {}}
    for mode in [2, 3]:
        n_cal = 1 if mode == 2 else 2
        direct = DirectLoadNetwork(n_cal, ap.shape[1])
        direct.load_state_dict(torch.load(a.cohort.parent / f"direct_{n_cal}load/model.pt",
                                          map_location="cpu", weights_only=True))
        direct.eval()
        per_cell = []
        for cell in cells:
            choose = (concat["cell"] == cell) & (concat["mode"] == mode)
            draws = concat["ta"][choose, :601:2]
            target = actual[cell, ix[2], :601:2]
            estimate = np.median(draws, axis=0)
            lo, hi = np.quantile(draws, [.05, .95], axis=0)
            baseline = actual[cell, ix[0], :601:2].copy()
            if mode == 3:
                baseline += 1.25 * (actual[cell, ix[1], :601:2] - baseline)
            with torch.no_grad():
                ids = ix[:n_cal]
                direct_prediction = direct(
                    ap[cell:cell+1], ca[cell:cell+1],
                    tension[cell:cell+1, ids, :],
                    torch.tensor(lam[ids], dtype=torch.float32).reshape(1, n_cal),
                    torch.tensor([1.125], dtype=torch.float32),
                ).numpy().ravel()
            per_cell.append({
                "cell": int(cell),
                "posterior_waveform_MAE_kPa": float(abs(estimate - target).mean()),
                "posterior_peak_MAE_kPa": float(abs(estimate.max() - target.max())),
                "posterior_90pct_pointwise_coverage": float(((target >= lo) & (target <= hi)).mean()),
                "posterior_mean_90pct_interval_width_kPa": float((hi-lo).mean()),
                "static_waveform_MAE_kPa": float(abs(baseline - target).mean()),
                "static_peak_MAE_kPa": float(abs(baseline.max() - target.max())),
                "direct_waveform_MAE_kPa": float(abs(direct_prediction - target).mean()),
                "direct_peak_MAE_kPa": float(abs(direct_prediction.max() - target.max())),
            })
        result["modes"][str(mode)] = {
            "mean_posterior_waveform_MAE_kPa": float(np.mean([x["posterior_waveform_MAE_kPa"] for x in per_cell])),
            "mean_posterior_peak_MAE_kPa": float(np.mean([x["posterior_peak_MAE_kPa"] for x in per_cell])),
            "mean_posterior_90pct_pointwise_coverage": float(np.mean([x["posterior_90pct_pointwise_coverage"] for x in per_cell])),
            "mean_posterior_90pct_interval_width_kPa": float(np.mean([x["posterior_mean_90pct_interval_width_kPa"] for x in per_cell])),
            "mean_static_waveform_MAE_kPa": float(np.mean([x["static_waveform_MAE_kPa"] for x in per_cell])),
            "mean_static_peak_MAE_kPa": float(np.mean([x["static_peak_MAE_kPa"] for x in per_cell])),
            "mean_direct_waveform_MAE_kPa": float(np.mean([x["direct_waveform_MAE_kPa"] for x in per_cell])),
            "mean_direct_peak_MAE_kPa": float(np.mean([x["direct_peak_MAE_kPa"] for x in per_cell])),
            "per_cell": per_cell,
        }
    output = a.cohort.parent / "posterior_true_forward_16x8_score.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"n_test_cells": len(cells), "modes": {
        k: {q:v for q,v in row.items() if q != "per_cell"} for k,row in result["modes"].items()}}, indent=2))


if __name__ == "__main__":
    main()
