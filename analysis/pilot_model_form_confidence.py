"""Exploratory, target-blind confidence diagnostic on existing dynamic cohorts.

The predictor uses baseline and C1/C2 only. Sealed long-hold truth is loaded
solely for scoring. These cohorts have already been examined in earlier work;
this is a feasibility audit, not a fresh confirmation experiment.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
OUT = HERE / "human_bidirectional_output"
WINDOW = slice(300, 602)
CAL_WINDOWS = [(0, slice(150, 260)), (1, slice(600, 710))]


def main() -> None:
    with h5py.File(OUT / "dynamic_discrepancy_cohort.mat") as f:
        truth = f["waveforms"][:].astype(np.float64)
        truth_ok = f["ok"][:].astype(bool).all(axis=1)
    with h5py.File(OUT / "dynamic_inverse_calibration.mat") as f:
        cal = f["cal"][:].astype(np.float64)
        tab = f["tab"][:].astype(np.float64).T
        mapping = f["map"][:].ravel().astype(int) - 1
    with h5py.File(OUT / "dynamic_model_candidates.mat") as f:
        candidate_cal = f["candidate_cal"][:].astype(np.float64)
        candidate_target = f["candidate_target"][:].astype(np.float64)
        candidate_ok = f["candidate_ok"][:].astype(bool)
    gate = np.loadtxt(OUT / "dynamic_inverse_gate.csv", delimiter=",")
    beta = json.loads((OUT / "dynamic_flux_pilot_metadata.json").read_text())["shared_beta"]

    assert tab.shape == (270, 16) and len(mapping) == len(tab) == len(gate)
    assert candidate_target.shape == (180, 3, 3, 3, 1001)
    assert truth.shape == (64, 3, 6, 3, 1001)
    selected = np.zeros(len(tab), dtype=int)
    selected[gate[:, 4] < .1] = 2
    contrast = np.log(np.maximum(gate[:, 5], .02) / np.maximum(gate[:, 6], .02))
    selected[(gate[:, 4] >= .1) & (contrast > .05 * beta / 2)] = 1

    # Pair two previously frozen inverse estimators at the same virtual cell.
    row_for = {(int(row[0]), int(row[1]), int(row[2])): k
               for k, row in enumerate(tab)}
    records: list[dict] = []
    for row_index, row in enumerate(tab):
        cell = int(row[0]) - 1
        form = int(row[1]) - 1
        obs_kind = int(row[2])
        if obs_kind != 1 or form not in (0, 1) or not truth_ok[cell]:
            continue
        base_index = row_for[(cell + 1, form + 1, 0)]
        uid = mapping[row_index]
        uid_base = mapping[base_index]
        pick = int(selected[row_index])
        if pick == 2 or not np.all(candidate_ok[uid, :2]):
            continue
        if not candidate_ok[uid_base, pick]:
            continue

        observed_base = truth[cell, form, 0, 0]
        actual = truth[cell, form, 3, 0]
        forecasts = np.stack([
            observed_base + candidate_target[uid, model, 0, 0] - cal[uid, 0, 0]
            for model in (0, 1)
        ])
        prediction = forecasts[pick]
        baseline_inverse_prediction = (
            observed_base + candidate_target[uid_base, pick, 0, 0]
            - cal[uid_base, 0, 0]
        )
        form_spread = float(np.abs(forecasts[0, WINDOW] - forecasts[1, WINDOW]).mean())
        inverse_spread = float(np.abs(prediction[WINDOW] -
                                      baseline_inverse_prediction[WINDOW]).mean())
        error = float(np.abs(prediction[WINDOW] - actual[WINDOW]).mean())
        cal_resid_parts = []
        for j, win in CAL_WINDOWS:
            model_response = candidate_cal[uid, pick, j, 0, win] - cal[uid, 0, 0, win]
            observed_response = truth[cell, form, j + 1, 0, win] - observed_base[win]
            cal_resid_parts.append(float(np.abs(model_response - observed_response).mean()))
        records.append({
            "cell_id": cell,
            "truth_form": form,
            "selected_form": pick,
            "long_hold_v_mae_mV": error,
            "severe_gt5mV": int(error > 5.0),
            "form_disagreement_mV": form_spread,
            "inverse_method_disagreement_mV": inverse_spread,
            "max_disagreement_mV": max(form_spread, inverse_spread),
            "calibration_v_mae_mV": float(np.mean(cal_resid_parts)),
            "predicted_second_peak_mV": float(prediction[WINDOW].max()),
            "predicted_window_mean_mV": float(prediction[WINDOW].mean()),
        })

    assert len(records) > 50
    labels = np.array([r["severe_gt5mV"] for r in records])
    if len(np.unique(labels)) != 2:
        raise RuntimeError("The pilot has only one risk class")
    metrics = {}
    for name in ("form_disagreement_mV", "inverse_method_disagreement_mV",
                 "max_disagreement_mV", "calibration_v_mae_mV",
                 "predicted_second_peak_mV", "predicted_window_mean_mV"):
        score = np.array([r[name] for r in records])
        order = np.argsort(-score)
        budget = max(1, int(np.ceil(.1 * len(records))))
        metrics[name] = {
            "auroc": float(roc_auc_score(labels, score)),
            "average_precision": float(average_precision_score(labels, score)),
            "top_10pct_budget": budget,
            "top_10pct_severe_recall": f"{int(labels[order[:budget]].sum())}/{int(labels.sum())}",
            "spearman_proxy_r": float(np.corrcoef(
                np.argsort(np.argsort(score)),
                np.argsort(np.argsort([r["long_hold_v_mae_mV"] for r in records]))
            )[0, 1]),
        }
    spread = np.array([r["form_disagreement_mV"] for r in records])
    lower_quartile = float(np.quantile(spread, .25))
    low_spread = spread <= lower_quartile
    # Resample virtual-cell identities, preserving paired feedback forms.
    unique_cells = np.array(sorted({r["cell_id"] for r in records}))
    block_index = {cell: np.flatnonzero([r["cell_id"] == cell for r in records])
                   for cell in unique_cells}
    peak_scores = np.array([r["predicted_second_peak_mV"] for r in records])
    inverse_scores = np.array([r["inverse_method_disagreement_mV"] for r in records])
    form_scores = spread
    rng = np.random.default_rng(20261010)
    boot_diffs: dict[str, list[float]] = {"inverse_minus_peak": [], "form_minus_peak": []}
    for _ in range(2000):
        sampled = rng.choice(unique_cells, size=len(unique_cells), replace=True)
        ids = np.concatenate([block_index[cell] for cell in sampled])
        if len(np.unique(labels[ids])) < 2:
            continue
        auc_peak = roc_auc_score(labels[ids], peak_scores[ids])
        boot_diffs["inverse_minus_peak"].append(
            roc_auc_score(labels[ids], inverse_scores[ids]) - auc_peak
        )
        boot_diffs["form_minus_peak"].append(
            roc_auc_score(labels[ids], form_scores[ids]) - auc_peak
        )
    top_n = max(1, int(np.ceil(.1 * len(records))))
    result = {
        "scope": "post-hoc feasibility audit on previously inspected same-core synthetic cohort",
        "target_blind_score_generation": True,
        "candidate_set": "original and saturating SAC, at the C1/C2 inferred physiology",
        "n_form_cell_pairs": int(len(records)),
        "n_unique_cells": int(len({r["cell_id"] for r in records})),
        "n_severe": int(labels.sum()),
        "metrics": metrics,
        "low_form_disagreement_threshold_mV": lower_quartile,
        "low_form_disagreement_n": int(low_spread.sum()),
        "severe_despite_low_form_disagreement": int(labels[low_spread].sum()),
        "all_rows_inverse_spread_exceeds_form_spread": bool(np.all(inverse_scores > spread)),
        "inverse_peak_top_10pct_overlap": int(len(set(np.argsort(-inverse_scores)[:top_n]) &
                                               set(np.argsort(-peak_scores)[:top_n]))),
        "cell_block_bootstrap_auc_differences_95pct": {
            k: [float(v) for v in np.quantile(values, [.025, .5, .975])]
            for k, values in boot_diffs.items()
        },
        "severe_cell_rows": [r for r in records if r["severe_gt5mV"]],
    }
    (OUT / "model_form_confidence_pilot.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (OUT / "model_form_confidence_pilot_cells.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)
    xnames = ["form_disagreement_mV", "inverse_method_disagreement_mV",
              "predicted_second_peak_mV"]
    titles = ["SAC-form spread", "Inverse-method spread", "Predicted second V peak"]
    errs = np.array([r["long_hold_v_mae_mV"] for r in records])
    for ax, name, title in zip(axes, xnames, titles):
        xx = np.array([r[name] for r in records])
        for form, marker in [(0, "o"), (1, "^")]:
            ids = np.array([r["truth_form"] == form for r in records])
            ax.scatter(xx[ids & (labels == 0)], errs[ids & (labels == 0)],
                       marker=marker, s=23, alpha=.55, color="#507b8e")
            ax.scatter(xx[ids & (labels == 1)], errs[ids & (labels == 1)],
                       marker=marker, s=52, alpha=.95, color="#c24a43")
        ax.axhline(5, color="#333333", ls="--", lw=.9)
        ax.set_yscale("log")
        ax.set_ylim(.005, 100)
        if name != "predicted_second_peak_mV":
            ax.set_xscale("log")
        ax.set_xlabel(title + " (mV)")
        ax.set_title(f"AUROC {metrics[name]['auroc']:.3f}; top 10% {metrics[name]['top_10pct_severe_recall']}")
    axes[0].set_ylabel("Sealed long-hold V MAE (mV)")
    fig.suptitle("Exploratory confidence audit: 44 virtual cells, paired SAC forms", fontsize=11)
    fig.savefig(OUT / "model_form_confidence_pilot.png", dpi=220)
    plt.close(fig)
    print(json.dumps({k: v for k, v in result.items() if k != "severe_cell_rows"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
