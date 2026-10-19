"""Cell-level WT phenotype screen and frozen Piezo1-cap2 projection.

One cell is one observation. All metrics and QC come from previously frozen
audits. The WT channel labels are perfectly confounded with acquisition date,
so date holdout is a robustness check, not batch-adjusted channel inference.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata

from within_cell_power import fit_one


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "Data" / "2022_pnas" / "2022_pnas"
FIELDS = ("log_wthresh", "work_alpha", "peak_delay_ms", "postpeak20_retention")
SETS = {"four_feature": FIELDS,
        "without_exponent": ("log_wthresh", "peak_delay_ms", "postpeak20_retention"),
        "timing_only": ("peak_delay_ms",),
        "threshold_retention": ("log_wthresh", "postpeak20_retention"),
        "without_timing": ("log_wthresh", "work_alpha", "postpeak20_retention")}
PENALTY = 0.25
PRIMARY = "without_exponent"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def chimera_exponents(summary: dict[str, dict]) -> dict[str, dict]:
    """Recompute chimA exponent using the identical WT fit rule and source fields."""
    data = defaultdict(lambda: ([], []))
    path = SOURCE / "chimera_experiments" / "capchims_approaches.csv"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            cell = row["uniqueID"]
            if cell in summary and row["sweep"] == summary[cell]["sweep"]:
                data[cell][0].append(float(row["work"]))
                data[cell][1].append(float(row["absi_blsub_mech"]))
    return {cell: fit_one(np.asarray(work), np.asarray(current),
                          float(summary[cell]["wthresh"]))
            for cell, (work, current) in data.items()}


def make_cells() -> list[dict]:
    waveform = read_csv(HERE / "frozen_wt_projection_scores.csv")
    wt_meta = {row["uniqueID"]: row for row in
               read_csv(SOURCE / "bychannel" / "bychannel_summary.csv")
               if row["construct"] in {"mp1", "mp2"}}
    chim_meta = {row["uniqueID"]: row for row in
                 read_csv(SOURCE / "chimera_experiments" / "capchims_summary.csv")
                 if row["construct"] == "chimA"}
    wt_alpha = {row["cell_id"]: row for row in read_csv(HERE / "within_cell_power.csv")}
    chim_alpha = chimera_exponents(chim_meta)
    groups = defaultdict(list)
    for row in waveform:
        if row["group"] in {"WT_mp1", "WT_mp2", "chimA"}:
            groups[row["cell_id"]].append(row)
    cells = []
    for cell, sweeps in sorted(groups.items()):
        group = sweeps[0]["group"]
        meta = chim_meta[cell] if group == "chimA" else wt_meta[cell]
        exponent = chim_alpha[cell] if group == "chimA" else wt_alpha[cell]
        alpha = float(exponent["alpha"] if group == "chimA" else exponent["work_alpha"])
        span = float(exponent["log_span"] if group == "chimA" else exponent["work_log_span"])
        r2 = float(exponent["log_r2"] if group == "chimA" else exponent["work_log_r2"])
        work_threshold = float(meta["wthresh"])
        record = {"cell_id": cell, "group": group, "date": sweeps[0]["date"],
                  "n_qc_sweeps": len(sweeps),
                  "wthresh_fJ": work_threshold,
                  "log_wthresh": float(np.log(work_threshold)),
                  "work_alpha": alpha, "work_log_span": span, "work_log_r2": r2,
                  "peak_delay_ms": float(np.median([float(x["peak_delay_ms"]) for x in sweeps])),
                  "postpeak20_retention": float(np.median([
                      float(x["postpeak_20ms_retention"]) for x in sweeps]))}
        if all(np.isfinite(record[field]) for field in FIELDS):
            cells.append(record)
    return cells


def design(rows: list[dict], fields: tuple[str, ...]) -> np.ndarray:
    return np.asarray([[float(row[field]) for field in fields] for row in rows], dtype=float)


def fit_axis(rows: list[dict], fields: tuple[str, ...], labels: np.ndarray | None = None) -> dict:
    x = design(rows, fields)
    y = (np.asarray([int(row["group"] == "WT_mp2") for row in rows])
         if labels is None else labels)
    assert set(y) == {0, 1}
    center, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    z = (x-center)/scale
    # Equal weight per acquisition date within each class; equal class weight.
    by_date = Counter((int(yi), row["date"]) for yi, row in zip(y, rows))
    n_dates = Counter(class_label for class_label, _ in
                      set((int(yi), row["date"]) for yi, row in zip(y, rows)))
    weight = np.asarray([1/(2*n_dates[int(yi)]*by_date[int(yi), row["date"]])
                         for yi, row in zip(y, rows)])

    def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
        linear = z@params[:-1]+params[-1]
        probability = expit(linear)
        loss = float(np.dot(weight, np.logaddexp(0, linear)-y*linear)
                     + 0.5*PENALTY*np.dot(params[:-1], params[:-1]))
        grad = np.r_[z.T@(weight*(probability-y)) + PENALTY*params[:-1],
                     np.dot(weight, probability-y)]
        return loss, grad

    result = minimize(objective, np.zeros(len(fields)+1), jac=True, method="BFGS",
                      options={"gtol": 1e-9, "maxiter": 250})
    if not np.all(np.isfinite(result.x)):
        raise RuntimeError("Non-finite phenotype axis")
    return {"fields": fields, "center": center, "scale": scale,
            "weights": result.x[:-1], "intercept": float(result.x[-1])}


def score(model: dict, rows: list[dict]) -> np.ndarray:
    return ((design(rows, model["fields"])-model["center"])/model["scale"]
            @model["weights"] + model["intercept"])


def auc(y: np.ndarray, value: np.ndarray) -> float:
    positives, negatives = y == 1, y == 0
    ranks = rankdata(value)
    return float((np.sum(ranks[positives])-np.sum(positives)*(np.sum(positives)+1)/2)
                 /(np.sum(positives)*np.sum(negatives)))


def evaluate(rows: list[dict], fields: tuple[str, ...],
             date_labels: dict[str, int] | None = None) -> tuple[list[dict], dict]:
    dates = sorted({row["date"] for row in rows})
    true = np.asarray([int(row["group"] == "WT_mp2") if date_labels is None
                       else date_labels[row["date"]] for row in rows])
    output = []
    for date in dates:
        train_indices = [i for i, row in enumerate(rows) if row["date"] != date]
        test_indices = [i for i, row in enumerate(rows) if row["date"] == date]
        train = [rows[i] for i in train_indices]
        test = [rows[i] for i in test_indices]
        model = fit_axis(train, fields, true[train_indices])
        for row, z, label in zip(test, score(model, test), true[test_indices]):
            output.append({"cell_id": row["cell_id"], "date": date,
                           "group": row["group"], "test_label": int(label),
                           "model": "+".join(fields), "logit": float(z),
                           "p_piezo2": float(expit(z))})
    labels = np.asarray([row["test_label"] for row in output])
    logits = np.asarray([row["logit"] for row in output])
    date_means = {date: float(np.mean([row["logit"] for row in output if row["date"] == date]))
                  for date in dates}
    date_y = np.asarray([date_labels[date] if date_labels else
                         int(next(row["group"] == "WT_mp2" for row in output
                                  if row["date"] == date)) for date in dates])
    date_z = np.asarray([date_means[date] for date in dates])
    metrics = {"cells": len(rows), "dates": len(dates), "cell_auc": auc(labels, logits),
               "date_auc": auc(date_y, date_z),
               "cell_balanced_accuracy": float(np.mean([
                   np.mean((logits[labels == y] > 0) == (y == 1)) for y in (0, 1)])),
               "date_balanced_accuracy": float(np.mean([
                   np.mean((date_z[date_y == y] > 0) == (y == 1)) for y in (0, 1)]))}
    return output, metrics


def main() -> None:
    cells = make_cells()
    wt = [row for row in cells if row["group"] in {"WT_mp1", "WT_mp2"}]
    chim = [row for row in cells if row["group"] == "chimA"]
    if len(wt) != 22 or len(chim) != 11:
        raise ValueError(f"Frozen cohort changed: {len(wt)} WT, {len(chim)} chimA")
    write_csv(HERE / "mechanotransduction_phenotype_cells.csv", cells)
    metrics, predictions = {}, []
    for name, fields in SETS.items():
        values, metrics[name] = evaluate(wt, fields)
        for row in values:
            row["model"] = name
        predictions += values
    write_csv(HERE / "mechanotransduction_phenotype_lodo.csv", predictions)

    # Exact date-label permutation: 3 of 11 entire dates assigned to Piezo2.
    # This tests separation beyond random *date* labels, but cannot remove the
    # real class/date confounding because no date has both WT constructs.
    dates = sorted({row["date"] for row in wt})
    observed_auc = metrics[PRIMARY]["date_auc"]
    null = []
    for chosen in combinations(dates, 3):
        labels = {date: int(date in chosen) for date in dates}
        _, result = evaluate(wt, SETS[PRIMARY], labels)
        null.append(result["date_auc"])
    permutation_p = float(sum(value >= observed_auc-1e-12 for value in null)/len(null))

    model = fit_axis(wt, SETS[PRIMARY])
    for row, z in zip(cells, score(model, cells)):
        row["frozen_axis_logit"] = float(z)
        row["frozen_axis_p_piezo2"] = float(expit(z))
    wt_p1_z = np.median([row["frozen_axis_logit"] for row in cells if row["group"] == "WT_mp1"])
    wt_p2_z = np.median([row["frozen_axis_logit"] for row in cells if row["group"] == "WT_mp2"])
    for row in cells:
        row["frozen_axis_p1_to_p2"] = (row["frozen_axis_logit"]-wt_p1_z)/(wt_p2_z-wt_p1_z)
    ablation = {}
    for name in ("timing_only", "threshold_retention", "without_timing"):
        submodel = fit_axis(wt, SETS[name])
        subscore = score(submodel, cells)
        p1 = float(np.median([z for row, z in zip(cells, subscore)
                              if row["group"] == "WT_mp1"]))
        p2 = float(np.median([z for row, z in zip(cells, subscore)
                              if row["group"] == "WT_mp2"]))
        for row, z in zip(cells, subscore):
            row[f"{name}_p1_to_p2"] = float((z-p1)/(p2-p1))
        ablation[name] = {group: float(np.median([
            row[f"{name}_p1_to_p2"] for row in cells if row["group"] == group]))
            for group in ("WT_mp1", "WT_mp2", "chimA")}
    write_csv(HERE / "mechanotransduction_phenotype_projection.csv", cells)
    date_summary = []
    for group, date in sorted({(row["group"], row["date"]) for row in cells}):
        part = [row for row in cells if row["group"] == group and row["date"] == date]
        date_summary.append({"group": group, "date": date, "cells": len(part),
                             **{f"median_{field}": float(np.median([row[field] for row in part]))
                                for field in FIELDS},
                             "median_axis": float(np.median([
                                 row["frozen_axis_p1_to_p2"] for row in part]))})
    write_csv(HERE / "mechanotransduction_phenotype_dates.csv", date_summary)

    positions = {}
    for field in FIELDS:
        p1 = np.median([row[field] for row in cells if row["group"] == "WT_mp1"])
        p2 = np.median([row[field] for row in cells if row["group"] == "WT_mp2"])
        ch = np.median([row[field] for row in chim])
        positions[field] = {"p1_median": float(p1), "p2_median": float(p2),
                            "chim_median": float(ch),
                            "chim_fraction_p1_to_p2": float((ch-p1)/(p2-p1))}
    # Diagnostic only: a steep exponent from a narrow log-work span can be an
    # estimation artifact. Fit span adjustment on WT, then inspect chimera.
    wt_span = np.asarray([row["work_log_span"] for row in wt])
    wt_alpha = np.asarray([row["work_alpha"] for row in wt])
    span_slope, span_intercept = np.polyfit(wt_span, wt_alpha, 1)
    common_span = float(np.median(wt_span))
    exponent_diagnostic = {"wt_alpha_vs_log_span_slope": float(span_slope),
                           "common_log_span": common_span, "by_group": {}}
    for group in ("WT_mp1", "WT_mp2", "chimA"):
        group_rows = [row for row in cells if row["group"] == group]
        exponent_diagnostic["by_group"][group] = {
            "median_log_span": float(np.median([r["work_log_span"] for r in group_rows])),
            "cells_span_below_0_2": sum(r["work_log_span"] < .2 for r in group_rows),
            "median_alpha_span_at_least_0_2": float(np.median([
                r["work_alpha"] for r in group_rows if r["work_log_span"] >= .2])),
            "median_alpha_linear_span_adjusted": float(np.median([
                r["work_alpha"]-span_slope*(r["work_log_span"]-common_span)
                for r in group_rows]))}
    summaries = {}
    for group in ("WT_mp1", "WT_mp2", "chimA"):
        group_rows = [row for row in cells if row["group"] == group]
        summaries[group] = {"cells": len(group_rows),
                            "dates": len({row["date"] for row in group_rows}),
                            "median_axis": float(np.median([row["frozen_axis_p1_to_p2"]
                                                              for row in group_rows])),
                            "median_p_piezo2": float(np.median([row["frozen_axis_p_piezo2"]
                                                                  for row in group_rows]))}
    output = {"cross_validation": metrics,
              "primary_axis": PRIMARY,
              "date_label_permutation": {"number_of_assignments": len(null),
                                         "observed_date_auc": observed_auc,
                                         "fraction_at_least_observed": permutation_p},
              "frozen_projection": summaries, "ablation_projection": ablation,
              "dimension_positions": positions, "exponent_diagnostic": exponent_diagnostic,
              "frozen_standardized_weights": {field: float(weight) for field, weight in
                                              zip(model["fields"], model["weights"])},
              "warning": "Every WT date has only one channel; date holdout cannot identify a batch-independent channel effect."}
    (HERE / "mechanotransduction_phenotype_results.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4), layout="constrained")
    colors = {"WT_mp1": "#4169a6", "WT_mp2": "#de8c32", "chimA": "#6b4a8d"}
    rng = np.random.default_rng(20260927)
    for group, xcenter in (("WT_mp1", 0), ("WT_mp2", 1), ("chimA", 2)):
        values = [row["frozen_axis_p1_to_p2"] for row in cells if row["group"] == group]
        axes[0].scatter(xcenter+rng.uniform(-0.12, 0.12, len(values)), values,
                        color=colors[group], alpha=.85, s=32)
        axes[0].plot([xcenter-.2, xcenter+.2], [np.median(values)]*2,
                     color="black", linewidth=2)
    axes[0].axhline(0, color="gray", linestyle=":")
    axes[0].axhline(1, color="gray", linestyle=":")
    axes[0].set_xticks([0, 1, 2], ["WT P1", "WT P2", "P1 cap2"])
    axes[0].set_ylabel("Frozen WT axis (P1 median=0, P2 median=1)")
    axes[0].set_title("Retrospective cell phenotype: threshold + delay + retention")
    for i, field in enumerate(FIELDS):
        p = positions[field]["chim_fraction_p1_to_p2"]
        axes[1].scatter(p, i, color="#6b4a8d", s=65)
    axes[1].axvline(0, color="#4169a6", linestyle="--", label="WT P1 median")
    axes[1].axvline(1, color="#de8c32", linestyle="--", label="WT P2 median")
    axes[1].set_yticks(range(len(FIELDS)),
                       ["log work threshold", "work exponent", "peak delay", "20-ms retention"])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Chimera median position along each WT difference")
    axes[1].set_title("Chimera dimensions (exponent is span-sensitive)")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.savefig(HERE / "mechanotransduction_phenotype.png", dpi=180)
    plt.close(fig)
    held = [row for row in predictions if row["model"] == PRIMARY]
    date_order = sorted({row["date"] for row in held})
    fig, ax = plt.subplots(figsize=(10.8, 4.7), layout="constrained")
    for i, date in enumerate(date_order):
        group_rows = [row for row in held if row["date"] == date]
        group = group_rows[0]["group"]
        p = [row["p_piezo2"] for row in group_rows]
        ax.scatter(i+rng.uniform(-.12, .12, len(p)), p, color=colors[group], s=34)
        ax.plot([i-.2, i+.2], [np.mean(p)]*2, color="black", linewidth=1.7)
    ax.axhline(.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(date_order)), date_order, rotation=55, ha="right")
    ax.set_ylabel("Held-date Piezo2 score (logistic scale)")
    ax.set_xlabel("Entire acquisition date held out")
    ax.set_title("WT test cells; each date contains only one channel")
    fig.savefig(HERE / "mechanotransduction_phenotype_holdout.png", dpi=180)
    plt.close(fig)
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
