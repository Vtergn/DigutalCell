"""Minimal held-out-well TTM feature prediction sanity check.

This is an audit of public response signal, not a personalized digital twin.
One independent unit is the putative well; multiple imaging areas are averaged.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "Data" / "ttm_public" / "ttm_original"
RNG = np.random.default_rng(20260928)


def main():
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        next(handle)
        rows = [dict(zip(("row", "area", "phase", "dose", "feature", "value", "drug"), raw))
                for raw in csv.reader(handle, delimiter=" ")]
    records = defaultdict(dict)
    meta = {}
    for row in rows:
        records[(row["area"], row["phase"])][row["feature"]] = row["value"]
        meta[row["area"]] = (row["drug"], row["dose"])
    features = sorted({row["feature"] for row in rows})
    by_well = defaultdict(list)
    for area, (drug, dose) in meta.items():
        if (area, "baseline") not in records or (area, "added") not in records:
            continue
        baseline, added = records[(area, "baseline")], records[(area, "added")]
        if any(baseline[f] == "NA" or added[f] == "NA" for f in features):
            continue
        fields = area.split(":")
        well = ":".join((fields[0], fields[1], fields[3]))
        base = np.log([float(baseline[f]) for f in features])
        ratio = np.log([float(added[f])/float(baseline[f]) for f in features])
        by_well[(drug, dose, well)].append((base, ratio))
    well_rows = [(drug, dose, well,
                  np.mean([v[0] for v in values], axis=0),
                  np.mean([v[1] for v in values], axis=0))
                 for (drug, dose, well), values in by_well.items()]
    groups = defaultdict(list)
    for index, (drug, dose, _, _, _) in enumerate(well_rows):
        groups[(drug, dose)].append(index)
    evaluated = [j for j, (drug, dose, _, _, _) in enumerate(well_rows)
                 if len(groups[(drug, dose)]) >= 3]
    errors = {"unchanged_baseline": [], "other_wells_same_drug_dose": [],
              "drug_dose_plus_baseline_deviation": []}
    cross_modal_contraction_errors = []
    contraction_indices = [k for k, f in enumerate(features) if f.startswith("contraction")]
    slopes = []
    for j in evaluated:
        drug, dose, _, baseline, truth = well_rows[j]
        others = [well_rows[k][4] for k in groups[(drug, dose)] if k != j]
        other_baselines = [well_rows[k][3] for k in groups[(drug, dose)] if k != j]
        population = np.mean(others, axis=0)
        # Pool a diagonal baseline-response slope across training wells after
        # centering within each drug-dose group. The test well is excluded from
        # both its group mean and this pooled coefficient fit.
        train_groups = defaultdict(list)
        for k, (d, c, _, x, y) in enumerate(well_rows):
            if k != j:
                train_groups[(d, c)].append((x, y))
        xs, ys = [], []
        for block in train_groups.values():
            if len(block) < 2:
                continue
            block_x = np.asarray([x for x, _ in block])
            block_y = np.asarray([y for _, y in block])
            xs.extend(block_x - np.mean(block_x, axis=0))
            ys.extend(block_y - np.mean(block_y, axis=0))
        xs, ys = np.asarray(xs), np.asarray(ys)
        variance = np.sum(xs**2, axis=0)
        slope = np.sum(xs*ys, axis=0)/(variance + 0.2*variance + 1e-12)
        slopes.append(slope)
        centered_baseline = baseline-np.mean(other_baselines, axis=0)
        personalized = population + slope*centered_baseline
        standardizer = np.std(xs, axis=0)
        standardizer[standardizer < 1e-8] = 1.
        standardized_x = xs/standardizer
        # Fixed-size, strongly regularized linear cross-modal model. The
        # subtraction above removes drug-dose mean effects and the held-out
        # well never contributes to coefficient estimation.
        ridge = 0.2*len(xs)
        coefficients = np.linalg.solve(
            standardized_x.T@standardized_x + ridge*np.eye(len(features)),
            standardized_x.T@ys[:, contraction_indices])
        cross_modal_co = (population[contraction_indices]
                          + (centered_baseline/standardizer)@coefficients)
        cross_modal_contraction_errors.append(np.abs(truth[contraction_indices]-cross_modal_co))
        errors["unchanged_baseline"].append(np.abs(truth))
        errors["other_wells_same_drug_dose"].append(np.abs(truth - population))
        errors["drug_dose_plus_baseline_deviation"].append(np.abs(truth-personalized))
    scores = {}
    for name, values in errors.items():
        arr = np.asarray(values)
        scores[name] = {
            "mean_absolute_log_ratio_error": float(np.mean(arr)),
            "median_well_mean_absolute_log_ratio_error": float(np.median(np.mean(arr, axis=1))),
            "by_modality_mean_absolute_log_ratio_error": {
                "AP_optical": float(np.mean(arr[:, [k for k, f in enumerate(features) if f.startswith("VSD")]])),
                "calcium_optical": float(np.mean(arr[:, [k for k, f in enumerate(features) if f.startswith("CSD")]])),
                "contraction_motion": float(np.mean(arr[:, [k for k, f in enumerate(features) if f.startswith("contraction")]])),
            },
        }
    diagonal_co = np.mean(np.asarray(errors["drug_dose_plus_baseline_deviation"])
                          [:, contraction_indices], axis=1)
    cross_modal_co = np.mean(np.asarray(cross_modal_contraction_errors), axis=1)
    cross_modal_by_drug = defaultdict(list)
    for index, diff in zip(evaluated, diagonal_co-cross_modal_co):
        cross_modal_by_drug[well_rows[index][0]].append(float(diff))
    cross_blocks = list(cross_modal_by_drug.values())
    cross_draws = []
    for _ in range(4000):
        selected = RNG.integers(0, len(cross_blocks), len(cross_blocks))
        cross_draws.append(np.mean([val for index in selected for val in cross_blocks[index]]))
    ordinary = np.mean(np.asarray(errors["other_wells_same_drug_dose"]), axis=1)
    personalized = np.mean(np.asarray(errors["drug_dose_plus_baseline_deviation"]), axis=1)
    improvement = ordinary-personalized
    by_drug = defaultdict(list)
    for index, diff in zip(evaluated, improvement):
        by_drug[well_rows[index][0]].append(float(diff))
    blocks = list(by_drug.values())
    cluster_draws = []
    for _ in range(4000):
        selected = RNG.integers(0, len(blocks), len(blocks))
        cluster_draws.append(np.mean([val for index in selected for val in blocks[index]]))
    result = {
        "source": "public GitLab ttm feature table; all 10 features fully present before and after drug",
        "well_grouping": "area ID fields 1,2,4; consistent with paper's 3 areas per well, but repository lacks formal ID schema",
        "complete_wells": len(well_rows),
        "evaluated_wells_with_at_least_2_other_wells_same_drug_dose": len(evaluated),
        "drug_dose_groups": len(groups),
        "drug_dose_group_size_min_median_max": list(map(float,
            [min(map(len, groups.values())), np.median([len(g) for g in groups.values()]),
             max(map(len, groups.values()))])),
        "targets": "well-mean log(post/baseline) for ten optical kinetic features",
        "personalization_baseline": "fixed ridge shrinkage; one within-drug-dose centered slope per matching feature, pooled across all training drug-dose groups; entire test well excluded",
        "mean_within_condition_baseline_to_log_response_slope": {
            feature: float(np.mean(np.asarray(slopes)[:, k])) for k, feature in enumerate(features)},
        "population_to_baseline_deviation_improvement": {
            "mean_absolute_log_ratio_error_reduction": float(np.mean(improvement)),
            "drug_cluster_bootstrap_95pct_interval": list(map(float,
                np.quantile(cluster_draws, [.025, .975]))),
            "fraction_wells_improved": float(np.mean(improvement > 0)),
        },
        "cross_modal_baseline_addition_for_contraction": {
            "diagonal_contraction_baseline_mae": float(np.mean(diagonal_co)),
            "all_AP_Ca_contraction_baselines_ridge_mae": float(np.mean(cross_modal_co)),
            "mean_well_error_reduction": float(np.mean(diagonal_co-cross_modal_co)),
            "drug_cluster_bootstrap_95pct_interval": list(map(float,
                np.quantile(cross_draws, [.025, .975]))),
            "fraction_wells_improved": float(np.mean(cross_modal_co < diagonal_co)),
            "model": "10 baseline optical features predict 3 contraction log-ratio residuals after drug-dose centering; fixed ridge 0.2*n, test well excluded",
        },
        "scores": scores,
        "cautions": ["This is leave-one-well-out within the same drug and dose; it does not test unseen drugs.",
                     "The population-effect baseline already knows drug identity and dose.",
                     "Area measurements from the same well are aggregated to prevent replicate leakage.",
                     "Optical contraction is a motion phenotype, not calibrated active tension or force."],
    }
    (HERE / "ttm_feature_prediction_pilot.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
