"""Screen mechanical drivers with causal lagged ridge models on future Piezo sweeps.

All preprocessing uses mechanical signals only. Current from one calibration
sweep provides the cell amplitude; test current is used only for scoring.
Mechanical scaling uses that same calibration sweep, never the test peak.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

from one_shot_work import read_metadata, first_calibration


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "Data" / "2022_pnas" / "2022_pnas" / "bychannel"
LAGS_MS = (0, 1, 2, 5, 10, 20)
FEATURES = ("F", "d", "Fdot", "ddot", "P", "W")


def causal_mean(values: np.ndarray, width: int = 3) -> np.ndarray:
    total = np.cumsum(np.r_[0.0, values])
    indices = np.arange(len(values))
    starts = np.maximum(indices - width + 1, 0)
    return (total[indices + 1] - total[starts]) / (indices - starts + 1)


def backward_rate(values: np.ndarray, width: int = 3) -> np.ndarray:
    indices = np.arange(len(values))
    starts = np.maximum(indices - width, 0)
    return (values - values[starts]) / np.maximum(indices-starts, 1)


def extract(data: dict[str, np.ndarray], mask: np.ndarray, offset: float) -> tuple[dict, dict]:
    sample = {key: data[key][mask][:8750].reshape(350, 25).mean(axis=1)
              for key in ("force", "position", "deflection", "work", "i_blsub")}
    force_raw = sample["force"]
    force = causal_mean(force_raw)
    position = causal_mean(sample["position"])
    deflection = causal_mean(sample["deflection"])
    work = causal_mean(sample["work"])
    baseline = force_raw[:10]
    threshold = max(1.0, 3.0*float(np.std(baseline)), 0.01*float(np.max(force)))
    above = force > threshold
    contact = next((j for j in range(10, 148) if all(above[j:j+3])), 149)
    displacement = position - deflection
    indentation = np.maximum(displacement - displacement[contact], 0.0)
    indentation[:contact] = 0.0
    force_positive = np.maximum(force, 0.0)
    force_rate = backward_rate(force)
    indentation_rate = backward_rate(indentation)
    power = 0.001*force_positive*indentation_rate  # nN * nm/ms -> fJ/ms
    work_rate = backward_rate(work)
    compression = slice(max(contact+3, 10), 149)
    audit = {
        "contact_ms": contact,
        "contact_force_nN": float(force[contact]),
        "compression_power_workrate_correlation": float(np.corrcoef(power[compression], work_rate[compression])[0, 1]),
        "compression_negative_work_steps": int(np.sum(np.diff(work[:150]) < 0)),
        "full_negative_work_steps": int(np.sum(np.diff(work) < 0)),
    }
    uncorrected_current = np.abs(data["i_blsub"][mask][:8750]).reshape(350, 25).mean(axis=1) - offset
    precontact_baseline = float(np.median(uncorrected_current[:10]))
    current = uncorrected_current - precontact_baseline
    audit["precontact_baseline_pA"] = precontact_baseline
    audit["precontact_noise_pA"] = float(np.std(current[:10]))
    return {
        "F": force_positive, "d": indentation,
        "Fdot": force_rate, "ddot": indentation_rate,
        "P": power, "W": work,
        "current": current,
        "contact_ms": contact,
    }, audit


def load_traces(by_cell: dict, calibration: dict) -> tuple[dict[tuple[str, int], dict], list[dict]]:
    with (ROOT / "bychannel_summary.csv").open(newline="", encoding="utf-8-sig") as handle:
        selected = list(csv.DictReader(handle))
    paths = {path.name: path for path in ROOT.rglob("*_preprocessed.feather")}
    traces, audits = {}, []
    for chosen in selected:
        cell = chosen["uniqueID"]
        if calibration[cell] is None or chosen["construct"] not in {"mp1", "mp2"}:
            continue
        path = paths[Path(chosen["path"]).name]
        with path.with_name(path.name.replace("_preprocessed.feather", "_summary.csv")).open(newline="", encoding="utf-8-sig") as handle:
            sweep_meta = {int(row["sweep"]): row for row in csv.DictReader(handle)}
        table = feather.read_table(path, columns=["sweep", "ti", "force", "position", "deflection", "work", "i_blsub", "v"])
        data = {key: table[key].to_numpy() for key in table.column_names}
        for row in by_cell[cell]:
            sweep = int(row["sweep"])
            if sweep < int(calibration[cell]["sweep"]):
                continue
            start = float(row["compression_end_ms"]) - 150
            mask = ((data["sweep"] == sweep) & (data["ti"] >= start-1e-6) & (data["ti"] <= start+350+1e-6))
            if np.sum(mask) != 8751:
                raise ValueError(f"Unexpected window for {cell}, sweep {sweep}")
            trace, audit = extract(data, mask, float(sweep_meta[sweep]["offset"]))
            trace["voltage_mV"] = float(np.median(data["v"][mask]))
            traces[(cell, sweep)] = trace
            audits.append({"cell_id": cell, "channel": row["channel"], "sweep": sweep, **audit})
    return traces, audits


def scales(cal: dict) -> dict[str, float]:
    f = max(float(np.max(cal["F"])), 1.0)
    d = max(float(np.max(cal["d"])), 1.0)
    p = max(float(np.max(np.maximum(cal["P"], 0))), 0.001)
    w = max(float(np.max(cal["W"])), 1.0)
    return {"F": f, "d": d, "Fdot": f/10.0, "ddot": d/10.0, "P": p, "W": w}


def lag_matrix(trace: dict, cal_scales: dict, feature: str, lags: tuple[int, ...]) -> np.ndarray:
    signal = trace[feature] / cal_scales[feature]
    return np.column_stack([np.r_[np.repeat(signal[0], lag), signal[:len(signal)-lag]] if lag else signal
                            for lag in lags])


def design(trace: dict, cal_scales: dict, features: tuple[str, ...], lags: tuple[int, ...]) -> np.ndarray:
    return np.column_stack([lag_matrix(trace, cal_scales, feature, lags) for feature in features])


def fit_ridge(training: list[tuple[np.ndarray, np.ndarray]], test: np.ndarray, cell_count: int) -> np.ndarray:
    matrix = np.vstack([x for x, _ in training])
    target = np.concatenate([y for _, y in training])
    center = matrix.mean(axis=0)
    scale = np.maximum(matrix.std(axis=0), 1e-6)
    standardized = (matrix-center)/scale
    weights = 1.0 + 4.0*(target > .1)
    full = np.column_stack([np.ones(len(target)), standardized])
    penalty = np.diag([0.0] + [50.0*cell_count]*matrix.shape[1])
    coefficients = np.linalg.solve(full.T @ (weights[:, None]*full) + penalty,
                                   full.T @ (weights*target))
    query = np.column_stack([np.ones(len(test)), (test-center)/scale])
    return np.maximum(query@coefficients, 0.0)


def onset_ms(signal: np.ndarray, threshold: float, start: int = 0) -> float:
    above = signal > threshold
    return float(next((j for j in range(max(start, 0), 350-2) if np.all(above[j:j+3])), np.nan))


def score(actual: np.ndarray, predicted: np.ndarray, threshold: float, contact: int) -> dict[str, float]:
    observed_onset = onset_ms(actual, threshold, contact-5)
    predicted_onset = onset_ms(predicted, threshold, contact-5)
    has_event = np.isfinite(observed_onset)
    event = actual > threshold
    peak_start = max(contact-5, 0)
    actual_peak_index = peak_start + int(np.argmax(actual[peak_start:]))
    predicted_peak_index = peak_start + int(np.argmax(predicted[peak_start:]))
    actual_peak = float(actual[actual_peak_index])
    predicted_peak = float(predicted[predicted_peak_index])
    return {
        "actual_onset_ms": observed_onset,
        "predicted_onset_ms": predicted_onset,
        "actual_peak_time_ms": float(actual_peak_index) if has_event else float("nan"),
        "predicted_peak_time_ms": float(predicted_peak_index) if predicted_peak>threshold else float("nan"),
        "actual_peak_pA": actual_peak,
        "predicted_peak_pA": predicted_peak,
        "mae_pA": float(np.mean(np.abs(actual-predicted))),
        "event_mae_pA": float(np.mean(np.abs(actual[event]-predicted[event]))) if np.any(event) else float("nan"),
        "peak_abs_error_pA": float(abs(actual_peak-predicted_peak)),
        "peak_time_abs_error_ms": float(abs(actual_peak_index-predicted_peak_index)) if has_event and predicted_peak>threshold else float("nan"),
        "onset_abs_error_ms": float(abs(observed_onset-predicted_onset)) if has_event and np.isfinite(predicted_onset) else float("nan"),
        "onset_missed": int(has_event and not np.isfinite(predicted_onset)),
        "actual_has_event": int(has_event),
        "charge_abs_error_fC": float(abs(np.sum(np.maximum(actual, 0))-np.sum(predicted))),
        "false_positive_charge_fC": float(np.sum(predicted[~event])),
    }


MODELS = {**{f"{feature}_now": ((feature,), (0,)) for feature in FEATURES},
          **{f"{feature}_lag": ((feature,), LAGS_MS) for feature in FEATURES},
          "all_lag": (FEATURES, LAGS_MS)}


def main(holdout_date: bool = False) -> None:
    by_cell = read_metadata()
    calibration = {cell: first_calibration(rows) for cell, rows in by_cell.items()}
    traces, audits = load_traces(by_cell, calibration)
    cell_data = {}
    for cell, row in calibration.items():
        if row is None or row["channel"] not in {"mp1", "mp2"}:
            continue
        cal_sweep = int(row["sweep"])
        trace = traces[(cell, cal_sweep)]
        amplitude = max(float(np.max(trace["current"])), 10.0)
        cell_data[cell] = {"channel": row["channel"], "date": row["date"], "cal_sweep": cal_sweep,
                           "amplitude": amplitude, "scale": scales(trace),
                           "threshold_pA": max(10.0, 5.0*float(row["stdev_pA"]))}
    output = []
    for cell, props in cell_data.items():
        future = [r for r in by_cell[cell] if int(r["sweep"]) > props["cal_sweep"]
                  and abs(traces[(cell, int(r["sweep"]))]["voltage_mV"] - traces[(cell, props["cal_sweep"])]["voltage_mV"]) <= 5]
        if not future:
            continue
        train_cells = [other for other, entry in cell_data.items() if other != cell and entry["channel"] == props["channel"]
                       and (not holdout_date or entry["date"] != props["date"])]
        if len(train_cells) < 4:
            continue
        shared_alpha = float(np.median([float(calibration[other]["work_alpha"]) for other in train_cells]))
        threshold_work = float(calibration[cell]["wthresh_fJ"])
        cal_trace = traces[(cell, props["cal_sweep"])]
        cal_onset_ms = float(calibration[cell]["threshold_time_ms"]) - (float(calibration[cell]["compression_end_ms"]) - 150)
        force_threshold = float(cal_trace["F"][int(np.clip(round(cal_onset_ms), 0, 149))])
        cal_basis = np.where(cal_trace["W"] >= threshold_work, np.maximum(cal_trace["W"], 0)**shared_alpha, 0.0)
        gain_valid = (cal_basis > 0) & (cal_trace["current"] > 0)
        if not np.any(gain_valid):
            continue
        static_gain = float(np.exp(np.median(np.log(cal_trace["current"][gain_valid]) - np.log(cal_basis[gain_valid]))))
        for row in future:
            sweep = int(row["sweep"])
            test = traces[(cell, sweep)]
            static_basis = np.where(test["W"] >= threshold_work, np.maximum(test["W"], 0)**shared_alpha, 0.0)
            force_gate = test["F"] >= force_threshold
            baseline_predictions = (
                ("zero", np.zeros(350)),
                ("one_shot_work", static_gain*static_basis),
                ("one_shot_work_linear", props["amplitude"]*np.maximum(test["W"], 0)/props["scale"]["W"]*(test["W"] >= threshold_work)),
                ("one_shot_force_step", props["amplitude"]*force_gate),
                ("one_shot_force_linear", props["amplitude"]*test["F"]/props["scale"]["F"]*force_gate),
            )
            for baseline, predicted in baseline_predictions:
                output.append({"cell_id": cell, "channel": props["channel"], "date": props["date"], "test_sweep": sweep,
                               "model": baseline, **score(test["current"], predicted, props["threshold_pA"], test["contact_ms"])})
        for model, (features, lags) in MODELS.items():
            training = []
            for other in train_cells:
                entry = cell_data[other]
                trace = traces[(other, entry["cal_sweep"])]
                training.append((design(trace, entry["scale"], features, lags), trace["current"]/entry["amplitude"]))
            for row in future:
                sweep = int(row["sweep"])
                test = traces[(cell, sweep)]
                predicted = props["amplitude"]*fit_ridge(training, design(test, props["scale"], features, lags), len(train_cells))
                output.append({"cell_id": cell, "channel": props["channel"], "date": props["date"], "test_sweep": sweep,
                               "model": model, **score(test["current"], predicted, props["threshold_pA"], test["contact_ms"])})
                if model in {"F_now", "F_lag", "P_lag", "W_lag", "all_lag"}:
                    gated = {f"{model}_force_gate": predicted*(test["F"] >= force_threshold)}
                    if model == "all_lag":
                        gated["all_lag_work_gate"] = predicted*(test["W"] >= threshold_work)
                    for gated_model, gated_prediction in gated.items():
                        output.append({"cell_id": cell, "channel": props["channel"], "date": props["date"], "test_sweep": sweep,
                                       "model": gated_model, **score(test["current"], gated_prediction, props["threshold_pA"], test["contact_ms"])})
    suffix = "_date_heldout" if holdout_date else ""
    with (HERE / f"dynamic_input_scores{suffix}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    if not holdout_date:
        with (HERE / "dynamic_input_audit.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(audits[0]))
            writer.writeheader()
            writer.writerows(audits)
    for channel in ("mp1", "mp2"):
        print("date_heldout" if holdout_date else "cell_heldout", channel)
        gated_names = ("F_now_force_gate", "F_lag_force_gate", "P_lag_force_gate", "W_lag_force_gate", "all_lag_force_gate", "all_lag_work_gate")
        for model in ("zero", "one_shot_work", "one_shot_work_linear", "one_shot_force_step", "one_shot_force_linear", *MODELS, *gated_names):
            rows = [r for r in output if r["channel"] == channel and r["model"] == model]
            grouped = defaultdict(list)
            for row in rows:
                grouped[row["cell_id"]].append(row)
            print(model, len(grouped), len(rows), round(float(np.median([np.mean([r["mae_pA"] for r in values]) for values in grouped.values()])), 2))


if __name__ == "__main__":
    main()
    main(holdout_date=True)
