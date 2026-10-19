"""Target-blind, action-conditioned benchmark for the embodied 51-state cells.

Only the no-load and 12 kPa, 250--350 ms calibration trajectories are read
when constructing a prediction. The two future protocols are used for scoring
after all predictions have been frozen. The LTI comparator reconstructs the
unit-step response from the observed 100 ms pulse, then superposes target
onset and offset responses. It has no learned parameters or target access.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "blind_closed_loop"
COHORTS = [HERE / "embodied_cohort", HERE / "embodied_cohort_confirm"]
CHANNELS = ("V_mV", "Ca_uM", "T_kPa", "lambda")
SCALE = np.array([1.0, 0.005, 1.0, 0.01])


def shift_step(step: np.ndarray, start_ms: int, length: int) -> np.ndarray:
    """Zero before a causal step, without wrapping the array."""
    result = np.zeros((length, step.shape[1]), dtype=float)
    if start_ms < length:
        result[start_ms:] = step[:length-start_ms]
    return result


def reconstruct_step(calibration_delta: np.ndarray, pressure: float,
                     onset: int, offset: int) -> np.ndarray:
    """Invert y(t)=p*[S(t-onset)-S(t-offset)] for the observed time horizon."""
    width = offset - onset
    if width <= 0 or pressure == 0:
        raise ValueError("Calibration must be a nonzero positive-duration pulse")
    horizon = len(calibration_delta)-onset
    step = np.zeros((horizon, calibration_delta.shape[1]), dtype=float)
    for tau in range(horizon):
        step[tau] = calibration_delta[tau+onset]/pressure
        if tau >= width:
            step[tau] += step[tau-width]
    return step


def lti_predict(calibration_delta: np.ndarray, calibration: tuple,
                target: tuple) -> np.ndarray:
    _, cal_pressure, cal_onset, cal_offset = calibration
    _, target_pressure, target_onset, target_offset = target
    step = reconstruct_step(calibration_delta, cal_pressure,
                            round(cal_onset), round(cal_offset))
    n = len(calibration_delta)
    return target_pressure * (
        shift_step(step, round(target_onset), n)
        - shift_step(step, round(target_offset), n))


def pulse_transport(calibration_delta: np.ndarray, calibration: tuple,
                    target: tuple) -> np.ndarray:
    """Shift and amplitude-scale the observed pulse response without wrapping."""
    _, cal_pressure, cal_onset, _ = calibration
    _, target_pressure, target_onset, _ = target
    result = np.zeros_like(calibration_delta)
    shift = round(target_onset-cal_onset)
    if shift >= 0:
        result[shift:] = calibration_delta[:len(result)-shift]
    else:
        result[:shift] = calibration_delta[-shift:]
    return result * target_pressure/cal_pressure


def summarize_error(truth: np.ndarray, pred: np.ndarray) -> dict:
    absolute = np.abs(pred-truth)
    return {
        "MAE": dict(zip(CHANNELS, np.mean(absolute, axis=0).tolist())),
        "normalized_joint_MAE": float(np.mean(absolute/SCALE)),
        "normalized_joint_P90_time": float(np.quantile(np.mean(absolute/SCALE, axis=1), .9)),
        "V_max_abs_error_mV": float(absolute[:, 0].max()),
    }


def distribution(values: np.ndarray) -> dict:
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "P90": float(np.quantile(values, .9)),
            "CVaR90": float(values[values >= np.quantile(values, .9)].mean())}


def read_cohorts():
    items = []
    for cohort in COHORTS:
        manifest = json.loads((cohort / "manifest.json").read_text(encoding="utf-8"))
        protocols = [tuple(p) for p in manifest["protocols"]]
        for i in range(1, manifest["n_planned"]+1):
            with np.load(cohort / f"cell_{i:02d}.npz") as data:
                items.append({"cohort": cohort.name, "cell": i,
                              "theta": data["theta"].tolist(),
                              "wave": data["wave"], "protocols": protocols})
    return items


def evaluate():
    OUT.mkdir(exist_ok=True)
    rows = []
    examples = {}
    items = read_cohorts()
    for cell in items:
        wave = cell["wave"]
        no_load = wave[0]
        cal_delta = wave[1]-no_load
        protocols = cell["protocols"]
        assert wave.shape == (4, 601, 4)
        assert protocols[0][1] == 0 and protocols[1][1] != 0
        assert all(int(v) == v for p in protocols for v in p[2:])
        # Algebraic self-check: the inferred step must reproduce calibration.
        check = lti_predict(cal_delta, protocols[1], protocols[1])
        if np.max(np.abs(check-cal_delta)) > 1e-9:
            raise AssertionError("LTI step reconstruction failed")
        for target_index in (2, 3):
            target_protocol = protocols[target_index]
            predictions = {
                "zero_response": np.zeros_like(cal_delta),
                "shift_scale": pulse_transport(cal_delta, protocols[1],
                                                target_protocol),
                "causal_LTI": lti_predict(cal_delta, protocols[1],
                                          target_protocol),
            }
            truth = wave[target_index]-no_load  # Open sealed response for scoring.
            for name, prediction in predictions.items():
                score = summarize_error(truth, prediction)
                rows.append({"cohort": cell["cohort"], "cell": cell["cell"],
                             "target": protocols[target_index][0], "model": name,
                             "true_event_deltaV_over_40mV": bool(np.max(truth[:, 0]) > 40),
                             "pred_event_deltaV_over_40mV": bool(np.max(prediction[:, 0]) > 40),
                             **score})
            if ((cell["cohort"] == "embodied_cohort" and cell["cell"] == 2)
                    or (cell["cohort"] == "embodied_cohort_confirm" and cell["cell"] == 1)):
                examples[f"{cell['cohort']}_cell{cell['cell']}_{protocols[target_index][0]}"] = {
                    "truth": truth, **predictions}
    summary = {}
    for cohort in (x.name for x in COHORTS):
        summary[cohort] = {}
        for target in (items[0]["protocols"][2][0], items[0]["protocols"][3][0]):
            summary[cohort][target] = {}
            for model in ("zero_response", "shift_scale", "causal_LTI"):
                subset = [r for r in rows if r["cohort"] == cohort and
                          r["target"] == target and r["model"] == model]
                summary[cohort][target][model] = {
                    "n_cells": len(subset),
                    "V_MAE_mV": distribution(np.array([r["MAE"]["V_mV"] for r in subset])),
                    "Ca_MAE_uM": distribution(np.array([r["MAE"]["Ca_uM"] for r in subset])),
                    "T_MAE_kPa": distribution(np.array([r["MAE"]["T_kPa"] for r in subset])),
                    "lambda_MAE": distribution(np.array([r["MAE"]["lambda"] for r in subset])),
                    "joint_normalized_MAE": distribution(np.array([
                        r["normalized_joint_MAE"] for r in subset])),
                    "V_error_over_5mV": int(sum(r["MAE"]["V_mV"] > 5 for r in subset)),
                    "true_large_depolarization_events": int(sum(
                        r["true_event_deltaV_over_40mV"] for r in subset)),
                    "missed_large_depolarization_events": int(sum(
                        r["true_event_deltaV_over_40mV"] and
                        not r["pred_event_deltaV_over_40mV"] for r in subset)),
                }
    output = {"design": {
        "prediction_inputs": "Same-cell no-load and calibration V/Ca/T/lambda, plus target stress pulse definition",
        "sealed_outputs": "Two target V/Ca/T/lambda waveforms; read only for scoring",
        "model_training": "None: all three baselines are per-cell target-blind rules",
        "cell_groups": "Separate 12-cell source and 12-cell confirmation Latin-hypercube cohorts",
        "physics": "Exploratory 51-state embodied model; common nominal pre-paced initial state",
        "normalization": dict(zip(CHANNELS, SCALE.tolist())),
        "event_definition": "Descriptive threshold only: peak stress-induced delta V > 40 mV; not a biological arrhythmia diagnosis",
        "caveat": "Same-equation synthetic experiment. Stress law and mechanical parameters are not calibrated to paired human-cell records."},
        "summary": summary, "per_cell": rows}
    (OUT / "results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    plot_examples(examples)
    return output


def plot_examples(examples: dict) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(12, 9), sharex=True,
                             layout="constrained")
    chosen = ["embodied_cohort_cell2_target18_300_450",
              "embodied_cohort_confirm_cell1_target18_300_450"]
    colors = {"truth": "#202020", "zero_response": "#999999",
              "shift_scale": "#d5872f", "causal_LTI": "#3475a5"}
    for col, key in enumerate(chosen):
        entry = examples[key]
        for channel, ax in enumerate(axes[:, col]):
            for model in ("truth", "zero_response", "shift_scale", "causal_LTI"):
                ax.plot(np.arange(601), entry[model][:, channel],
                        color=colors[model], lw=1.4 if model == "truth" else 1.0,
                        label=model)
            ax.set_ylabel("Δ"+CHANNELS[channel])
            ax.grid(alpha=.2)
        axes[0, col].set_title(key.replace("_", " "))
        axes[-1, col].set_xlabel("Time (ms)")
    axes[0, 1].legend(ncol=2, fontsize=8)
    fig.suptitle("Target-blind stress-response prediction; synthetic closed-loop cells")
    fig.savefig(OUT / "blind_trajectory_examples.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    result = evaluate()
    for cohort, targets in result["summary"].items():
        for target, models in targets.items():
            print(cohort, target)
            for name, metrics in models.items():
                print(" ", name, "V mean/median/P90", metrics["V_MAE_mV"],
                      "joint mean", metrics["joint_normalized_MAE"]["mean"])
