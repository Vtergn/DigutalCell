"""Plot predeclared median-benefit future sweeps as visual diagnostics."""

from __future__ import annotations

import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dynamic_input_screen import HERE, ROOT, load_traces, onset_ms
from one_shot_work import read_metadata, first_calibration


def chosen_examples() -> dict[str, tuple[str, int]]:
    with (HERE / "dynamic_input_scores_date_heldout.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grouped = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        if row["model"] in {"one_shot_force_linear", "one_shot_work"}:
            grouped[(row["channel"], row["cell_id"])][int(row["test_sweep"])][row["model"]] = float(row["mae_pA"])
    output = {}
    for channel in ("mp1", "mp2"):
        differences = [(cell, float(np.mean([scores["one_shot_force_linear"]-scores["one_shot_work"]
                                             for scores in sweeps.values()])))
                       for (ch, cell), sweeps in grouped.items() if ch == channel]
        center = float(np.median([difference for _, difference in differences]))
        cell = min(differences, key=lambda value: abs(value[1]-center))[0]
        sweep = min(grouped[(channel, cell)], key=lambda number: abs(
            grouped[(channel, cell)][number]["one_shot_force_linear"]
            - grouped[(channel, cell)][number]["one_shot_work"] - center
        ))
        output[channel] = cell, sweep
    return output


def main() -> None:
    by_cell = read_metadata()
    calibration = {cell: first_calibration(rows) for cell, rows in by_cell.items()}
    traces, _ = load_traces(by_cell, calibration)
    examples = chosen_examples()
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), sharex="col", layout="constrained")
    for column, channel in enumerate(("mp1", "mp2")):
        cell, sweep = examples[channel]
        row = calibration[cell]
        cal = traces[(cell, int(row["sweep"]))]
        test = traces[(cell, sweep)]
        work_threshold = float(row["wthresh_fJ"])
        force_threshold = float(cal["F"][int(np.clip(round(float(row["threshold_time_ms"])
                                                         - (float(row["compression_end_ms"])-150)), 0, 149))])
        amplitude = max(float(np.max(cal["current"])), 10)
        predicted_force = amplitude * test["F"] / max(float(np.max(cal["F"])), 1) * (test["F"] >= force_threshold)
        train_alpha = [float(value["work_alpha"]) for other, value in calibration.items()
                       if value is not None and other != cell and value["channel"] == channel and value["date"] != row["date"]]
        alpha = float(np.median(train_alpha))
        cal_basis = np.where(cal["W"] >= work_threshold, np.maximum(cal["W"], 0)**alpha, 0)
        valid = (cal_basis > 0) & (cal["current"] > 0)
        gain = float(np.exp(np.median(np.log(cal["current"][valid])-np.log(cal_basis[valid]))))
        predicted_work = gain*np.where(test["W"] >= work_threshold, np.maximum(test["W"], 0)**alpha, 0)
        threshold_current = max(10.0, 5*float(row["stdev_pA"]))
        onset = onset_ms(test["current"], threshold_current, test["contact_ms"]-5)
        time = np.arange(350)
        ax = axes[0, column]
        ax.plot(time, test["current"], color="black", lw=1.6, label="observed current")
        ax.plot(time, predicted_force, color="#1768ac", lw=1.5, label="one-shot gated force")
        ax.plot(time, predicted_work, color="#c44900", lw=1.2, label="one-shot work power law")
        ax.axvline(test["contact_ms"], color="gray", ls=":", lw=1, label="estimated contact")
        if np.isfinite(onset):
            ax.axvline(onset, color="black", ls="--", lw=1, label="observed onset")
        ax.set_title(f"{'Piezo1' if channel == 'mp1' else 'Piezo2'} | {cell} | future sweep {sweep}")
        ax.set_ylabel("Current (pA)")
        ax.legend(fontsize=7, loc="upper left")
        ax = axes[1, column]
        ax.plot(time, test["F"]/max(float(np.max(cal["F"])), 1), color="#1768ac", label="force / calibration peak")
        ax.plot(time, np.maximum(test["W"], 0)/max(float(np.max(cal["W"])), 1), color="#c44900", label="work / calibration peak")
        ax.plot(time, test["P"]/max(float(np.max(np.maximum(cal["P"], 0))), .001), color="#668a2f", label="power / calibration peak")
        ax.axhline(force_threshold/max(float(np.max(cal["F"])), 1), color="#1768ac", ls="--", lw=1, label="calibration force threshold")
        ax.axvline(test["contact_ms"], color="gray", ls=":", lw=1)
        ax.set_xlabel("Time from compression start (ms)")
        ax.set_ylabel("Mechanical input / calibration scale")
        ax.legend(fontsize=7, loc="upper left")
    figure.suptitle("Representative cells selected by median force-vs-work MAE difference", fontsize=12)
    figure.savefig(HERE / "dynamic_input_examples.png", dpi=220)
    figure.savefig(HERE / "dynamic_input_examples.pdf")
    plt.close(figure)
    print(examples)


if __name__ == "__main__":
    main()
