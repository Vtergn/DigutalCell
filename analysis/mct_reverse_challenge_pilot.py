"""Small rabbit closed-loop mechanical-challenge pilot.

The four feedback gains are controlled model variants, not sampled biological
cells. All cases start from the same supplied initial state and are paced for
20 beats at 1000 ms. This script scores only the last four beats and does not
fit or validate a predictive AI model.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from score_mct2024_reference import beats, load_case


OUT = Path(__file__).resolve().parent / "mct_feedback_audit"
CASES = {
    "off": 0.0,
    "half": 0.5,
    "on": 1.0,
    "gain15": 1.5,
}
METRICS = ("Ca_peak_uM", "Ca_amplitude_uM", "APD_above_minus67_mV_ms",
           "gel_stress_peak_kPa", "pCa_multiplier_peak")


def main() -> None:
    rows = []
    traces = {}
    for tag, gain in CASES.items():
        for load in ("lo", "hi"):
            key = f"{load}_{tag}"
            case = load_case(key)
            traces[key] = case
            last_four = beats(case)
            row = {"gain": gain, "load": load, "case": key,
                   "beats_scored": [17, 18, 19, 20]}
            for metric in METRICS:
                vals = np.array([b[metric] for b in last_four], float)
                row[metric] = float(vals[-1])
                row[f"{metric}_beat17_to20_change"] = float(vals[-1] - vals[0])
            rows.append(row)

    by_key = {row["case"]: row for row in rows}
    contrasts = []
    for tag, gain in CASES.items():
        lo, hi = by_key[f"lo_{tag}"], by_key[f"hi_{tag}"]
        entry = {"gain": gain}
        for metric in METRICS:
            entry[f"delta_{metric}"] = hi[metric] - lo[metric]
            entry[f"interaction_{metric}"] = (
                hi[metric] - lo[metric]
                - by_key["hi_off"][metric] + by_key["lo_off"][metric]
            )
        contrasts.append(entry)

    result = {
        "domain": "rabbit MCT2024 Shannon-Bers/Negroni model",
        "design": "CL 0.02 versus 0.10 crossed with MCT gain 0, 0.5, 1, 1.5",
        "initial_state": "common author-supplied yfinal_2000.mat",
        "pacing_ms": 1000,
        "beats": 20,
        "apd_metric": "last-beat milliseconds above -67 mV; not APD90",
        "rows": rows,
        "paired_challenge_contrasts": contrasts,
        "limitations": [
            "Only MCT gain varies; these are controlled model variants, not independent cells.",
            "CL is gel crosslinker concentration, a model load setting rather than a measured force trajectory.",
            "No parameter inference, held-out prediction, alternans evaluation, or human model transfer is tested.",
        ],
    }
    (OUT / "reverse_challenge_pilot.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fig, axs = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, metric, ylabel in (
        (axs[0, 0], "Ca_peak_uM", "Calcium peak (µM)"),
        (axs[0, 1], "APD_above_minus67_mV_ms", "APD above −67 mV (ms)"),
    ):
        for tag, gain in CASES.items():
            vals = [by_key[f"{load}_{tag}"][metric] for load in ("lo", "hi")]
            ax.plot([0, 1], vals, marker="o", label=f"MCT gain {gain:g}")
        ax.set_xticks([0, 1], ["CL 0.02", "CL 0.10"])
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
    axs[0, 0].legend(frameon=False, fontsize=8)
    for ax, key, ylabel in (
        (axs[1, 0], "Vm_mV", "Voltage (mV)"),
        (axs[1, 1], "Cai_uM", "Calcium (µM)"),
    ):
        for tag, gain in CASES.items():
            case = traces[f"hi_{tag}"]
            t = case["t_ms"]
            sel = (t >= 19000) & (t < 20000)
            ax.plot(t[sel] - 19000, case[key][sel], label=f"MCT gain {gain:g}", lw=1.1)
        ax.set_xlabel("Time within beat 20 (ms)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
    fig.savefig(OUT / "reverse_challenge_pilot.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"paired_challenge_contrasts": contrasts}, indent=2))


if __name__ == "__main__":
    main()
