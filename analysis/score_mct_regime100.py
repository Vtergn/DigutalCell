"""Score the frozen 100-beat rabbit MCT2024 regime viability gate."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from score_mct2024_reference import beats


ROOT = Path(__file__).resolve().parent / "mct_feedback_audit" / "reverse_robust3"
OUT = ROOT / "regime100"


def label_beats(rows: list[dict]) -> dict:
    apd = np.array([r["APD_above_minus67_mV_ms"] for r in rows], dtype=float)
    ca = np.array([r["Ca_peak_uM"] for r in rows], dtype=float)
    late_a, late_c = apd[-10:], ca[-10:]
    odd_a, even_a = late_a[::2], late_a[1::2]
    odd_c, even_c = late_c[::2], late_c[1::2]
    differences = np.diff(late_a)
    turns = int(np.sum(differences[:-1] * differences[1:] < 0))
    period2 = (
        abs(float(np.mean(odd_a) - np.mean(even_a))) >= 15
        and abs(float(np.mean(odd_c) - np.mean(even_c))) >= .03
        and max(np.ptp(odd_a), np.ptp(even_a)) <= 10
        and max(np.ptp(odd_c), np.ptp(even_c)) <= .02
        and turns >= 8
    )
    ca_relative_range = float(np.ptp(late_c) / max(abs(np.mean(late_c)), 1e-12))
    stable = (
        np.ptp(late_a) <= 5
        and ca_relative_range <= .01
        and abs(float(np.mean(apd[:10]) - np.mean(apd[10:]))) <= 5
        and abs(float(np.mean(ca[:10]) - np.mean(ca[10:]))) <= .02
    )
    label = "period2_candidate" if period2 else "stable_1to1" if stable else "drifting_or_complex"
    return {
        "label": label,
        "apd_last10_median_ms": float(np.median(late_a)),
        "ca_last10_median_uM": float(np.median(late_c)),
        "apd_last10_range_ms": float(np.ptp(late_a)),
        "ca_last10_relative_range": ca_relative_range,
        "apd_window_mean_shift_ms": float(np.mean(apd[10:]) - np.mean(apd[:10])),
        "ca_window_mean_shift_uM": float(np.mean(ca[10:]) - np.mean(ca[:10])),
        "apd_odd_even_gap_ms": float(abs(np.mean(odd_a) - np.mean(even_a))),
        "ca_odd_even_gap_uM": float(abs(np.mean(odd_c) - np.mean(even_c))),
        "apd_parity_range_ms": [float(np.ptp(odd_a)), float(np.ptp(even_a))],
        "ca_parity_range_uM": [float(np.ptp(odd_c)), float(np.ptp(even_c))],
        "apd_alternating_turns_of_8": turns,
        "apd_beats_81_100_ms": apd.tolist(),
        "ca_beats_81_100_uM": ca.tolist(),
        "apd_parity_mean_ms": [float(np.mean(odd_a)), float(np.mean(even_a))] if period2 else None,
        "ca_parity_mean_uM": [float(np.mean(odd_c)), float(np.mean(even_c))] if period2 else None,
    }


def main() -> None:
    OUT.mkdir(exist_ok=True)
    with (ROOT / "design.csv").open(newline="", encoding="utf-8") as f:
        design = list(csv.DictReader(f))
    records = []
    missing = []
    for row in design:
        path = OUT / "native" / f"{row['id']}_l10_100.mat"
        if not path.exists():
            missing.append(row["id"])
            continue
        with h5py.File(path) as f:
            group = f["a"]
            case = {key: group[key][()].ravel() for key in group}
        try:
            scored = label_beats(beats(case, final_beat=100, window=20))
            beat40_long = beats(case, final_beat=40, window=1)[0]
        except Exception as exc:
            raise RuntimeError(f"Beat scoring failed for {row['id']}: {path}") from exc
        scored["apd_beat40_from_long_ms"] = beat40_long["APD_above_minus67_mV_ms"]
        scored["ca_beat40_from_long_uM"] = beat40_long["Ca_peak_uM"]
        short_path = ROOT / "native_cohort" / f"{row['id']}_l10.mat"
        if short_path.exists():
            with h5py.File(short_path) as f:
                short_group = f["a"]
                short_case = {key: short_group[key][()].ravel() for key in short_group}
            beat40 = beats(short_case, final_beat=40, window=1)[0]
            scored["apd_beat40_ms"] = beat40["APD_above_minus67_mV_ms"]
            scored["ca_beat40_uM"] = beat40["Ca_peak_uM"]
            scored["apd_short_minus_long_at_beat40_ms"] = (scored["apd_beat40_ms"]
                                                            - scored["apd_beat40_from_long_ms"])
            scored["ca_short_minus_long_at_beat40_uM"] = (scored["ca_beat40_uM"]
                                                           - scored["ca_beat40_from_long_uM"])
            scored["apd_late_median_minus_beat40_ms"] = (scored["apd_last10_median_ms"]
                                                          - scored["apd_beat40_ms"])
            scored["ca_late_median_minus_beat40_uM"] = (scored["ca_last10_median_uM"]
                                                         - scored["ca_beat40_uM"])
        tight_path = (ROOT / "train036_high100_tight.mat" if row["id"] == "train_036"
                      else OUT / "tight" / f"{row['id']}_l10_100_tight.mat")
        if tight_path.exists():
            with h5py.File(tight_path) as f:
                tight_group = f["a"]
                tight_case = {key: tight_group[key][()].ravel() for key in tight_group}
            tight_score = label_beats(beats(tight_case, final_beat=100, window=20))
            scored["tight_label"] = tight_score["label"]
            scored["tight_apd_odd_even_gap_ms"] = tight_score["apd_odd_even_gap_ms"]
            scored["tight_ca_odd_even_gap_uM"] = tight_score["ca_odd_even_gap_uM"]
        records.append({**row, **scored})
    counts = {split: dict(Counter(r["label"] for r in records if r["split"] == split))
              for split in ["train", "test"]}
    complete = len(records) == len(design) and not missing
    gate_passed = complete and all([
        counts["train"].get("period2_candidate", 0) >= 5,
        counts["test"].get("period2_candidate", 0) >= 2,
        counts["train"].get("stable_1to1", 0) >= 5,
        counts["test"].get("stable_1to1", 0) >= 5,
    ])
    summary = {"protocol": "analysis/mct_regime100_protocol.md", "n_expected": len(design),
               "n_completed": len(records), "missing_ids": missing, "counts": counts,
               "gate_passed": gate_passed, "records": records}
    (OUT / "score.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    colors = {"stable_1to1": "#287a5e", "drifting_or_complex": "#e69f00",
              "period2_candidate": "#aa3377"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for split, marker in [("train", "o"), ("test", "^")]:
        for label, color in colors.items():
            subset = [r for r in records if r["split"] == split and r["label"] == label]
            if not subset:
                continue
            axes[0].scatter([float(r["mct_gain"]) for r in subset],
                            [float(r["serca_scale"]) for r in subset],
                            color=color, marker=marker, s=55, alpha=.85, label=f"{label}, {split}")
            axes[1].scatter([r["apd_last10_median_ms"] for r in subset],
                            [r["ca_last10_median_uM"] for r in subset],
                            color=color, marker=marker, s=55, alpha=.85)
    axes[0].set(xlabel="MCT feedback gain", ylabel="SERCA scale", title="100-beat high-load regime map")
    axes[1].set(xlabel="Last-10 APD median (ms)", ylabel="Last-10 Ca peak median (µM)",
                title="Late response magnitude")
    axes[0].legend(fontsize=6, frameon=False, loc="best")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(OUT / "regime_map.png", dpi=180)
    plt.close(fig)
    examples = []
    for label in ["stable_1to1", "drifting_or_complex", "period2_candidate"]:
        candidates = [r for r in records if r["label"] == label]
        if candidates:
            examples.append(candidates[0])
    fig, axes = plt.subplots(2, len(examples), figsize=(4 * len(examples), 5),
                             sharex=True, constrained_layout=True, squeeze=False)
    for i, record in enumerate(examples):
        beat_numbers = np.arange(81, 101)
        axes[0, i].plot(beat_numbers, record["apd_beats_81_100_ms"], "o-", ms=3)
        axes[1, i].plot(beat_numbers, record["ca_beats_81_100_uM"], "o-", ms=3)
        axes[0, i].set_title(f"{record['id']}: {record['label'].replace('_', ' ')}", fontsize=9)
        axes[1, i].set_xlabel("Beat number")
        for ax in axes[:, i]:
            ax.grid(alpha=.2)
    axes[0, 0].set_ylabel("APD above −67 mV (ms)")
    axes[1, 0].set_ylabel("Ca peak (µM)")
    fig.savefig(OUT / "late_beat_examples.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for label, color in colors.items():
        subset = [r for r in records if r["label"] == label]
        if not subset:
            continue
        axes[0].scatter([r["apd_beat40_ms"] for r in subset],
                        [r["apd_last10_median_ms"] for r in subset],
                        color=color, s=35, alpha=.8, label=label)
        axes[1].scatter([r["ca_beat40_uM"] for r in subset],
                        [r["ca_last10_median_uM"] for r in subset],
                        color=color, s=35, alpha=.8)
    for ax in axes:
        low = min(ax.get_xlim()[0], ax.get_ylim()[0])
        high = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([low, high], [low, high], "k--", lw=1, alpha=.6)
        ax.grid(alpha=.2)
    axes[0].set(xlabel="Beat-40 APD (ms)", ylabel="Late APD median (ms)")
    axes[1].set(xlabel="Beat-40 Ca peak (µM)", ylabel="Late Ca peak median (µM)")
    axes[0].legend(fontsize=7, frameon=False)
    fig.savefig(OUT / "beat40_vs_late.png", dpi=180)
    plt.close(fig)
    print(json.dumps({k: summary[k] for k in ["n_completed", "missing_ids", "counts", "gate_passed"]}, indent=2))


if __name__ == "__main__":
    main()
