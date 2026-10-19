"""Score exploratory MCT gain ablation using the frozen 100-beat labels."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from score_mct2024_reference import beats
from score_mct_regime100 import label_beats


ROOT = Path(__file__).resolve().parent / "mct_feedback_audit" / "reverse_robust3" / "regime100"
IDS = ["train_016", "train_036", "test_006"]


def main() -> None:
    cohort = json.loads((ROOT / "score.json").read_text(encoding="utf-8"))
    native = {r["id"]: r for r in cohort["records"]}
    records = []
    for id_ in IDS:
        for gain in [0, .7]:
            path = ROOT / "gain_ablation" / f"{id_}_gain{round(gain * 10):02d}_l10_100.mat"
            if not path.exists():
                continue
            with h5py.File(path) as f:
                group = f["a"]
                case = {key: group[key][()].ravel() for key in group}
            score = label_beats(beats(case, final_beat=100, window=20))
            records.append({"id": id_, "gain": gain, **score})
        records.append({"id": id_, "gain": float(native[id_]["mct_gain"]),
                        **{k: v for k, v in native[id_].items() if k in [
                            "label", "apd_last10_median_ms", "ca_last10_median_uM",
                            "apd_odd_even_gap_ms", "ca_odd_even_gap_uM",
                            "apd_beats_81_100_ms", "ca_beats_81_100_uM"]}})
    output = {"protocol": "analysis/mct_regime100_gain_ablation_protocol.md",
              "n_expected_new": 6, "n_completed_new": len(records) - 3, "records": records}
    (ROOT / "gain_ablation_score.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(2, 3, figsize=(12, 5), sharex=True, constrained_layout=True)
    fig.suptitle("Feedback gain: blue 0 · orange 0.7 · green native", fontsize=11)
    for i, id_ in enumerate(IDS):
        for r in (x for x in records if x["id"] == id_):
            label = f"gain {r['gain']:.2f}: {r['label']}"
            axes[0, i].plot(np.arange(81, 101), r["apd_beats_81_100_ms"],
                            "o-", ms=2, label=label)
            axes[1, i].plot(np.arange(81, 101), r["ca_beats_81_100_uM"],
                            "o-", ms=2, label=label)
        axes[0, i].set_title(id_)
        axes[1, i].set_xlabel("Beat number")
        for ax in axes[:, i]:
            ax.grid(alpha=.2)
    axes[0, 0].set_ylabel("APD above −67 mV (ms)")
    axes[1, 0].set_ylabel("Ca peak (µM)")
    fig.savefig(ROOT / "gain_ablation_beats.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"completed": len(records) - 3,
                      "labels": [(r["id"], r["gain"], r["label"],
                                  round(r["apd_odd_even_gap_ms"], 1),
                                  round(r["ca_odd_even_gap_uM"], 3)) for r in records]}, indent=2))


if __name__ == "__main__":
    main()
