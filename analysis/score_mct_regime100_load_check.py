"""Score selected long low/middle-load traces with the frozen regime labels."""
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
        for load in [.02, .06]:
            path = ROOT / "load_check" / f"{id_}_l{round(load * 100):02d}_100.mat"
            if not path.exists():
                continue
            with h5py.File(path) as f:
                group = f["a"]
                case = {key: group[key][()].ravel() for key in group}
            records.append({"id": id_, "CL": load,
                            **label_beats(beats(case, final_beat=100, window=20))})
        records.append({"id": id_, "CL": .10,
                        **{k: v for k, v in native[id_].items() if k in [
                            "label", "apd_last10_median_ms", "ca_last10_median_uM",
                            "apd_odd_even_gap_ms", "ca_odd_even_gap_uM",
                            "apd_beats_81_100_ms", "ca_beats_81_100_uM"]}})
    output = {"protocol": "analysis/mct_regime100_load_check_protocol.md",
              "n_expected_new": 6, "n_completed_new": len(records) - 3, "records": records}
    (ROOT / "load_check_score.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(2, 3, figsize=(12, 5), sharex=True, constrained_layout=True)
    fig.suptitle("Load: blue CL 0.02 · orange CL 0.06 · green CL 0.10", fontsize=11)
    for i, id_ in enumerate(IDS):
        for r in (x for x in records if x["id"] == id_):
            label = f"CL {r['CL']:.2f}: {r['label']}"
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
    fig.savefig(ROOT / "load_check_beats.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"completed": len(records) - 3,
                      "labels": [(r["id"], r["CL"], r["label"],
                                  round(r["apd_odd_even_gap_ms"], 1),
                                  round(r["ca_odd_even_gap_uM"], 3)) for r in records]}, indent=2))


if __name__ == "__main__":
    main()
