"""Exploratory CytoD threshold comparisons within acquisition dates.

Thresholds are only interpreted for author-labeled responsive cells. The
result therefore conditions on responsiveness and is not a causal estimate.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "Data" / "2022_pnas" / "2022_pnas" / "cytod"
GROUPS = (
    ("Piezo1", "mp1-mp2-cytod/piezocd_summary.csv", "mp1", "mp1cytod"),
    ("Piezo2", "mp1-mp2-cytod/piezocd_summary.csv", "mp2-dmso", "mp2-cytod"),
    ("TRAAK", "k2p-cytod/k2pcd_summary.csv", "mtraakdmso", "mtraakcytod"),
    ("TREK1", "k2p-cytod/k2pcd_summary.csv", "mtrek1dmso", "mtrek1cytod"),
)


def main() -> None:
    summary = []
    daily = []
    rng = np.random.default_rng(20260926)
    for label, filename, control, treatment in GROUPS:
        with (ROOT / filename).open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        matched_dates = sorted(
            {r["date"] for r in rows if r["construct"] == control}
            & {r["date"] for r in rows if r["construct"] == treatment}
        )
        usable = []
        for date in matched_dates:
            groups = []
            for construct in (control, treatment):
                chosen = [r for r in rows if r["date"] == date and r["construct"] == construct]
                responsive = [r for r in chosen if r["responds"] == "1" and float(r["wthresh"]) > 0]
                groups.append(np.asarray([float(r["wthresh"]) for r in responsive]))
                daily.append({
                    "channel": label, "date": date, "group": "control" if construct == control else "CytoD",
                    "total_cells": len(chosen), "responsive_cells": len(responsive),
                    "median_wthresh_fJ": float(np.median(groups[-1])) if len(groups[-1]) else float("nan"),
                })
            if all(len(group) for group in groups):
                usable.append((date, *groups))
        ratios = np.asarray([np.median(t) / np.median(c) for _, c, t in usable])
        observed = float(np.mean(np.log(ratios)))
        permuted = np.empty(10000)
        for rep in range(len(permuted)):
            differences = []
            for _, c, t in usable:
                combined = np.r_[c, t]
                scrambled = combined[rng.permutation(len(combined))]
                differences.append(np.log(np.median(scrambled[len(c):]) / np.median(scrambled[:len(c)])))
            permuted[rep] = np.mean(differences)
        summary.append({
            "channel": label, "matched_dates": len(matched_dates), "usable_dates": len(usable),
            "control_responsive_cells": sum(len(c) for _, c, _ in usable),
            "cytod_responsive_cells": sum(len(t) for _, _, t in usable),
            "geometric_mean_day_median_threshold_ratio": float(np.exp(observed)),
            "stratified_permutation_two_sided_p": float((1 + np.sum(np.abs(permuted) >= abs(observed))) / (len(permuted) + 1)),
        })
    for filename, values in (("cytod_by_day.csv", daily), ("cytod_summary.csv", summary)):
        with (HERE / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
