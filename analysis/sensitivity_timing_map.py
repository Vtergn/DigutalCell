"""Physical-coordinate perturbation map from Duke cell-level summaries.

The two within-date contrasts are shown in separate panels. WT Piezo1/2 are
cross-date context, not paired controls. Sensitivity uses responders' measured
work thresholds; right-censored cells are annotated separately, never placed
at an invented threshold coordinate.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "Data" / "2022_pnas" / "2022_pnas"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def median_point(rows: list[dict], label: str, date: str, contrast: str) -> dict:
    work = np.asarray([float(r["wthresh"]) for r in rows])
    delay = np.asarray([float(r["delay"]) for r in rows])
    return {"contrast": contrast, "date": date, "group": label, "responsive_cells": len(rows),
            "median_work_threshold_fJ": float(np.median(work)),
            "median_delay_ms": float(np.median(delay)),
            "sensitivity_minus_log10_work": float(-np.log10(np.median(work)))}


def main() -> None:
    wt = [r for r in read_csv(DATA / "bychannel" / "bychannel_summary.csv")
          if r["construct"] in {"mp1", "mp2"} and float(r["responds"]) == 1
          and abs(float(r["vstep"])+80) <= 5]
    chimeras = [r for r in read_csv(DATA / "chimera_experiments" / "capchims_summary.csv")
                if r["construct"] in {"chimA", "chimB"}]
    cyto = [r for r in read_csv(DATA / "cytod" / "mp1-mp2-cytod" / "piezocd_summary.csv")
            if r["construct"] in {"mp2-dmso", "mp2-cytod"} and r["responds"] == "1"
            and r["date"] in {"20200930", "20201013", "20201014"}]
    points = [median_point([r for r in wt if r["construct"] == name], name, "cross-date", "WT reference")
              for name in ("mp1", "mp2")]
    for date in ("20201108", "20201110"):
        for name in ("chimA", "chimB"):
            points.append(median_point([r for r in chimeras if r["date"] == date
                                        and r["construct"] == name], name, date,
                                       "same-date chimeras"))
    for date in ("20200930", "20201013", "20201014"):
        for name in ("mp2-dmso", "mp2-cytod"):
            points.append(median_point([r for r in cyto if r["date"] == date
                                        and r["construct"] == name], name, date,
                                       "same-date Piezo2 drug"))
    with (HERE / "sensitivity_timing_map_points.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    index = {(p["group"], p["date"]): p for p in points}
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.1), sharex=True, sharey=True,
                             layout="constrained")
    for ax in axes:
        for group, label, marker in (("mp1", "WT P1", "x"), ("mp2", "WT P2", "+")):
            p = index[group, "cross-date"]
            ax.scatter(p["sensitivity_minus_log10_work"], p["median_delay_ms"],
                       s=110, marker=marker, color="#a9a9a9", linewidth=2)
            ax.annotate(label+" (other dates)",
                        (p["sensitivity_minus_log10_work"], p["median_delay_ms"]),
                        xytext=(5, 5), textcoords="offset points", fontsize=8, color="#777777")
        ax.set_xlabel("Mechanical sensitivity, −log10(work threshold / fJ) →")
        ax.grid(alpha=.2)
        ax.set_ylim(-.6, 10.6)
    axes[0].set_ylabel("AFM-force-peak → current-peak delay (ms)")
    axes[0].set_title("Chimera summary: same-day B → A")
    for date, color in (("20201108", "#7353a1"), ("20201110", "#b18ccc")):
        b, a = index["chimB", date], index["chimA", date]
        for p, name in ((b, "B"), (a, "A")):
            axes[0].scatter(p["sensitivity_minus_log10_work"], p["median_delay_ms"],
                            color=color, s=65, edgecolor="black", zorder=3)
            offset = {( "20201108", "A"): (8, -20),
                      ( "20201110", "A"): (8, 4),
                      ( "20201108", "B"): (8, -14),
                      ( "20201110", "B"): (8, 4)}[date, name]
            axes[0].annotate(date[-2:]+" "+name,
                             (p["sensitivity_minus_log10_work"], p["median_delay_ms"]),
                             xytext=offset, textcoords="offset points", fontsize=8)
        axes[0].annotate("", (a["sensitivity_minus_log10_work"], a["median_delay_ms"]),
                         (b["sensitivity_minus_log10_work"], b["median_delay_ms"]),
                         arrowprops={"arrowstyle": "->", "color": color, "lw": 2})
    axes[0].text(.03, .97, "B molecular identity unverified\nWT references are cross-date",
                 transform=axes[0].transAxes, va="top", fontsize=8,
                 bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none"})

    axes[1].set_title("Piezo2: same-day DMSO → CytoD")
    for date, color in (("20200930", "#117c75"), ("20201013", "#42a89e"),
                        ("20201014", "#82c5be")):
        control, treated = index["mp2-dmso", date], index["mp2-cytod", date]
        for p, name in ((control, "D"), (treated, "C")):
            axes[1].scatter(p["sensitivity_minus_log10_work"], p["median_delay_ms"],
                            color=color, s=65, edgecolor="black", zorder=3)
            offset = {( "20200930", "D"): (6, 5),
                      ( "20200930", "C"): (6, 4),
                      ( "20201013", "D"): (6, -17),
                      ( "20201013", "C"): (6, -16),
                      ( "20201014", "D"): (6, 5),
                      ( "20201014", "C"): (6, 5)}[date, name]
            axes[1].annotate(date[-2:]+" "+name,
                             (p["sensitivity_minus_log10_work"], p["median_delay_ms"]),
                             xytext=offset, textcoords="offset points", fontsize=8)
        axes[1].annotate("", (treated["sensitivity_minus_log10_work"],
                              treated["median_delay_ms"]),
                         (control["sensitivity_minus_log10_work"], control["median_delay_ms"]),
                         arrowprops={"arrowstyle": "->", "color": color, "lw": 2})
    axes[1].text(.03, .97, "Responder-only x coordinate\nActive dates: 14/19 → 18/18 respond\nAll cell-ID blocks: CytoD before DMSO\nCensored ratio scenario: 0.28–0.50",
                 transform=axes[1].transAxes, va="top", fontsize=8,
                 bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none"})
    fig.suptitle("Sensitivity and response timing: physical coordinates, separate protocols")
    fig.savefig(HERE / "sensitivity_timing_perturbation_map.png", dpi=190)
    plt.close(fig)
    print("Wrote map and", len(points), "cell-median group/date coordinates")


if __name__ == "__main__":
    main()
