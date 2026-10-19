"""Plot directly observed external-dataset summaries without cross-study fitting."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "Data"


def main() -> None:
    eth_path = next((ROOT / "CalciumImaging").rglob("SourceData_*.xlsx"))
    eth = openpyxl.load_workbook(eth_path, read_only=True, data_only=True)
    critical = list(eth["Critical Aspiration Pressures"].values)[1:]
    elife = openpyxl.load_workbook(ROOT / "elife-70988-fig5-data1-v1.xlsx",
                                    read_only=True, data_only=True)
    t50 = np.array([float(r[1]) for r in list(elife["Figure 5D"].values)[1:]
                    if isinstance(r[1], (float, int))])
    slope = np.array([float(r[1]) for r in list(elife["Figure 5E"].values)[1:]
                      if isinstance(r[1], (float, int))])
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), constrained_layout=True)
    force_levels = (2, 12.5, 25)
    styles = (("Control", "#3668a7"), ("CytoD", "#ba573b"), ("MargAc", "#548461"))
    for condition, color in styles:
        groups = [np.array([float(r[3]) for r in critical if r[0] == condition and r[2] == force])
                  for force in force_levels]
        medians = [np.median(group) for group in groups]
        low = [np.quantile(group, .25) for group in groups]
        high = [np.quantile(group, .75) for group in groups]
        axes[0].errorbar(force_levels, medians,
                         yerr=[np.array(medians)-low, np.array(high)-medians],
                         marker="o", capsize=3, linewidth=1.5, color=color, label=condition)
    axes[0].set_xlabel("FluidFM indentation (nN)")
    axes[0].set_ylabel("Critical aspiration (mbar)")
    axes[0].set_title("ETH: cell-level thresholds")
    axes[0].legend(frameon=False, fontsize=8)
    for ax, values, title, color in ((axes[1], t50, r"eLife: $T_{50}$", "#7565a6"),
                                     (axes[2], slope, r"eLife: slope $k$", "#bf8b38")):
        ax.hist(values, bins=8, color=color, alpha=.75)
        ax.axvline(np.median(values), color="black", linestyle="--", linewidth=1.5)
        ax.set_xlabel("mN/m")
        ax.set_ylabel("Patches")
        ax.set_title(title+f" (n={len(values)})")
    fig.savefig(HERE / "external_data_intake.png", dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    main()
