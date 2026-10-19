"""Plot component-level validation without implying patient-level agreement."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent


def main() -> None:
    beat = np.load(HERE / "full_beat_validation_lambda100.npz")
    t = beat["time"]
    v = beat["predicted_voltage"]
    ca = beat["predicted_cai"]
    original = np.load(HERE / "land_calcium_replay_original_Land_intact.npz")
    coupled = np.load(HERE / "land_calcium_replay_published_ToRORd_Land_kinetics.npz")
    only_ku = np.load(HERE / "land_calcium_replay_only_ktm_unblock_0_021.npz")
    only_n = np.load(HERE / "land_calcium_replay_only_nperm_2_036.npz")
    v_peak_t = t[np.argmax(v)]

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 250})
    fig, axs = plt.subplots(2, 2, figsize=(9.2, 6.2), constrained_layout=True)
    axs[0, 0].plot(t, v, color="#294a77", lw=1.6)
    axs[0, 0].set(xlim=(0, 400), ylabel="Membrane voltage (mV)",
                  title="A  Baseline electrical response")
    axs[0, 0].text(.98, .95, "APD90 = 265 ms", transform=axs[0, 0].transAxes,
                   ha="right", va="top")
    axs[0, 1].plot(t, ca, color="#15897f", lw=1.6)
    axs[0, 1].set(xlim=(0, 400), ylabel="Free cytosolic Ca (µM)",
                  title="B  Baseline calcium transient")
    axs[0, 1].text(.98, .95, "peak = 0.497 µM", transform=axs[0, 1].transAxes,
                   ha="right", va="top")
    traces = [
        (original, "Standalone Land defaults", "#b24a4a", "-"),
        (only_ku, "Only ku = 0.021", "#936b2b", "--"),
        (only_n, "Only nperm = 2.036", "#4d74a0", "--"),
        (coupled, "Published coupled kinetics", "#1d7b53", "-"),
    ]
    for arr, label, color, ls in traces:
        axs[1, 0].plot(t, arr["active_tension"], label=label, color=color,
                       ls=ls, lw=1.4)
    axs[1, 0].axvspan(v_peak_t + 147, v_peak_t + 172,
                      color="#829a9e", alpha=.17, label="in-vitro TTP window")
    axs[1, 0].set(xlim=(0, 420), xlabel="Time since stimulus (ms)",
                  ylabel="Active tension (kPa)",
                  title="C  Same Ca input, different Land kinetics")
    axs[1, 0].legend(frameon=False, fontsize=7, ncol=2)

    pts = [(float(arr["active_tension"].max()),
            float(t[np.argmax(arr["active_tension"])] - v_peak_t),
            label, color) for arr, label, color, _ in traces]
    axs[1, 1].axvspan(147, 172, color="#829a9e", alpha=.17)
    axs[1, 1].axhspan(15, 25, color="#829a9e", alpha=.17)
    for peak, ttp, label, color in pts:
        axs[1, 1].scatter(ttp, peak, s=55, color=color, zorder=3)
    axs[1, 1].annotate("Land defaults", (pts[0][1], pts[0][0]),
                       xytext=(5, -12), textcoords="offset points", fontsize=8)
    axs[1, 1].annotate("Coupled kinetics", (pts[3][1], pts[3][0]),
                       xytext=(-67, 7), textcoords="offset points", fontsize=8)
    axs[1, 1].set(xlabel="Tension time-to-peak after V peak (ms)",
                  ylabel="Peak active tension (kPa)",
                  title="D  Reference windows are descriptive")
    fig.suptitle("Component validation of the human closed-loop model",
                 fontweight="bold", fontsize=13)
    fig.text(.01, .014,
             "Panels C–D replay one model Ca trace. Shading: in-vitro windows compiled by BPSLand 2022; not a matched experiment.",
             fontsize=7, color="#555555")
    fig.savefig(HERE / "component_validation.png")
    fig.savefig(HERE / "component_validation.pdf")


if __name__ == "__main__":
    main()
