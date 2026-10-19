"""Schematic movie driven by solved endogenous length, never imposed λ."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Ellipse
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    data = np.load(HERE / "embodied_length_pilot.npz")
    t = data["load_feedback_time"]
    loaded = data["load_feedback_wave"]
    unloaded = data["free_feedback_wave"]
    fig, (cell_ax, trace_ax) = plt.subplots(1, 2, figsize=(9, 4),
                                           gridspec_kw={"width_ratios": [1, 1.5]},
                                           layout="constrained")
    cell_ax.set(aspect="equal", xlim=(-1.5, 1.5), ylim=(-1.0, 1.0))
    cell_ax.axis("off")
    cell_ax.set_title("Solved-length cell (schematic)")
    silhouette = Ellipse((0, 0), 2.1, .68, facecolor="#dd8096", edgecolor="#5b1730", lw=2)
    cell_ax.add_patch(silhouette)
    readout = cell_ax.text(0, -.72, "", ha="center", va="center", fontsize=9)
    trace_ax.plot(t, loaded[:, 2], color="#b2293d", lw=1.8, label="external load")
    trace_ax.plot(t, unloaded[:, 2], color="#777777", lw=1.4, label="free twitch")
    trace_ax.axvspan(250, 350, color="#aaaaaa", alpha=.2)
    dot, = trace_ax.plot([], [], "o", color="#b2293d", ms=7)
    trace_ax.set(xlim=(0, 700), ylim=(.94, 1.06), xlabel="Time (ms)",
                 ylabel="Solved sarcomere stretch λ")
    trace_ax.legend(frameon=False, fontsize=8)
    trace_ax.grid(alpha=.2)

    frames = np.arange(0, len(t), 7)
    def update(i):
        k = frames[i]
        lam, v, ca = loaded[k, 2], loaded[k, 0], loaded[k, 1]
        # Area-preserving ellipse is a rendering assumption, not a measured contour.
        silhouette.width = 2.1 * lam
        silhouette.height = .68 / lam
        ca_color = np.clip((ca-.07)/.45, 0, 1)
        silhouette.set_facecolor((.78-.35*ca_color, .48-.25*ca_color, .62+.12*ca_color))
        dot.set_data([t[k]], [lam])
        readout.set_text(f"λ={lam:.3f}    V={v:.1f} mV    Ca={ca:.3f} µM")
        return silhouette, dot, readout

    animation = FuncAnimation(fig, update, frames=len(frames), interval=45)
    animation.save(HERE / "embodied_length_schematic.gif", writer=PillowWriter(fps=20))
    plt.close(fig)


if __name__ == "__main__":
    main()
