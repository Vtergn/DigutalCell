"""Score the controlled two-parameter rabbit reverse-challenge pilot.

The nine parameter-grid points are too few for model-selection claims. A
leave-one-variant-out ridge is included only as a diagnostic baseline.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from score_mct2024_reference import beats


OUT = Path(__file__).resolve().parent / "mct_feedback_audit"
DATA = OUT / "reverse_cohort"
METRICS = ("APD_above_minus67_mV_ms", "Ca_peak_uM", "gel_stress_peak_kPa")


def read_mat(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path) as f:
        return {key: f["a"][key][()].ravel() for key in f["a"]
                if isinstance(f["a"][key], h5py.Dataset)}


def ridge_loo(x: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    pred = np.zeros_like(y)
    for i in range(len(y)):
        train = np.arange(len(y)) != i
        mx = x[train].mean(axis=0)
        sx = np.maximum(x[train].std(axis=0), 1e-8)
        my = y[train].mean(axis=0)
        sy = np.maximum(y[train].std(axis=0), 1e-8)
        xx = (x[train] - mx) / sx
        yy = (y[train] - my) / sy
        weights = np.linalg.solve(xx.T @ xx + alpha * np.eye(x.shape[1]), xx.T @ yy)
        pred[i] = my + sy * (((x[i] - mx) / sx) @ weights)
    return pred


def main() -> None:
    rows = []
    for gain in (0.25, 0.75, 1.25):
        for scale in (0.9, 1.0, 1.1):
            prefix = f"g{round(gain*100):03d}_p{round(scale*100):03d}"
            case = {}
            for load in ("l02", "l10"):
                path = DATA / f"{prefix}_{load}.mat"
                if not path.exists():
                    raise FileNotFoundError(path)
                all_beats = beats(read_mat(path))
                case[load] = all_beats[-1]
                case[f"{load}_drift"] = {
                    k: float(all_beats[-1][k] - all_beats[0][k]) for k in METRICS
                }
            rows.append({"id": prefix, "mct_gain": gain, "pCa0_scale": scale,
                         "low": case["l02"], "high": case["l10"],
                         "low_beat17_to20_change": case["l02_drift"],
                         "high_beat17_to20_change": case["l10_drift"],
                         "delta": {k: case["l10"][k] - case["l02"][k] for k in METRICS}})
    x = np.array([[r["low"][k] for k in METRICS[:2]] for r in rows])
    y = np.array([[r["high"][k] for k in METRICS[:2]] for r in rows])
    pred = ridge_loo(x, y)
    # A strong, parameter-free control: assume the mechanical challenge has
    # no electrical effect, so the high-load APD/Ca equal low-load values.
    persistence = x.copy()
    err = np.mean(np.abs(pred - y), axis=0)
    err0 = np.mean(np.abs(persistence - y), axis=0)

    sd = np.maximum(x.std(axis=0), 1e-8)
    candidate_pairs = []
    for i, j in combinations(range(len(rows)), 2):
        candidate_pairs.append({
            "ids": [rows[i]["id"], rows[j]["id"]],
            "low_apd_difference_ms": float(abs(x[i, 0] - x[j, 0])),
            "low_ca_difference_uM": float(abs(x[i, 1] - x[j, 1])),
            "high_apd_difference_ms": float(abs(y[i, 0] - y[j, 0])),
            "high_ca_difference_uM": float(abs(y[i, 1] - y[j, 1])),
            "standardized_baseline_distance": float(np.linalg.norm((x[i] - x[j]) / sd)),
        })
    candidate_pairs.sort(key=lambda p: p["standardized_baseline_distance"])
    pair_40 = []
    for gain, scale in ((0.25, 1.1), (1.25, 1.0)):
        prefix = f"g{round(gain*100):03d}_p{round(scale*100):03d}"
        entry = {"id": prefix}
        for load in ("l02", "l10"):
            path = DATA / f"{prefix}_{load}_40.mat"
            if not path.exists():
                break
            last_four = beats(read_mat(path), final_beat=40)
            entry[load] = last_four[-1]
            entry[f"{load}_beat37_to40_change"] = {
                k: float(last_four[-1][k] - last_four[0][k]) for k in METRICS
            }
        if "l02" in entry and "l10" in entry:
            pair_40.append(entry)
    result = {
        "domain": "rabbit MCT2024 controlled synthetic variants",
        "n_variants": len(rows),
        "design": "3 MCT gains × 3 basal ICaL permeability scales × CL 0.02/0.10",
        "pacing_ms": 1000,
        "beats": 20,
        "rows": rows,
        "nearest_low_load_pairs": candidate_pairs[:5],
        "nearest_pair_40_beat_check": pair_40,
        "exploratory_loo_ridge_high_load_MAE": {k: float(v) for k, v in zip(METRICS[:2], err)},
        "persistence_high_load_MAE": {k: float(v) for k, v in zip(METRICS[:2], err0)},
        "interpretation": "Development pilot only. Grid points are not independent experimental cells; n=9 cannot validate a predictor or quantify alternans risk.",
    }
    (OUT / "reverse_cohort_score.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    fig, axs = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    marker_by_scale = {0.9: "o", 1.0: "s", 1.1: "^"}
    color_by_gain = {g: plt.cm.viridis((g-0.25)/1.0) for g in (0.25, 0.75, 1.25)}
    for ax, idx, unit in ((axs[0], 0, "APD above −67 mV (ms)"),
                          (axs[1], 1, "Ca peak (µM)")):
        for r in rows:
            a = r["low"][METRICS[idx]]
            b = r["high"][METRICS[idx]]
            ax.plot([0, 1], [a, b], marker=marker_by_scale[r["pCa0_scale"]], lw=1.2,
                    color=color_by_gain[r["mct_gain"]], alpha=0.7)
        ax.set_xticks([0, 1], ["CL 0.02", "CL 0.10"])
        ax.set_ylabel(unit)
        ax.grid(alpha=0.2)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=color_by_gain[g], lw=2,
                      label=f"MCT gain {g:g}") for g in color_by_gain]
    handles += [Line2D([0], [0], color="gray", marker=m, linestyle="none",
                       label=f"ICaL scale {s:g}") for s, m in marker_by_scale.items()]
    axs[0].legend(handles=handles, fontsize=7, frameon=False, ncol=2)
    fig.savefig(OUT / "reverse_cohort_response.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"n_variants": len(rows), "ridge_MAE": result["exploratory_loo_ridge_high_load_MAE"],
                      "persistence_MAE": result["persistence_high_load_MAE"],
                      "nearest_pair": candidate_pairs[0]}, indent=2))


if __name__ == "__main__":
    main()
