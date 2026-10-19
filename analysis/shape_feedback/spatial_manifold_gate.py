"""Does spatial sarcomere state improve future motion prediction?

Leave one substrate stiffness (two whole cells) out.  This is a mechanical
observability test only: the real movies do not contain paired AP/Ca signals.
Third-order features are statistical moments, not identified 3-body forces.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
HISTORY, HORIZON, STRIDE = 12, 10, 5
LOADS = (5, 15, 85)


def third_moment(v):
    v = v - np.mean(v)
    sd = np.std(v)
    return float(np.mean((v / max(sd, 1e-5)) ** 3))


def track_windows(data, prefix, load):
    mean = data[f"{prefix}_length_um"].astype(float)
    sarc = data[f"{prefix}_sarcomere_length_um"].astype(float)
    first = np.nanmedian(mean[:300])
    sb = np.nanmedian(sarc[:, :300], axis=1)
    if not np.isfinite(first) or first <= 0:
        return []
    mean /= first
    sarc = sarc / sb[:, None]
    rows = []
    for t in range(300, len(mean) - HORIZON, STRIDE):
        mh = mean[t - HISTORY:t]
        future = mean[t:t + HORIZON]
        history_s = sarc[:, t-HISTORY:t]
        valid_history = np.isfinite(history_s).all(axis=1)
        s = history_s[valid_history]
        future_s = sarc[valid_history, t:t+HORIZON]
        if len(s) < 8 or not np.isfinite(mh).all() or not np.isfinite(future).all():
            continue
        if np.any(np.isfinite(future_s).sum(axis=0) < 8):
            continue
        v = np.mean(np.diff(s[:, -4:], axis=1), axis=1)
        pos = s[:, -1]
        pair = np.asarray([
            np.std(pos), np.std(v),
            np.mean(np.diff(pos)**2),
        ])
        p = pos - np.mean(pos)
        q = v - np.mean(v)
        triple = np.asarray([
            third_moment(pos), third_moment(v),
            np.mean(p[:-2] * p[1:-1] * p[2:]) / max(np.std(pos)**3, 1e-6),
        ])
        future_disp = np.nanstd(future_s, axis=0)
        current_disp = float(np.std(pos))
        rows.append((np.r_[mh, np.log(load)], pair, triple,
                     future - mh[-1], mh[-1],
                     future_disp - current_disp, current_disp))
    return rows


def make_dataset():
    data = np.load(HERE / "sarcomere_tracks.npz")
    result = {}
    for load in LOADS:
        result[load] = {}
        cell_ids = sorted({k.split("_loi")[0] for k in data.files
                           if k.startswith(f"load{load}_cell") and k.endswith("_length_um")
                           and "_sarcomere_" not in k})
        for cell_id in cell_ids:
            rows = []
            for prefix in sorted({k.removesuffix("_length_um") for k in data.files
                                  if k.startswith(cell_id + "_loi")
                                  and k.endswith("_length_um")
                                  and "_sarcomere_" not in k}):
                rows += track_windows(data, prefix, load)
            result[load][cell_id] = rows
    return result


def pack(rows, model, target="mean"):
    x0 = np.asarray([r[0] for r in rows])
    pair = np.asarray([r[1] for r in rows])
    triple = np.asarray([r[2] for r in rows])
    if model == "mean":
        x = x0
    elif model == "pair":
        x = np.c_[x0, pair]
    elif model == "pair_duplicated":
        x = np.c_[x0, pair, pair]
    elif model == "triple":
        x = np.c_[x0, pair, triple]
    else:
        raise ValueError(model)
    if target == "mean":
        y = np.asarray([r[3] for r in rows])
        last = np.asarray([r[4] for r in rows])
    else:
        y = np.asarray([r[5] for r in rows])
        last = np.asarray([r[6] for r in rows])
    return x, y, last


def score_fold(dataset, hold_load, model, target):
    train = [r for load in LOADS if load != hold_load
             for rows in dataset[load].values() for r in rows]
    xtr, ytr, _ = pack(train, model, target)
    mu = np.mean(xtr, axis=0)
    sd = np.std(xtr, axis=0)
    sd = np.maximum(sd, 1e-4)
    fit = Ridge(alpha=10.0)
    fit.fit((xtr-mu)/sd, ytr)
    results = {}
    for cell, rows in dataset[hold_load].items():
        x, y, last = pack(rows, model, target)
        pred = fit.predict((x-mu)/sd)
        results[cell] = {
            "n_windows": len(rows),
            "mae_fraction": float(np.mean(np.abs(pred-y))),
            "mae_last_fraction": float(np.mean(np.abs(pred[:, -1]-y[:, -1]))),
        }
    return results


def main():
    dataset = make_dataset()
    models = ("persistence", "mean", "pair", "pair_duplicated", "triple")
    scores = {}
    for target in ("mean", "dispersion"):
        scores[target] = {}
        for load in LOADS:
            scores[target][str(load)] = {}
            for model in models:
                if model == "persistence":
                    col = 3 if target == "mean" else 5
                    scores[target][str(load)][model] = {
                        cell: {"n_windows": len(rows),
                               "mae_fraction": float(np.mean(np.abs(np.asarray([r[col] for r in rows])))),
                               "mae_last_fraction": float(np.mean(np.abs(np.asarray([r[col] for r in rows])[:, -1])))}
                        for cell, rows in dataset[load].items()}
                else:
                    scores[target][str(load)][model] = score_fold(dataset, load, model, target)
    cells = [(str(load), cell) for load in LOADS for cell in dataset[load]]
    summary = {target: {model: {
        "cell_macro_mae_fraction": float(np.mean([scores[target][l][model][c]["mae_fraction"] for l,c in cells])),
        "cell_macro_last_mae_fraction": float(np.mean([scores[target][l][model][c]["mae_last_fraction"] for l,c in cells])),
        "cells_better_than_mean": int(sum(scores[target][l][model][c]["mae_fraction"] <
                                            scores[target][l]["mean"][c]["mae_fraction"]
                                            for l,c in cells)) if model != "mean" else None,
    } for model in models} for target in ("mean", "dispersion")}
    rng = np.random.default_rng(20261018)
    contrasts = {}
    for target in ("mean", "dispersion"):
        contrasts[target] = {}
        for base, candidate in (("mean", "pair"), ("persistence", "pair"),
                                ("pair_duplicated", "triple")):
            gain = np.asarray([scores[target][l][base][c]["mae_fraction"] -
                               scores[target][l][candidate][c]["mae_fraction"]
                               for l,c in cells])
            resample = gain[rng.integers(0, len(gain), size=(10000, len(gain)))].mean(axis=1)
            contrasts[target][f"{candidate}_vs_{base}"] = {
                "mean_gain": float(np.mean(gain)),
                "cell_bootstrap95": np.quantile(resample, [.025,.975]).tolist(),
                "n_cells_improved": int(np.sum(gain > 0)),
            }
    report = {
        "source": "Härtter et al. Zenodo 17564384, six independent cells",
        "task": "12 observed frames -> next 10 normalized mean length and spatial dispersion frames",
        "validation": "leave-one-stiffness-out; both cells and all LOIs sealed for each fold",
        "alpha": 10.0, "history_frames": HISTORY,
        "future_frames": HORIZON, "stride": STRIDE,
        "feature_sets": {
            "mean": "mean length history + log substrate stiffness",
            "pair": "mean + spatial length SD, velocity SD, adjacent squared difference",
            "pair_duplicated": "mean + two copies of pair features; same added feature count as triple",
            "triple": "pair + length skewness, velocity skewness, adjacent triplet centered product",
        },
        "by_target_load_cell": scores, "summary": summary,
        "exploratory_cell_bootstrap_contrasts": contrasts,
        "interpretation_limit": "No paired AP/Ca or controlled acute stretch; third moments do not establish causal 3-body interactions; n=6 independent cells",
    }
    (HERE / "spatial_manifold_gate.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(2, 1, figsize=(8.3, 7.2), sharex=True, layout="constrained")
    xs = np.arange(len(cells))
    for ax, target in zip(axes, ("mean", "dispersion")):
        for j, model in enumerate(models):
            offset = (j - 2) * .14
            values = [scores[target][l][model][c]["mae_fraction"] for l,c in cells]
            ax.scatter(xs + offset, values, label=model, s=45)
        ax.set_ylabel(f"Future {target} MAE / initial length")
        ax.grid(axis="y", alpha=.2)
    axes[-1].set_xticks(xs, [f"{l} kPa\n{c.rsplit('cell',1)[-1]}" for l,c in cells])
    axes[0].set_title("Spatial information gate: six whole held-out cells")
    axes[0].legend(ncol=3, fontsize=8, frameon=False)
    fig.savefig(HERE / "spatial_manifold_gate.png", dpi=180)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
