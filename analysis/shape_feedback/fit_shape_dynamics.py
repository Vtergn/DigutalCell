"""Small real-sarcomere forecast gate: train 5/85 kPa, seal 15 kPa.

This predicts 1D average sarcomere length from its recent history and known
substrate stiffness. No AP or Ca were recorded for these same cells, so this
is a mechanical trajectory model, not a complete electromechanical twin.
"""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
HISTORY, HORIZON, STRIDE = 12, 10, 5


def get_tracks(data, load):
    keys = sorted(k for k in data.files if k.startswith(f"load{load}_cell")
                  and k.endswith("_length_um")
                  and "_sarcomere_length_um" not in k)
    tracks = []
    for key in keys:
        length = data[key]
        # Initial 300 frames are a past calibration period for every cell.
        baseline = float(np.nanmedian(length[:300]))
        normalized = length / baseline
        tracks.append((key, normalized))
    return tracks


def windows(track, load, selection):
    n = len(track)
    start = HISTORY if selection == "calibrate" else max(300, HISTORY)
    end = n - HORIZON
    cutoff = int(.8 * n)
    rows = []
    for t in range(start, end, STRIDE):
        if selection == "train" and t >= cutoff:
            continue
        if selection == "valid" and t < cutoff:
            continue
        if selection == "calibrate" and t + HORIZON > 300:
            continue
        history = track[t-HISTORY:t]
        future = track[t:t+HORIZON]
        if not np.isfinite(history).all() or not np.isfinite(future).all():
            continue
        x = np.r_[history, np.log(load)]
        rows.append((x, future, history[-1], t))
    return rows


class ShapeMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(HISTORY + 1, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, HORIZON))

    def forward(self, x):
        return self.net(x)


def run(train_cell_limit=None):
    torch.set_num_threads(1)
    rng = np.random.default_rng(20261018)
    torch.manual_seed(20261018)
    source = np.load(HERE / "sarcomere_tracks.npz")
    tracks = {load: get_tracks(source, load) for load in (5, 15, 85)}
    split = {}
    for name, loads in (("train", (5, 85)), ("valid", (5, 85)), ("test", (15,))):
        entries = []
        for load in loads:
            for key, traj in tracks[load]:
                if name != "test" and train_cell_limit is not None:
                    if key.split("_loi")[0].split("cell")[-1] != train_cell_limit:
                        continue
                entries += windows(traj, load, name)
        split[name] = entries
    arrays = {}
    for name, rows in split.items():
        arrays[name] = (
            np.asarray([r[0] for r in rows], np.float32),
            np.asarray([r[1] for r in rows], np.float32),
            np.asarray([r[2] for r in rows], np.float32),
        )
    x_train, y_train, last_train = arrays["train"]
    # All models predict a change from the last observed length.
    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0) + 1e-5
    x_std[-1] = max(x_std[-1], .1)
    def standardize(x):
        return (x - x_mean) / x_std
    model = ShapeMLP()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    xt = torch.tensor(standardize(x_train))
    yt = torch.tensor((y_train-last_train[:, None]) * 100)
    xv = torch.tensor(standardize(arrays["valid"][0]))
    yv = torch.tensor((arrays["valid"][1]-arrays["valid"][2][:, None]) * 100)
    best, best_state, patience = float("inf"), None, 0
    for epoch in range(401):
        order = rng.permutation(len(xt))
        model.train()
        for group in np.array_split(order, max(1, len(order)//256)):
            out = model(xt[group])
            loss = torch.mean((out-yt[group])**2)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            score = float(torch.mean((model(xv)-yv)**2))
        if score < best - 1e-6:
            best = score
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= 50:
            break
    model.load_state_dict(best_state)
    model.eval()

    ridge = Ridge(alpha=10.0)
    ridge.fit(standardize(x_train), (y_train-last_train[:, None]) * 100)
    # A new cell supplies only its first 300 frames (~5 s) for a shared
    # contraction-amplitude calibration. Its later windows remain sealed.
    calibration_gains = {}
    calibration_window_count = 0
    for key, traj in tracks[15]:
        cell_id = key.split("_loi")[0]
        if cell_id not in calibration_gains:
            calibration_gains[cell_id] = {"rows": []}
        calibration_gains[cell_id]["rows"] += windows(traj, 15, "calibrate")
    for cell_id, item in calibration_gains.items():
        rows = item.pop("rows")
        calibration_window_count += len(rows)
        xc = np.asarray([r[0] for r in rows], np.float32)
        yc = np.asarray([r[1] for r in rows], np.float32)
        lc = np.asarray([r[2] for r in rows], np.float32)
        with torch.no_grad():
            nn_delta = model(torch.tensor(standardize(xc))).numpy()/100
        ridge_delta = ridge.predict(standardize(xc))/100
        target_delta = yc-lc[:, None]
        def gain(pred_delta):
            return float(np.clip(np.sum(pred_delta*target_delta)
                                 / max(np.sum(pred_delta**2), 1e-12), 0, 2))
        item.update(mlp=gain(nn_delta), ridge=gain(ridge_delta), windows=len(rows))
    test_cell_keys = []
    for key, traj in tracks[15]:
        test_cell_keys.extend([key.split("_loi")[0]] * len(windows(traj, 15, "test")))
    results, predictions = {}, {}
    for name, (x, y, last) in arrays.items():
        with torch.no_grad():
            nn_pred = last[:, None] + model(torch.tensor(standardize(x))).numpy()/100
        ridge_pred = last[:, None] + ridge.predict(standardize(x))/100
        persistence = np.repeat(last[:, None], HORIZON, axis=1)
        pred = {"persistence": persistence, "ridge": ridge_pred, "mlp": nn_pred}
        if name == "test":
            ridge_gains = np.asarray([calibration_gains[c]["ridge"] for c in test_cell_keys])
            nn_gains = np.asarray([calibration_gains[c]["mlp"] for c in test_cell_keys])
            pred["calibrated_ridge"] = last[:, None] + ridge_gains[:, None]*(ridge_pred-last[:, None])
            pred["calibrated_mlp"] = last[:, None] + nn_gains[:, None]*(nn_pred-last[:, None])
        predictions[name] = (y, pred)
        results[name] = {key: {
            "mae_full_fraction": float(np.mean(np.abs(value-y))),
            "mae_1step_fraction": float(np.mean(np.abs(value[:, 0]-y[:, 0]))),
            "mae_10step_fraction": float(np.mean(np.abs(value[:, -1]-y[:, -1]))),
        } for key, value in pred.items()}
        results[name]["windows"] = int(len(x))
        if name == "test":
            results["test_by_cell"] = {}
            for cell_id in sorted(set(test_cell_keys)):
                idx = np.asarray([c == cell_id for c in test_cell_keys])
                results["test_by_cell"][cell_id] = {
                    key: float(np.mean(np.abs(value[idx]-y[idx])))
                    for key, value in pred.items()}
    suffix = "_two_train" if train_cell_limit is not None else ""
    report = {
        "data": "Haertter et al. Zenodo 17564384; 5/85 kPa cells train, all 15 kPa cells sealed",
        "independent_training_cells": len({key.split("_loi")[0] for load in (5, 85)
                                            for key, _ in tracks[load]
                                            if train_cell_limit is None or
                                            key.split("_loi")[0].split("cell")[-1] == train_cell_limit}),
        "independent_test_cells": len(calibration_gains),
        "time_step_ms_approx": 16.4,
        "history_frames": HISTORY, "forecast_frames": HORIZON,
        "best_validation_mse_scaled": best,
        "training_epochs": epoch + 1,
        "test_cell_first_300_frames_calibration_windows": calibration_window_count,
        "test_cell_calibration_gains": calibration_gains,
        "results": results,
        "limit": "LOIs within a cell are correlated; no paired AP/Ca, no unseen protocol, no long free rollout",
    }
    (HERE / f"shape_dynamics_pilot{suffix}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.save({"model": model.state_dict(), "x_mean": x_mean, "x_std": x_std},
               HERE / f"shape_dynamics_pilot{suffix}.pt")
    y, pred = predictions["test"]
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    k = min(10, len(y)-1)
    x_ms = np.arange(HORIZON)*16.4
    ax.plot(x_ms, y[k], "ko-", label="observed 15 kPa", lw=1.8)
    for key, color in (("persistence", "#888888"), ("ridge", "#287fb8"),
                       ("mlp", "#ba2d3b"), ("calibrated_mlp", "#ef9c32")):
        ax.plot(x_ms, pred[key][k], "o-", label=key, color=color, lw=1.3)
    ax.set(xlabel="Forecast horizon (ms)", ylabel="Sarcomere length / initial median",
           title="Sealed cell at unseen 15 kPa load: one window")
    ax.legend(frameon=False)
    ax.grid(alpha=.2)
    fig.savefig(HERE / f"shape_dynamics_pilot{suffix}.png", dpi=170)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cell-limit", default=None,
                        help="Retain only this cell ID per training stiffness, e.g. 01")
    args = parser.parse_args()
    run(args.train_cell_limit)
