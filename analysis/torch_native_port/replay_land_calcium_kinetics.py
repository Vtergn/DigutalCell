"""Isolate Ca-to-tension kinetics with the same prescribed Ca transient.

This does not rerun the full 50-state closed loop under new parameters.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.io import loadmat

from .land2016_torch import land2016


HERE = Path(__file__).resolve().parent


def main() -> None:
    torch.set_num_threads(1)
    beat = np.load(HERE / "full_beat_validation_lambda100.npz")
    time_grid = beat["time"]
    ca_grid = beat["predicted_cai"]
    initial = loadmat(HERE / "prepace_state_v5.mat")["state"].ravel()[43:50]
    float64 = torch.float64
    cases = {
        "original_Land_intact": None,
        "only_ktm_unblock_0_021": {"ktm_unblock": 0.021},
        "only_nperm_2_036": {"nperm": 2.036},
        "published_ToRORd_Land_kinetics": {"ktm_unblock": 0.021, "nperm": 2.036},
    }
    report = {
        "calcium_source": "same lambda=1.00 coupled PyTorch whole-beat Ca trajectory",
        "published_parameter_source": (
            "Data/human_torland_reference/src/Initializer/Initializer.jl"
        ),
        "coupled_reference_tension_peak_kPa": float(
            beat["predicted_active_tension"].max()
        ),
        "cases": {},
    }
    for name, kinetics in cases.items():
        y = initial.copy()
        start = time.perf_counter()
        last_change = None
        warmup_beats = 0 if kinetics is None else 40
        success = True

        def rhs(t: float, state: np.ndarray) -> np.ndarray:
            cai = float(np.interp(t, time_grid, ca_grid))
            with torch.no_grad():
                dy, _, _, _ = land2016(
                    torch.from_numpy(state),
                    torch.tensor(cai, dtype=float64),
                    lam=1.0,
                    kinetics=kinetics,
                )
            return dy.numpy()

        for n in range(warmup_beats):
            sol = solve_ivp(
                rhs, (0.0, 1000.0), y, method="BDF",
                rtol=1e-8, atol=1e-10, max_step=5.0
            )
            success &= bool(sol.success)
            if not sol.success:
                break
            new_y = sol.y[:, -1]
            last_change = float(np.max(np.abs(new_y - y)))
            y = new_y
            if last_change < 1e-8:
                break
        sol = solve_ivp(
            rhs, (0.0, 1000.0), y, method="BDF",
            t_eval=time_grid, rtol=1e-8, atol=1e-10, max_step=2.0
        )
        success &= bool(sol.success)
        states = torch.from_numpy(sol.y.T)
        with torch.no_grad():
            _, _, ta, _ = land2016(
                states, torch.from_numpy(ca_grid), lam=1.0, kinetics=kinetics
            )
        active = ta.numpy()
        peak_i = int(np.argmax(active))
        case = {
            "success": success,
            "warmup_beats": n + 1 if warmup_beats else 0,
            "warmup_last_state_change_max": last_change,
            "tension_peak_kPa": float(active[peak_i]),
            "tension_time_to_peak_ms_from_stimulus": float(time_grid[peak_i]),
            "runtime_seconds": time.perf_counter() - start,
        }
        if kinetics is None:
            case["mae_vs_coupled_tension_kPa"] = float(
                np.mean(np.abs(active - beat["predicted_active_tension"]))
            )
        report["cases"][name] = case
        np.savez_compressed(
            HERE / f"land_calcium_replay_{name}.npz",
            time=time_grid, cai=ca_grid, active_tension=active
        )
    (HERE / "land_calcium_replay.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
