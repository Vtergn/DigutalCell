"""Build an offline visual comparison from the existing 51-state cell model.

The morphology and channel coordinates in the viewer are schematic. Length,
voltage, free Ca, tension, and the displayed ionic current magnitudes come
from the native ODE solution. The experimental image and sarcomere track are
kept in their own time frame and are not fitted to the synthetic cell.
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from analysis.shape_feedback.embodied_length_pilot import EmbodiedRHS, simulate
from analysis.torch_native_port.electrical_gating_partial import gating_intermediates
from analysis.torch_native_port.land2016_torch import land2016


def compact(a, digits=5):
    return np.round(np.asarray(a, dtype=float), digits).tolist()


def make_scenarios():
    torch.set_num_threads(1)
    state = loadmat(ROOT / "analysis/torch_native_port/prepace_state_v5.mat")["state"].ravel()
    with torch.no_grad():
        _, resting, _, _ = land2016(
            torch.from_numpy(state[43:50]),
            torch.tensor(1000 * state[5], dtype=torch.float64),
        )
    y0 = np.r_[state, 1.0]
    settings = {
        "free": {"sac_on": True, "trpn_on": True, "pressure": 0.0,
                 "name": "无外加载荷 · 双反馈开启"},
        "loaded": {"sac_on": True, "trpn_on": True, "pressure": 12.0,
                   "name": "12 kPa 脉冲 · 双反馈开启"},
        "no_sac": {"sac_on": False, "trpn_on": True, "pressure": 12.0,
                   "name": "12 kPa · 关闭 SAC 电反馈"},
        "no_mef": {"sac_on": False, "trpn_on": False, "pressure": 12.0,
                   "name": "12 kPa · 关闭两条反馈"},
    }
    old = np.load(ROOT / "analysis/shape_feedback/embodied_length_pilot.npz")
    old_names = {"free": "free_feedback", "loaded": "load_feedback",
                 "no_sac": "load_no_sac", "no_mef": "load_no_mef"}
    out = {}
    for key, cfg in settings.items():
        model = EmbodiedRHS(
            sac_on=cfg["sac_on"], trpn_on=cfg["trpn_on"],
            preload=float(resting),
        ).double()
        exemplar = (
            torch.from_numpy(y0), torch.tensor(0., dtype=torch.float64),
            torch.tensor(0., dtype=torch.float64),
        )
        compiled = torch.jit.trace(model, exemplar, check_trace=False)
        result = simulate(
            compiled, y0, pressure=cfg["pressure"], include_states=True,
        )
        saved = old[f"{old_names[key]}_wave"]
        deviation = float(np.max(np.abs(result["wave"] - saved)))
        if deviation > 1e-5:
            raise AssertionError(f"{key}: ODE rerun differs from saved trace by {deviation}")
        state_grid = result["state_grid"]
        stretch_for_sac = (np.maximum(state_grid[:, 50], 1.0)
                           if cfg["sac_on"] else np.ones(len(state_grid)))
        with torch.no_grad():
            local = gating_intermediates(
                torch.from_numpy(state_grid[:, :43]),
                lam=torch.from_numpy(stretch_for_sac),
            )
        current = {
            "NaV": compact(local["INa"].numpy()),
            "CaV": compact(local["ICaL"].numpy()),
            "Kv": compact((local["IKr"] + local["IKs"] + local["IK1"]).numpy()),
            "SAC": compact(local["INs"].numpy()),
        }
        wave = result["wave"]
        # Exact ODE outputs; "active" is calculated at adaptive solver times.
        out[key] = {
            "name": cfg["name"], "sac_on": cfg["sac_on"],
            "trpn_on": cfg["trpn_on"], "external_stress_kPa": cfg["pressure"],
            "time_ms": compact(result["time"], 0),
            "V_mV": compact(wave[:, 0]),
            "Ca_uM": compact(wave[:, 1]),
            "lambda": compact(wave[:, 2], 6),
            "tension_kPa": compact(result["tension"]),
            "currents": current,
            "qc_saved_wave_max_abs_deviation": deviation,
        }
        print(key, f"{result['runtime_s']:.1f}s", "saved trace max Δ", deviation,
              flush=True)
    return out


def make_real():
    tracks = np.load(ROOT / "analysis/shape_feedback/sarcomere_tracks.npz")
    stem = "load5_cell01_loi0"
    t = tracks[f"{stem}_time_s"]
    length = tracks[f"{stem}_length_normalized"]
    keep = (t >= 1.2) & (t <= 3.0)
    # Preserve actual scale and actual time: no shape matching or fitting.
    return {
        "source": "Härtter et al. ACTN2-Citrine hiPSC-CM, 5 kPa substrate, cell 01, LOI 0",
        "time_s": compact(t[keep], 5),
        "length_normalized": compact(length[keep], 6),
        "time_window_s": [1.2, 3.0],
        "image_file": "real_5kpa_loi0_source.png",
        "image_note": "Original processed summary image extracted unchanged from 5kPa_cell_01.zip",
    }


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ROOT / "Data/sarcomere_shape_pilot/5kPa_cell_01.zip") as archive:
        image_name = next(name for name in archive.namelist()
                          if name.endswith("267_73_951_100_0.65/summary_loi.png"))
        (HERE / "real_5kpa_loi0_source.png").write_bytes(archive.read(image_name))
    payload = {"simulated": make_scenarios(), "experimental": make_real()}
    (HERE / "viewer_data.js").write_text(
        "window.CELL_VIEWER_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    qc = {
        "source_simulator": "51-state ToR-ORd/SAC-Land plus exploratory 1D mechanical closure",
        "source_image": str(ROOT / "Data/sarcomere_shape_pilot/5kPa_cell_01.zip"),
        "real_and_simulated_are_paired": False,
        "model_lambda_range_loaded": [
            min(payload["simulated"]["loaded"]["lambda"]),
            max(payload["simulated"]["loaded"]["lambda"]),
        ],
        "real_length_range_displayed": [
            min(payload["experimental"]["length_normalized"]),
            max(payload["experimental"]["length_normalized"]),
        ],
        "saved_wave_max_abs_deviation": {
            name: v["qc_saved_wave_max_abs_deviation"]
            for name, v in payload["simulated"].items()
        },
    }
    (HERE / "validation.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
