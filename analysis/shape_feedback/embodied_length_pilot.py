"""One-degree-of-freedom embodied cardiomyocyte closure.

Exploratory mechanical boundary law.  It is not a calibrated human-cell model.
The existing validated 50-state electrical/Ca/Land RHS is left unchanged;
the new 51st state is endogenous sarcomere length.  External stress rather
than length is imposed, and the solved length enters both SAC and Land.
"""
from pathlib import Path
import json
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.torch_native_port.native_bidirectional import native_rhs
from analysis.torch_native_port.land2016_torch import land2016

HERE = Path(__file__).resolve().parent


class EmbodiedRHS(torch.nn.Module):
    def __init__(self, sac_on=True, trpn_on=True, stiffness=220.0, viscosity=5500.0,
                 preload=0.0, sac_gain=1.0, land_beta=(2.3, -2.4)):
        super().__init__()
        self.sac_on = sac_on
        self.trpn_on = trpn_on
        self.stiffness = stiffness  # kPa per unit stretch
        self.viscosity = viscosity  # kPa ms per unit stretch
        self.preload = preload      # kPa; balances active resting tension
        self.sac_gain = sac_gain
        self.land_beta = land_beta

    def forward(self, state, time_ms, external_stress):
        x, lam = state[:50], state[50]
        _, total_tension, _, _ = land2016(x[43:50], 1000 * x[5], lam=lam,
                                          beta=self.land_beta)
        velocity = (self.preload + external_stress
                    - self.stiffness * (lam - 1) - total_tension) / self.viscosity
        dx, _ = native_rhs(x, time_ms, lam, trpn_on=self.trpn_on,
                           sac_on=self.sac_on, dlam_dt=velocity,
                           sac_lam=torch.clamp(lam, min=1.0),
                           sac_gain=None if self.sac_gain == 1.0 else self.sac_gain,
                           land_beta=self.land_beta)
        return torch.cat((dx, velocity.unsqueeze(0)))


def simulate(compiled, y0, pressure=0.0, onset=250.0, offset=350.0,
             duration=700.0, rtol=1e-6, atol=1e-8, max_step=2.0,
             land_beta=(2.3, -2.4), include_states=False):
    calls = 0

    def rhs(t, y):
        nonlocal calls
        calls += 1
        load = pressure if onset <= t < offset else 0.0
        with torch.no_grad():
            return compiled(torch.from_numpy(y), torch.tensor(t),
                            torch.tensor(load)).numpy()

    # Force jumps require segment boundaries for robust stiff integration.
    sections = sorted(set([0.0, onset, offset, duration]))
    all_t, all_y = [], []
    state = y0.copy()
    start = time.perf_counter()
    for a, b in zip(sections[:-1], sections[1:]):
        sol = solve_ivp(rhs, (a, b), state, method="BDF", rtol=rtol,
                        atol=atol, max_step=max_step, dense_output=False)
        if not sol.success:
            raise RuntimeError(f"BDF failed at {a}-{b} ms: {sol.message}")
        ts = sol.t if not all_t else sol.t[1:]
        ys = sol.y.T if not all_y else sol.y.T[1:]
        all_t.extend(ts)
        all_y.extend(ys)
        state = sol.y[:, -1]
    t = np.asarray(all_t)
    y = np.asarray(all_y)
    grid = np.arange(0, duration + 1)
    wave = np.column_stack([np.interp(grid, t, y[:, j]) for j in (0, 5, 50)])
    wave[:, 1] *= 1000  # mM to µM
    with torch.no_grad():
        _, total, active, passive = land2016(
            torch.from_numpy(y[:, 43:50]), torch.from_numpy(1000*y[:, 5]),
            lam=torch.from_numpy(y[:, 50]), beta=land_beta)
    tension_grid=np.interp(grid,t,total.numpy())
    result = dict(time=grid, wave=wave, tension=tension_grid,
                stress_time=t, lam=y[:, 50],
                active=active.numpy(), passive=passive.numpy(),
                runtime_s=time.perf_counter()-start, rhs_calls=calls)
    if include_states:
        result["state_grid"] = np.column_stack(
            [np.interp(grid, t, y[:, j]) for j in range(y.shape[1])])
    return result


def main():
    torch.set_num_threads(1)
    y50 = loadmat(ROOT / "analysis/torch_native_port/prepace_state_v5.mat")["state"].ravel()
    with torch.no_grad():
        _, resting_tension, _, _ = land2016(
            torch.from_numpy(y50[43:50]), torch.tensor(1000*y50[5], dtype=torch.float64))
    preload = float(resting_tension)
    y0 = np.r_[y50, 1.0]
    scenarios = {
        "free_feedback": dict(sac_on=True, trpn_on=True, pressure=0.0),
        "load_feedback": dict(sac_on=True, trpn_on=True, pressure=12.0),
        "load_no_sac": dict(sac_on=False, trpn_on=True, pressure=12.0),
        "load_no_mef": dict(sac_on=False, trpn_on=False, pressure=12.0),
    }
    output = {}
    for name, setting in scenarios.items():
        model = EmbodiedRHS(sac_on=setting["sac_on"], trpn_on=setting["trpn_on"],
                            preload=preload).double()
        exemplar = (torch.from_numpy(y0), torch.tensor(0., dtype=torch.float64),
                    torch.tensor(0., dtype=torch.float64))
        compiled = torch.jit.trace(model, exemplar, check_trace=False)
        output[name] = simulate(compiled, y0, pressure=setting["pressure"])
        print(name, "runtime_s", round(output[name]["runtime_s"], 2),
              "lambda_range", [round(float(output[name]["lam"].min()), 5),
                               round(float(output[name]["lam"].max()), 5)],
              flush=True)

    ref = output["free_feedback"]["wave"]
    metrics = {"resting_preload_kPa": preload,
               "mechanical_stiffness_kPa_per_stretch": 220.0,
               "mechanical_viscosity_kPa_ms_per_stretch": 5500.0,
               "external_pulse_kPa": 12.0,
               "scenarios": {}}
    for name, result in output.items():
        a = result["wave"]
        metrics["scenarios"][name] = {
            "lambda_min": float(result["lam"].min()),
            "lambda_max": float(result["lam"].max()),
            "V_delta_vs_free_max_mV": float(np.max(np.abs(a[:, 0]-ref[:, 0]))),
            "Ca_delta_vs_free_max_uM": float(np.max(np.abs(a[:, 1]-ref[:, 1]))),
            "peak_active_tension_kPa": float(result["active"].max()),
            "runtime_seconds": result["runtime_s"],
            "rhs_calls": result["rhs_calls"],
        }
    metrics["sac_effect_under_load"] = {
        "V_max_difference_mV": float(np.max(np.abs(
            output["load_feedback"]["wave"][:, 0]
            - output["load_no_sac"]["wave"][:, 0]))),
        "Ca_max_difference_uM": float(np.max(np.abs(
            output["load_feedback"]["wave"][:, 1]
            - output["load_no_sac"]["wave"][:, 1]))),
    }
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "embodied_length_pilot.json").write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(HERE / "embodied_length_pilot.npz", **{
        f"{name}_{key}": val for name, result in output.items()
        for key, val in result.items() if isinstance(val, np.ndarray)})
    fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True, layout="constrained")
    # The no-TnC scenario changes the baseline state and is retained only in
    # the numeric audit; omitting it here avoids a misleading visual ablation.
    colors = dict(free_feedback="#666666", load_feedback="#ba2d3b",
                  load_no_sac="#2455a4")
    for name, result in output.items():
        if name == "load_no_mef":
            continue
        t = result["time"]
        wave = result["wave"]
        axes[0].plot(result["stress_time"], result["lam"], label=name, color=colors[name])
        axes[1].plot(t, wave[:, 0], color=colors[name])
        axes[2].plot(t, wave[:, 1], color=colors[name])
        axes[3].plot(result["stress_time"], result["active"], color=colors[name])
    for ax, lab in zip(axes, ("Endogenous stretch λ", "Voltage (mV)",
                              "Cytosolic Ca (µM)", "Active tension (kPa)")):
        ax.set_ylabel(lab)
        ax.axvspan(250, 350, color="#aaaaaa", alpha=.12)
        ax.grid(alpha=.18)
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    axes[-1].set_xlabel("Time (ms)")
    fig.savefig(HERE / "embodied_length_pilot.png", dpi=180)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
