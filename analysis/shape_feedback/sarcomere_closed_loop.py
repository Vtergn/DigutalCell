"""Human electrical/Ca model closed around Härtter's coupled sarcomeres.

The upstream sarcomere force law is used directly from the MIT-licensed
SarcomereModel package in external/SarcomereModel. Its abstract sine activation
is replaced by Land thin-filament availability. The length conversion is an
explicit, uncalibrated bridge. This first integration uses the deterministic
drift of the published stochastic model; stochastic replay is separate.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
from scipy.integrate import solve_ivp
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external/SarcomereModel/src"))
from model import Model, ModelParams, SimParams
from analysis.torch_native_port.native_bidirectional import native_rhs
from analysis.torch_native_port.land2016_torch import land2016


def sarcomere_params(n=20, load_stiffness=2.0):
    """Parameters from the upstream quickstart, without random forcing."""
    return ModelParams(N=n, mu=0.006, eta=[0.0, 0.0],
                       k_l=load_stiffness, k_s=[3.5, 3.0], u_s=0.1,
                       o_s=None, f_s=[-0.77, -0.058, 0.1, 0.012], h=0.0)


def source_model(n=20, load_stiffness=2.0):
    p = sarcomere_params(n, load_stiffness)
    s = SimParams(t1=0.7, dt=0.002, a_mode="sine",
                  a=[1.0, 0.0, 1.0], p=0.0, y0=np.zeros(2*n))
    return Model(p, s, verbose=False)


class ElectricalRHS(torch.nn.Module):
    def __init__(self, sac_on=True, trpn_on=True):
        super().__init__()
        self.sac_on = sac_on
        self.trpn_on = trpn_on

    def forward(self, y, t, lam, dlam_dt):
        dy, _ = native_rhs(y, t, lam, dlam_dt=dlam_dt,
                           sac_on=self.sac_on, trpn_on=self.trpn_on,
                           sac_lam=torch.clamp(lam, min=1.0))
        return dy


class ClosedLoop:
    def __init__(self, n=20, stiffness=2.0, sac_on=True, trpn_on=True,
                 length_scale=0.3, activation_scale=0.2,
                 electrical_feedback=True, calcium_motor=True):
        self.n = n
        self.sarcomeres = source_model(n, stiffness)
        self.length_scale = length_scale
        self.activation_scale = activation_scale
        self.electrical_feedback = electrical_feedback
        self.calcium_motor = calcium_motor
        e = ElectricalRHS(sac_on=sac_on, trpn_on=trpn_on).double()
        sample = (torch.from_numpy(np.zeros(50)),
                  torch.tensor(0., dtype=torch.float64),
                  torch.tensor(1., dtype=torch.float64),
                  torch.tensor(0., dtype=torch.float64))
        self.electrical = torch.jit.trace(e, sample, check_trace=False)
        self.calls = 0

    def fields(self, y):
        v = y[50:50+self.n]
        x = y[50+self.n:50+2*self.n]
        # Upstream x increases during shortening; it reports length as -x.
        lam = 1.0 - self.length_scale * float(np.mean(x))
        dlam_dt = -self.length_scale * float(np.mean(v)) / 1000.0
        return v, x, lam, dlam_dt

    def rhs(self, t_ms, y, prestrain=0.0):
        self.calls += 1
        v, x, lam, dlam_dt = self.fields(y)
        with torch.no_grad():
            dy_e = self.electrical(
                torch.from_numpy(y[:50]), torch.tensor(t_ms),
                torch.tensor(lam if self.electrical_feedback else 1.0),
                torch.tensor(dlam_dt if self.electrical_feedback else 0.0)).numpy()
        if self.calcium_motor:
            # 1 - TmBlocked is Land's available thin-filament fraction.
            # The scale only maps its range into the upstream abstract [0,1]
            # motor-activation input; it is not a measured force conversion.
            act = float(np.clip((1.0-y[46]) / self.activation_scale, 0, 1))
        else:
            act = 0.0
        # Reuse the published, non-monotonic sarcomere force/velocity law.
        single = self.sarcomeres.force_single(x, v, act)
        p = self.sarcomeres.model_params
        # Original force_coupling with an externally controlled p(t).
        coupling = -p.k_l / (p.mu*p.N) * (np.sum(x)-prestrain)
        dv_dt_ms = (single + coupling) / 1000.0
        dx_dt_ms = v / 1000.0
        return np.r_[dy_e, dv_dt_ms, dx_dt_ms]


def run(model, initial, pulse=(-5.0, 250.0, 350.0), duration=700.0,
        rtol=1e-6, atol=1e-8, max_step=2.0, pulses=None):
    """Integrate one or more piecewise-constant source prestrain pulses.

    ``pulse`` keeps the original API. ``pulses`` supplies an arbitrary list of
    (amplitude, onset_ms, offset_ms) segments; overlapping amplitudes add.
    Discontinuities are always exact BDF segment boundaries.
    """
    protocol = [pulse] if pulses is None else list(pulses)
    for size, onset, offset in protocol:
        if not (np.isfinite(size) and np.isfinite(onset) and np.isfinite(offset)
                and 0 <= onset < offset <= duration):
            raise ValueError(f"Invalid pulse {(size, onset, offset)} for {duration} ms")
    boundaries = sorted({0.0, duration} | {
        float(edge) for _, onset, offset in protocol for edge in (onset, offset)})
    times, states = [], []
    y = initial.copy()
    begin = time.perf_counter()
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        p = sum(size for size, onset, offset in protocol if onset <= a < offset)
        sol = solve_ivp(lambda t, z: model.rhs(t, z, p), (a, b), y,
                        method="BDF", max_step=max_step, rtol=rtol,
                        atol=atol)
        if not sol.success:
            raise RuntimeError(f"BDF {a}-{b} ms: {sol.message}")
        times.append(sol.t if not states else sol.t[1:])
        states.append(sol.y.T if not states else sol.y.T[1:])
        y = sol.y[:, -1]
    t = np.concatenate(times)
    state = np.vstack(states)
    grid = np.arange(0, duration+1)
    sampled = np.column_stack([np.interp(grid, t, state[:, j])
                               for j in range(state.shape[1])])
    x = sampled[:, 50+model.n:50+2*model.n]
    v = sampled[:, 50:50+model.n]
    lam = 1-model.length_scale*np.mean(x, axis=1)
    dlam = -model.length_scale*np.mean(v, axis=1)/1000
    with torch.no_grad():
        _, tension, active, passive = land2016(
            torch.from_numpy(sampled[:, 43:50]),
            torch.from_numpy(1000*sampled[:, 5]),
            lam=torch.from_numpy(lam), dlam_dt=torch.from_numpy(dlam))
    return {"time_ms": grid, "state": sampled, "lambda": lam,
            "local_lambda": 1-model.length_scale*x,
            "activation": (np.clip((1-sampled[:, 46])/model.activation_scale,0,1)
                           if model.calcium_motor else np.zeros(len(grid))),
            "tension_kpa": tension.numpy(), "active_kpa": active.numpy(),
            "passive_kpa": passive.numpy(),
            "runtime_s": time.perf_counter()-begin, "rhs_calls": model.calls}


def initial_state(n=20):
    electrical = loadmat(ROOT / "analysis/torch_native_port/prepace_state_v5.mat")
    # The upstream model starts with small, random position perturbations.
    # Freeze one draw across interventions so comparisons remain paired.
    positions = np.random.default_rng(20261018).normal(0, 0.02, n)
    return np.r_[electrical["state"].ravel(), np.zeros(n), positions]


def summarize(result, reference=None):
    y = result["state"]
    out = {"voltage_min_mV": float(y[:,0].min()),
           "voltage_max_mV": float(y[:,0].max()),
           "calcium_peak_uM": float(1000*y[:,5].max()),
           "lambda_min": float(result["lambda"].min()),
           "lambda_max": float(result["lambda"].max()),
           "local_lambda_std_peak": float(result["local_lambda"].std(axis=1).max()),
           "activation_peak": float(result["activation"].max()),
           "tension_peak_kpa": float(result["tension_kpa"].max()),
           "rhs_calls": result["rhs_calls"],
           "runtime_s": result["runtime_s"]}
    if reference is not None:
        out["voltage_max_difference_mV"] = float(np.max(np.abs(y[:,0]-reference["state"][:,0])))
        out["calcium_max_difference_uM"] = float(1000*np.max(np.abs(y[:,5]-reference["state"][:,5])))
    return out


def main():
    torch.set_num_threads(1)
    n = 20
    y0 = initial_state(n)
    cases = {
        "closed_pulse": dict(pulse=(-5.,250.,350.),sac_on=True,trpn_on=True),
        "no_external_pulse": dict(pulse=(0.,250.,350.),sac_on=True,trpn_on=True),
        "pulse_no_sac": dict(pulse=(-5.,250.,350.),sac_on=False,trpn_on=True),
        "pulse_no_motor": dict(pulse=(-5.,250.,350.),sac_on=True,trpn_on=True,
                               calcium_motor=False),
        "pulse_no_mechanical_feedback": dict(pulse=(-5.,250.,350.),sac_on=True,
                                             trpn_on=True,electrical_feedback=False),
    }
    outputs = {}
    for name, setup in cases.items():
        pulse = setup.pop("pulse")
        model = ClosedLoop(n=n, **setup)
        outputs[name] = run(model, y0, pulse=pulse)
        print(name, summarize(outputs[name]), flush=True)
    ref = outputs["closed_pulse"]
    report = {
        "model": "ToR-ORd/SAC-Land + upstream Härtter sarcomere drift",
        "sarcomere_source": "external/SarcomereModel, MIT, quickstart parameters",
        "bridge": "Land thin-filament availability drives original motor activation; mean sarcomere x drives lambda back to SAC and Land",
        "limits": "Deterministic drift only; abstract source force and load units; length_scale 0.3 and activation_scale 0.2 exploratory; no human-cell joint calibration",
        "pulse": "source prestrain p=-5 on 250-350 ms; this is not a kPa load",
        "scenario_metrics": {key: summarize(value, ref if key != "closed_pulse" else None)
                             for key, value in outputs.items()},
    }
    (HERE / "sarcomere_closed_loop.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(HERE / "sarcomere_closed_loop.npz", **{
        f"{name}_{key}": val for name, output in outputs.items()
        for key, val in output.items() if isinstance(val, np.ndarray)})
    fig, axes = plt.subplots(5, 1, figsize=(10, 11), sharex=True, layout="constrained")
    colors = {"closed_pulse":"#c34d48", "no_external_pulse":"#777777",
              "pulse_no_sac":"#4968a3", "pulse_no_motor":"#8b7d34",
              "pulse_no_mechanical_feedback":"#57a182"}
    for name, result in outputs.items():
        t, y = result["time_ms"], result["state"]
        vals = (y[:,0], 1000*y[:,5], result["activation"],
                result["lambda"], result["tension_kpa"])
        for ax, val in zip(axes, vals):
            ax.plot(t, val, label=name, color=colors[name], lw=1.1)
    labels = ("V (mV)", "Ca_i (µM)", "Thin filament activation",
              "Mean sarcomere stretch λ", "Land tension (kPa)")
    for ax, lab in zip(axes, labels):
        ax.set_ylabel(lab)
        ax.axvspan(250,350,color="#dddddd",alpha=.35)
        ax.grid(alpha=.2)
    axes[0].legend(ncol=2,fontsize=8,frameon=False)
    axes[-1].set_xlabel("Time (ms)")
    fig.savefig(HERE / "sarcomere_closed_loop.png",dpi=170)


if __name__ == "__main__":
    main()
