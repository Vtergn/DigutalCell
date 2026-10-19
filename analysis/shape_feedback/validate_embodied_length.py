"""Check graph inputs, finite RHS and tighter solver settings for the 51-state pilot."""
from pathlib import Path
import json
import sys

import numpy as np
import torch
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.shape_feedback.embodied_length_pilot import EmbodiedRHS, simulate
from analysis.torch_native_port.land2016_torch import land2016
from analysis.torch_native_port.native_bidirectional import native_rhs


def main():
    torch.set_num_threads(1)
    prev = np.load(Path(__file__).with_name("embodied_length_pilot.npz"))
    x = loadmat(ROOT / "analysis/torch_native_port/prepace_state_v5.mat")["state"].ravel()
    with torch.no_grad():
        _, total, _, _ = land2016(torch.from_numpy(x[43:50]),
                                  torch.tensor(1000*x[5], dtype=torch.float64))
    y0 = np.r_[x, 1.0]
    module = EmbodiedRHS(preload=float(total)).double()
    t = torch.tensor(260., dtype=torch.float64)
    force = torch.tensor(12., dtype=torch.float64)
    traced = torch.jit.trace(module, (torch.from_numpy(y0), t, force),
                             check_trace=False)
    derivative_error = []
    for lam in (.96, 1.0, 1.04):
        y = torch.from_numpy(np.r_[x, lam])
        with torch.no_grad():
            direct = module(y, t, force)
            compiled = traced(y, t, force)
        derivative_error.append(float(torch.max(torch.abs(direct-compiled))))
        if not torch.isfinite(compiled).all():
            raise AssertionError(f"nonfinite RHS at lambda={lam}")
    with torch.no_grad():
        original = native_rhs(torch.from_numpy(x), t, 1.04)[0]
        explicit_default = native_rhs(torch.from_numpy(x), t, 1.04, sac_lam=1.04)[0]
    default_interface_error = float(torch.max(torch.abs(original-explicit_default)))
    strict = simulate(traced, y0, pressure=12., rtol=1e-8, atol=1e-10,
                      max_step=.5)
    normal = prev["load_feedback_wave"]
    normal_lam = np.interp(strict["time"], prev["load_feedback_stress_time"],
                           prev["load_feedback_lam"])
    report = {
        "compiled_eager_max_abs_rhs_difference_by_length": derivative_error,
        "native_rhs_default_api_difference": default_interface_error,
        "strict_solver_runtime_seconds": strict["runtime_s"],
        "strict_vs_default_voltage_MAE_mV": float(np.mean(np.abs(
            strict["wave"][:, 0]-normal[:, 0]))),
        "strict_vs_default_Ca_MAE_uM": float(np.mean(np.abs(
            strict["wave"][:, 1]-normal[:, 1]))),
        "strict_vs_default_lambda_MAE": float(np.mean(np.abs(
            strict["wave"][:, 2]-normal_lam))),
        "strict_vs_default_voltage_max_mV": float(np.max(np.abs(
            strict["wave"][:, 0]-normal[:, 0]))),
    }
    Path(__file__).with_name("embodied_length_validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
