"""Integrate the traced native 50-state RHS against MATLAB ode15s."""
from pathlib import Path
import json
import time

import numpy as np
import torch
from scipy.io import loadmat

from analysis.torch_native_port.benchmark_compiled_rhs import NativeRHS

HERE = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(1)
    z = loadmat(HERE / "trajectory_reference.mat")
    ref = z["states"][:, :, 1]
    yy = torch.tensor(ref[0], dtype=torch.float64)
    lam = torch.tensor(1.08, dtype=yy.dtype)
    t0 = torch.tensor(0., dtype=yy.dtype)
    rhs = torch.jit.trace(NativeRHS(), (yy, t0, lam), check_trace=False)
    h = 0.001  # ms; explicit RK4 needs a small step for this stiff system
    steps = int(round(float(z["times"][-1, 0])/h))
    stride = int(round(0.02/h))
    samples = [yy.detach().numpy()]
    begin = time.perf_counter()
    with torch.no_grad():
        for j in range(steps):
            tt = torch.tensor(j*h, dtype=yy.dtype)
            k1 = rhs(yy, tt, lam)
            k2 = rhs(yy + h*.5*k1, tt + h*.5, lam)
            k3 = rhs(yy + h*.5*k2, tt + h*.5, lam)
            k4 = rhs(yy + h*k3, tt + h, lam)
            yy = yy + (h/6)*(k1+2*k2+2*k3+k4)
            if (j+1) % stride == 0:
                samples.append(yy.detach().numpy())
            if not torch.isfinite(yy).all():
                break
    seconds = time.perf_counter()-begin
    pred = np.asarray(samples)
    comparable = min(len(pred), len(ref))
    err = np.abs(pred[:comparable]-ref[:comparable])
    metrics = {
        "duration_ms": float((len(pred)-1)*0.02),
        "requested_duration_ms": float(z["times"][-1,0]),
        "step_ms": h,
        "runtime_seconds": seconds,
        "finite": bool(np.isfinite(pred).all()),
        "max_state_abs_error": float(np.nanmax(err)),
        "max_voltage_abs_error_mV": float(np.nanmax(err[:,0])),
        "max_cai_abs_error_uM": float(1000*np.nanmax(err[:,5])),
        "final_voltage_error_mV": float(err[-1,0]),
        "final_cai_error_uM": float(1000*err[-1,5]),
    }
    (HERE / "compiled_trajectory_validation.json").write_text(json.dumps(metrics,indent=2))
    np.savez_compressed(HERE / "compiled_trajectory_validation.npz",pred=pred,ref=ref[:comparable])
    print(json.dumps(metrics,indent=2))


if __name__ == "__main__":
    main()
