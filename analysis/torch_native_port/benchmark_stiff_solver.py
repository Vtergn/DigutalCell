"""CPU reference speed: traced PyTorch RHS inside SciPy's stiff BDF solver."""
from pathlib import Path
import json
import time

import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.io import loadmat

from analysis.torch_native_port.benchmark_compiled_rhs import NativeRHS

HERE = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(1)
    z = loadmat(HERE / "trajectory_reference.mat")
    ref = z["states"][:, :, 1]
    grid = z["times"].ravel()
    state = torch.from_numpy(ref[0].copy())
    lam = torch.tensor(1.08, dtype=state.dtype)
    model = torch.jit.trace(NativeRHS(), (state, torch.tensor(0.,dtype=state.dtype), lam),check_trace=False)
    calls = 0

    def rhs(t, yy):
        nonlocal calls
        calls += 1
        with torch.no_grad():
            return model(torch.from_numpy(yy),torch.tensor(t,dtype=state.dtype),lam).numpy()

    start = time.perf_counter()
    result = solve_ivp(rhs,(0.,2.),ref[0],method="BDF",t_eval=grid,
                       rtol=1e-8,atol=1e-10,max_step=.05)
    elapsed = time.perf_counter()-start
    pred = result.y.T
    err = np.abs(pred-ref[:len(pred)])
    report = {
        "success": bool(result.success),
        "message": result.message,
        "runtime_seconds": elapsed,
        "rhs_calls": calls,
        "solver_nfev": result.nfev,
        "solver_njev": result.njev,
        "solver_nlu": result.nlu,
        "max_voltage_error_mV": float(err[:,0].max()),
        "final_voltage_error_mV": float(err[-1,0]),
        "max_cai_error_uM": float(1000*err[:,5].max()),
        "max_state_abs_error": float(err.max()),
    }
    (HERE / "stiff_solver_benchmark.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
