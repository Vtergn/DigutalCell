"""Benchmark a trace compiled 50-state RHS and check derivatives/gradients."""
from pathlib import Path
import json
import time

import numpy as np
import torch
from scipy.io import loadmat

from analysis.torch_native_port.native_bidirectional import native_rhs

HERE = Path(__file__).resolve().parent


class NativeRHS(torch.nn.Module):
    def forward(self, state, time_ms, length):
        return native_rhs(state, time_ms, length)[0]


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    torch.set_num_threads(1)
    data = loadmat(HERE / "trajectory_reference.mat")
    device = torch.device("cpu")
    state = torch.tensor(data["states"][20, :, 1], dtype=torch.float64, device=device)
    t = torch.tensor(0.4, dtype=state.dtype, device=device)
    length = torch.tensor(1.08, dtype=state.dtype, device=device)
    eager = NativeRHS()

    t0 = time.perf_counter()
    traced = torch.jit.trace(eager, (state, t, length), check_trace=False)
    trace_seconds = time.perf_counter() - t0

    with torch.no_grad():
        a = eager(state, t, length)
        b = traced(state, t, length)
        value_error = float(torch.max(torch.abs(a-b)))
        # Check that time and length are live graph inputs.
        altered = float(torch.max(torch.abs(traced(state, t, torch.tensor(1.0))-b)))
        stim_off = float(torch.max(torch.abs(traced(state, torch.tensor(1.5), length)-b)))
        for fn in (eager, traced):
            for _ in range(5):
                fn(state, t, length)
        timings = {}
        for name, fn in (("eager", eager), ("traced", traced)):
            start = time.perf_counter()
            for _ in range(300):
                fn(state, t, length)
            synchronize(device)
            timings[name] = (time.perf_counter()-start)/300

    gradient = {}
    for name, fn in (("eager", eager), ("traced", traced)):
        x = state.clone().requires_grad_(True)
        ell = length.clone().requires_grad_(True)
        dx = fn(x, t, ell)
        objective = dx[0] + 1e3*dx[5] + dx[45]
        gy, gl = torch.autograd.grad(objective, (x, ell))
        gradient[name] = {"state": gy.detach().numpy(), "length": float(gl)}
    gradient_error = float(np.max(np.abs(gradient["eager"]["state"] - gradient["traced"]["state"])))
    length_gradient_error = abs(gradient["eager"]["length"] - gradient["traced"]["length"])

    results = {
        "torch_version": torch.__version__,
        "device": str(device),
        "dtype": str(state.dtype),
        "trace_seconds": trace_seconds,
        "eager_rhs_seconds": timings["eager"],
        "traced_rhs_seconds": timings["traced"],
        "speedup": timings["eager"]/timings["traced"],
        "max_value_difference": value_error,
        "max_state_gradient_difference": gradient_error,
        "length_gradient_difference": length_gradient_error,
        "length_input_effect": altered,
        "time_input_effect": stim_off,
    }
    (HERE / "compiled_rhs_benchmark.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
