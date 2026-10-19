"""Autograd vs finite difference through a stiff, short 50-state rollout."""
from pathlib import Path
import json
import time

import torch
from scipy.io import loadmat

from analysis.torch_native_port.native_bidirectional import native_rhs

HERE=Path(__file__).resolve().parent


class GainRHS(torch.nn.Module):
    def forward(self,y,t,lam,gain):
        return native_rhs(y,t,lam,sac_gain=gain)[0]


def main():
    torch.set_num_threads(1)
    y0=torch.from_numpy(loadmat(HERE/"prepace_state_v5.mat")["state"].ravel().copy())
    lam=torch.tensor(1.08,dtype=y0.dtype)
    zero=torch.tensor(0.,dtype=y0.dtype)
    example_gain=torch.tensor(.8,dtype=y0.dtype)
    rhs=torch.jit.trace(GainRHS(),(y0,zero,lam,example_gain),check_trace=False)
    h=.001; n=100

    def rollout(gain):
        y=y0
        for k in range(n):
            t=torch.tensor(k*h,dtype=y0.dtype)
            k1=rhs(y,t,lam,gain)
            k2=rhs(y+.5*h*k1,t+.5*h,lam,gain)
            k3=rhs(y+.5*h*k2,t+.5*h,lam,gain)
            k4=rhs(y+h*k3,t+h,lam,gain)
            y=y+(h/6)*(k1+2*k2+2*k3+k4)
        return y

    start=time.perf_counter()
    gain=example_gain.clone().requires_grad_(True)
    final=rollout(gain)
    sensitivity=torch.autograd.grad(final[0],gain)[0]
    autodiff_seconds=time.perf_counter()-start

    start=time.perf_counter()
    with torch.no_grad():
        eps=1e-4
        hi=rollout(example_gain+eps)[0]
        lo=rollout(example_gain-eps)[0]
        fd=(hi-lo)/(2*eps)
    fd_seconds=time.perf_counter()-start
    report={
        "duration_ms":n*h,"step_ms":h,
        "final_voltage_mV":float(final[0]),
        "autograd_dV_dgain_mV":float(sensitivity),
        "finite_difference_dV_dgain_mV":float(fd),
        "relative_gradient_difference":float(abs(sensitivity-fd)/max(abs(fd),1e-12)),
        "autodiff_runtime_seconds":autodiff_seconds,
        "two_finite_difference_rollouts_seconds":fd_seconds,
        "finite_state":bool(torch.isfinite(final).all()),
    }
    (HERE/"differentiable_rollout_validation.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
