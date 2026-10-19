"""Numerical and causal checks for the exploratory cell-heterogeneity interface."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

from embodied_length_pilot import ROOT, EmbodiedRHS
from analysis.torch_native_port.land2016_torch import land2016


def main():
    torch.set_num_threads(1)
    x=loadmat(ROOT/"analysis/torch_native_port/prepace_state_v5.mat")["state"].ravel()
    with torch.no_grad():
        _,rest,_,_=land2016(torch.from_numpy(x[43:50]),
                            torch.tensor(1000*x[5],dtype=torch.float64))
    y0=np.r_[x,1.0]
    t=torch.tensor(270.,dtype=torch.float64)
    force=torch.tensor(12.,dtype=torch.float64)
    model=EmbodiedRHS(preload=float(rest),sac_gain=.8,
                      land_beta=(2.3,-3.1),stiffness=205.,
                      viscosity=6200.).double()
    traced=torch.jit.trace(model,(torch.from_numpy(y0),t,force),check_trace=False)
    rhs_error=[]
    for lam in (.96,1.,1.04):
        state=torch.from_numpy(np.r_[x,lam])
        with torch.no_grad():
            a=model(state,t,force)
            b=traced(state,t,force)
        rhs_error.append(float(torch.max(torch.abs(a-b))))
    gain0=EmbodiedRHS(preload=float(rest),sac_gain=0.,
                      land_beta=(2.3,-3.1)).double()
    gain1=EmbodiedRHS(preload=float(rest),sac_gain=1.,
                      land_beta=(2.3,-3.1)).double()
    g0base=EmbodiedRHS(preload=float(rest),sac_gain=0.,
                      land_beta=(2.3,-3.1),sac_on=False).double()
    with torch.no_grad():
        state=torch.from_numpy(np.r_[x,1.04])
        zero=gain0(state,t,force)
        sac_off=g0base(state,t,force)
        one=gain1(state,t,force)
    report={"eager_vs_traced_max_rhs_by_lambda":rhs_error,
            "gain_zero_vs_sac_off_max_rhs":float(torch.max(torch.abs(zero-sac_off))),
            "gain_one_vs_zero_voltage_rhs_difference":float(abs(one[0]-zero[0])),
            "note":"gain=0 removes direct stretch-electrical RHS difference; TnC coupling remains active in both."}
    output=Path(__file__).with_name("parameterized_rhs_validation.json")
    output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
