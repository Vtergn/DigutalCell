"""Check both sides of the modeled triggered-AP boundary with strict BDF."""
from __future__ import annotations

import json

import numpy as np
import torch
from scipy.io import loadmat

from audit_triggered_response import OUT, HERE, ROOT, one
from analysis.torch_native_port.land2016_torch import land2016


def main():
    with np.load(OUT/"cell_02.npz") as f:
        _,beta1,k,eta=map(float,f["theta"])
    x=loadmat(ROOT/"analysis/torch_native_port/prepace_state_v5.mat")["state"].ravel()
    with torch.no_grad():
        _,rest,_,_=land2016(torch.from_numpy(x[43:50]),
                            torch.tensor(1000*x[5],dtype=torch.float64))
    y0=np.r_[x,1.]
    cases=[]
    for gain in (1.35,1.40):
        result=one(gain,beta1,k,eta,y0,float(rest),strict=True)
        v=result["wave"][:,0]
        cases.append({"gain":gain,"post_450_max_V_mV":float(v[450:].max()),
                      "triggered_post_450_V_above_0":bool(np.any(v[450:]>0)),
                      "runtime_s":result["runtime_s"]})
        print(cases[-1],flush=True)
    (HERE/"event_boundary_strict_validation.json").write_text(
        json.dumps({"strict_solver_boundary_endpoints":cases},indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
