"""Validate a whole 1000 ms beat against MATLAB's native saved waveform."""
from pathlib import Path
import json
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy.integrate import solve_ivp
from scipy.io import loadmat

from analysis.torch_native_port.benchmark_compiled_rhs import NativeRHS
from analysis.torch_native_port.land2016_torch import land2016

HERE=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(1)
    length=float(sys.argv[1]) if len(sys.argv)>1 else 1.0
    key=f"lambda{int(round(100*length)):03d}"
    reference=HERE.parent/"human_bidirectional_output"/f"trpn1_sac1_{key}.csv"
    ref=pd.read_csv(reference)
    t_eval=ref["grid"].to_numpy() if "grid" in ref else ref.iloc[:,0].to_numpy()
    y0=loadmat(HERE/"prepace_state_v5.mat")["state"].ravel().copy()
    state=torch.from_numpy(y0)
    lam=torch.tensor(length,dtype=state.dtype)
    traced=torch.jit.trace(NativeRHS(),(state,torch.tensor(0.,dtype=state.dtype),lam),check_trace=False)
    calls=0

    def rhs(t,y):
        nonlocal calls
        calls+=1
        with torch.no_grad():
            return traced(torch.from_numpy(y),torch.tensor(t,dtype=state.dtype),lam).numpy()

    start=time.perf_counter()
    result=solve_ivp(rhs,(float(t_eval[0]),float(t_eval[-1])),y0,method="BDF",
                     t_eval=t_eval,rtol=1e-6,atol=1e-8,max_step=2.)
    seconds=time.perf_counter()-start
    y=result.y.T
    voltage=y[:,0]
    cai=1000*y[:,5]
    with torch.no_grad():
        _,_,active,passive=land2016(torch.from_numpy(y[:,43:50]),torch.from_numpy(cai),lam=lam)
        active=active.numpy()
    ref_v=ref["Vm_mV"].to_numpy()
    ref_ca=ref["Cai_uM"].to_numpy()
    ref_t=ref["Ta_kPa"].to_numpy()
    n=min(len(y),len(ref))
    report={
        "length":length,
        "success":bool(result.success),"message":result.message,
        "computed_ms":float(result.t[-1]-result.t[0]),
        "runtime_seconds":seconds,"rhs_calls":calls,
        "solver_nfev":int(result.nfev),"solver_njev":int(result.njev),
        "V_MAE_mV":float(np.mean(np.abs(voltage[:n]-ref_v[:n]))),
        "V_max_error_mV":float(np.max(np.abs(voltage[:n]-ref_v[:n]))),
        "Ca_MAE_uM":float(np.mean(np.abs(cai[:n]-ref_ca[:n]))),
        "T_MAE_kPa":float(np.mean(np.abs(active[:n]-ref_t[:n]))),
    }
    (HERE/f"full_beat_validation_{key}.json").write_text(json.dumps(report,indent=2))
    np.savez_compressed(HERE/f"full_beat_validation_{key}.npz",time=result.t,
                        predicted_voltage=voltage,predicted_cai=cai,predicted_active_tension=active)
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
