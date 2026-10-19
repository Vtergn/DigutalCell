"""Audit the single strong-load triggered response in the exploratory cohort."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat

from embodied_length_pilot import ROOT, HERE, EmbodiedRHS, simulate
from analysis.torch_native_port.land2016_torch import land2016

torch.set_num_threads(1)
warnings.filterwarnings("ignore",category=torch.jit.TracerWarning)
OUT=HERE/"embodied_cohort"


def one(gain,beta1,k,eta,y0,preload,strict=False):
    model=EmbodiedRHS(preload=preload,sac_gain=gain,
                      land_beta=(2.3,beta1),stiffness=k,viscosity=eta).double()
    exemplar=(torch.from_numpy(y0),torch.tensor(0.,dtype=torch.float64),
              torch.tensor(0.,dtype=torch.float64))
    compiled=torch.jit.trace(model,exemplar,check_trace=False)
    settings=dict(rtol=1e-8,atol=1e-10,max_step=.5) if strict else {}
    return simulate(compiled,y0,pressure=18.,onset=300.,offset=450.,
                    duration=600.,land_beta=(2.3,beta1),**settings)


def main():
    with np.load(OUT/"cell_02.npz") as f:
        theta=f["theta"]
        stored=f["wave"]
    gain,beta1,k,eta=map(float,theta)
    x=loadmat(ROOT/"analysis/torch_native_port/prepace_state_v5.mat")["state"].ravel()
    with torch.no_grad():
        _,resting,_,_=land2016(torch.from_numpy(x[43:50]),
                                torch.tensor(1000*x[5],dtype=torch.float64))
    y0=np.r_[x,1.]
    strict=one(gain,beta1,k,eta,y0,float(resting),strict=True)
    strict_v=strict["wave"][:,0]
    stored_v=stored[2,:,0]
    checkpoints=[]
    gains=[0.,.9,1.1,1.25,1.3,1.35,1.4,gain,1.5]
    traces={}
    for candidate in gains:
        result=one(candidate,beta1,k,eta,y0,float(resting))
        v=result["wave"][:,0]
        traces[float(candidate)]=v
        checkpoints.append({"gain":float(candidate),
                            "post_450_max_V_mV":float(v[450:].max()),
                            "post_450_max_dV_mV":float((v-stored[0,:,0])[450:].max()),
                            "triggered_post_450_V_above_0":bool(np.any(v[450:]>0)),
                            "runtime_s":result["runtime_s"]})
        print(checkpoints[-1],flush=True)
    report={"cell_id":2,"fixed_other_parameters":
            {"Land_beta1":beta1,"K_kPa_per_stretch":k,
             "viscosity_kPa_ms_per_stretch":eta},
            "strict_vs_original_voltage_MAE_mV":float(np.mean(np.abs(strict_v-stored_v))),
            "strict_vs_original_voltage_max_difference_mV":float(np.max(np.abs(strict_v-stored_v))),
            "strict_post_450_max_V_mV":float(strict_v[450:].max()),
            "gain_scan":checkpoints,
            "interpretation":"Within-model threshold audit only; not evidence of biological arrhythmia or a population of discrete cell states."}
    (HERE/"triggered_response_audit.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    fig,axs=plt.subplots(2,1,figsize=(9,6),sharex=True,constrained_layout=True)
    axs[0].plot(stored[0,:,0],color="black",lw=1.4,label="no load")
    axs[0].plot(stored_v,color="#c44e52",lw=1.2,label="stored, 18 kPa")
    axs[0].plot(strict_v,color="#4c72b0",ls="--",lw=1.2,
                label="strict solver, 18 kPa")
    axs[0].legend(fontsize=8)
    for candidate in gains:
        axs[1].plot(traces[float(candidate)],lw=1.1,
                    label=f"gain {candidate:.3f}")
    axs[1].legend(fontsize=7,ncol=3)
    axs[0].set_ylabel("V (mV)")
    axs[1].set_ylabel("V (mV)")
    axs[1].set(xlabel="Time (ms)",xlim=(280,600))
    fig.suptitle("Numerical and effective-gain audit of triggered response")
    fig.savefig(HERE/"triggered_response_audit.png",dpi=180)
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    main()
