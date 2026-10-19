"""Small paired stress-controlled 51-state virtual-cell cohort.

This cohort is a controlled first-beat perturbation study. All cells begin
from the same nominal pre-paced electrical state at lambda=1. Cell-specific
mechanical steady-state pre-pacing is NOT performed. K and eta are exploratory
engineering values, not estimates from human-cell experiments.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
import time
import warnings

import numpy as np
from scipy.io import loadmat
import torch

from embodied_length_pilot import ROOT, HERE, EmbodiedRHS, simulate
from analysis.torch_native_port.land2016_torch import land2016

torch.set_num_threads(1)
warnings.filterwarnings("ignore",category=torch.jit.TracerWarning)
OUT=HERE/"embodied_cohort"
OUT.mkdir(exist_ok=True)
N=12
SEED=20261019
NAMES=["sac_gain","land_beta1","stiffness_kPa_per_stretch",
       "viscosity_kPa_ms_per_stretch"]
LO=np.array([.4,-4.1,180.,4000.])
HI=np.array([1.6,-.9,260.,7000.])
PROTOCOLS=[("free",0.,250.,350.),
           ("cal12_250_350",12.,250.,350.),
           ("target18_300_450",18.,300.,450.),
           ("target12_400_500",12.,400.,500.)]
DURATION=600.


def latin_hypercube(n,seed):
    rng=np.random.default_rng(seed)
    u=np.column_stack([(rng.permutation(n)+rng.random(n))/n for _ in NAMES])
    return LO+(HI-LO)*u


def one_cell(cell_id,theta,y50,preload):
    gain,beta1,k,eta=map(float,theta)
    y0=np.r_[y50,1.0]
    module=EmbodiedRHS(stiffness=k,viscosity=eta,preload=preload,
                       sac_gain=gain,land_beta=(2.3,beta1)).double()
    exemplar=(torch.from_numpy(y0),torch.tensor(0.,dtype=torch.float64),
              torch.tensor(0.,dtype=torch.float64))
    compiled=torch.jit.trace(module,exemplar,check_trace=False)
    waves=[];runtimes=[]
    for label,pressure,onset,offset in PROTOCOLS:
        result=simulate(compiled,y0,pressure=pressure,onset=onset,offset=offset,
                        duration=DURATION,rtol=1e-6,atol=1e-8,max_step=2.,
                        land_beta=(2.3,beta1))
        w=np.column_stack([result["wave"][:,0],result["wave"][:,1],
                           result["tension"],result["wave"][:,2]])
        if not np.isfinite(w).all() or w[:,3].min()<.80 or w[:,3].max()>1.25:
            raise RuntimeError(f"QC failed {label}: λ range {w[:,3].min():.3f}–{w[:,3].max():.3f}")
        waves.append(w)
        runtimes.append(float(result["runtime_s"]))
        print(f"cell {cell_id:02d}/{N} {label} {result['runtime_s']:.1f}s λ={w[:,3].min():.4f}..{w[:,3].max():.4f}",flush=True)
    return np.stack(waves),runtimes


def main(limit=N, seed=SEED, out=OUT):
    out=Path(out)
    out.mkdir(exist_ok=True)
    theta=latin_hypercube(N,seed)
    y50=loadmat(ROOT/"analysis/torch_native_port/prepace_state_v5.mat")["state"].ravel()
    with torch.no_grad():
        _,resting,_,_=land2016(torch.from_numpy(y50[43:50]),
                               torch.tensor(1000*y50[5],dtype=torch.float64))
    preload=float(resting)
    status=[]
    started=time.perf_counter()
    for i in range(min(N,limit)):
        path=out/f"cell_{i+1:02d}.npz"
        if path.exists():
            with np.load(path) as previous:
                if np.allclose(previous["theta"],theta[i]) and previous["wave"].shape==(4,601,4):
                    print(f"cell {i+1:02d}/{N} existing valid checkpoint",flush=True)
                    status.append({"cell":i+1,"ok":True,"checkpoint":True})
                    continue
            raise RuntimeError(f"Existing checkpoint has different parameters: {path}")
        try:
            wave,runtime=one_cell(i+1,theta[i],y50,preload)
            np.savez_compressed(path,theta=theta[i],wave=wave,runtime_s=runtime)
            status.append({"cell":i+1,"ok":True,"runtime_s":runtime})
        except Exception as exc:
            status.append({"cell":i+1,"ok":False,"error":repr(exc)})
            print(f"cell {i+1:02d}/{N} FAILED {exc}",flush=True)
        metadata={"seed":seed,"n_planned":N,"parameter_names":NAMES,
                  "parameter_lower":LO.tolist(),"parameter_upper":HI.tolist(),
                  "protocols":PROTOCOLS,"duration_ms":DURATION,
                  "preload_kPa":preload,"initial_state":"Shared nominal pre-paced 50-state state; no cell-specific mechanical pre-pacing",
                  "status":status,"elapsed_s":time.perf_counter()-started}
        (out/"manifest.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(f"COMPLETE {sum(v['ok'] for v in status)}/{N}",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--limit",type=int,default=N)
    parser.add_argument("--seed",type=int,default=SEED)
    parser.add_argument("--out",type=Path,default=OUT)
    args=parser.parse_args()
    main(args.limit,args.seed,args.out)
