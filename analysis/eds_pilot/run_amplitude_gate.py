"""Test whether a second field amplitude exposes nonlinear membrane mechanics."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from full_model import forward, infer, infer_nonlinear

OUT=Path(__file__).resolve().parent/"results"
SEED=int(os.getenv("EDS_SEED","20261013"))
TAG=os.getenv("EDS_TAG","")
RNG=np.random.default_rng(SEED)
FREQ=np.geomspace(1e5,15e6,21)
TARGET_FREQ=np.array([.35,1.225,4.286])*1e6
N=100
NOISE=float(os.getenv("EDS_AXIS_NOISE","0.005"))
TRUTH_FORM=os.getenv("EDS_TRUTH_FORM","linear_plus")
TRUTH_CLOSURE=os.getenv("EDS_TRUTH_CLOSURE","balanced")


def measure(dims,volume,noise):
    a=dims[:,0]*np.exp(noise[:,0]);b=dims[:,1]*np.exp(noise[:,1])
    return np.column_stack([a,b,volume/(a*b)])


def main():
    design=json.loads((OUT/"ellipsoid_gate.json").read_text(encoding="utf-8"))["design_frequencies_MHz"]
    d3=np.array(design["dopt3"])*1e6
    d5=np.array(design["dopt5"])*1e6
    truth=np.column_stack([RNG.uniform(2,8,N),RNG.uniform(.3,1.1,N),
                           RNG.uniform(1.5,4,N),RNG.uniform(.5,3,N)])
    if os.getenv("EDS_KAPPA_ZERO", "0") == "1":
        truth[:,3]=0.0
    a0=RNG.uniform(7.2,8.6,N)*1e-6
    c0=RNG.uniform(2.3,3.1,N)*1e-6
    protocols={"full21_E1":(FREQ,np.ones(len(FREQ))),
               "full21_high_E":(FREQ,np.full(len(FREQ),1.2**2)),
               "five_E1":(d5,np.ones(len(d5))),
               "six_high_E":(FREQ[np.round(np.linspace(0,20,6)).astype(int)],np.full(6,1.2**2)),
               "ten_high_E":(FREQ[np.round(np.linspace(0,20,10)).astype(int)],np.full(10,1.2**2)),
               "three_two_E":(np.tile(d3,2),np.repeat([.8**2,1.2**2],len(d3))),
               "five_two_E":(np.tile(d5,2),np.repeat([.8**2,1.2**2],len(d5)))}
    methods={"linear":lambda f,d,g,a:infer(f,d,a),
             "nonlinear":lambda f,d,g,a:infer_nonlinear(f,d,g,a)}
    result={p:{m:{"parameter_error":[],"target_lambda_error":[]} for m in methods} for p in protocols}
    for i in range(N):
        volume=a0[i]*a0[i]*c0[i]
        # Shared noise for the same cell, amplitude and frequency across designs.
        noise_lut=RNG.normal(0,NOISE,(3,len(FREQ),2))
        target=forward(TARGET_FREQ,truth[i,:3],a0[i],a0[i],c0[i],
                       field_gain=1.4**2,nonlinear_kappa=truth[i,3],
                       mechanics_form=TRUTH_FORM,closure=TRUTH_CLOSURE)[:,0]/a0[i]
        for pname,(freq,field) in protocols.items():
            clean=np.vstack([forward([f],truth[i,:3],a0[i],a0[i],c0[i],
                                     field_gain=float(g),nonlinear_kappa=truth[i,3],
                                     mechanics_form=TRUTH_FORM,closure=TRUTH_CLOSURE)[0]
                             for f,g in zip(freq,field)])
            freq_index=np.abs(FREQ[None,:]-freq[:,None]).argmin(axis=1)
            amp_index=np.abs(np.array([.8**2,1.0,1.2**2])[None,:]-field[:,None]).argmin(axis=1)
            noise=noise_lut[amp_index,freq_index]
            measured=measure(clean,volume,noise)
            for mname,fitfun in methods.items():
                if mname=="linear" and not np.all(field==1):
                    # Linear fit must also know the intended field amplitude.
                    from scipy.optimize import least_squares
                    from full_model import residual
                    fit=least_squares(lambda p:residual(freq,measured,a0[i],*np.exp(p),
                                                        field_gain=field)/1e-11,
                                      np.log([4.7,.65,2.5]),
                                      bounds=(np.log([.1,.1,.1]),np.log([30,3,30])),max_nfev=80)
                    est=np.exp(fit.x)
                else:
                    est=fitfun(freq,measured,field,a0[i])
                kappa=est[3] if len(est)==4 else 0.0
                pred=forward(TARGET_FREQ,est[:3],a0[i],a0[i],c0[i],
                             field_gain=1.4**2,nonlinear_kappa=kappa)[:,0]/a0[i]
                result[pname][mname]["parameter_error"].append(
                    np.r_[np.abs(np.log(est[:3]/truth[i,:3])),abs(kappa-truth[i,3])])
                result[pname][mname]["target_lambda_error"].append(np.mean(np.abs(pred-target)))
    report={"scope":"synthetic 4-param nonlinear membrane; second field amplitude is an explicit additional intervention",
            "random_seed":SEED,
            "kappa_zero_control":bool(np.all(truth[:,3]==0)),
            "truth_mechanics_form":TRUTH_FORM,
            "truth_shape_closure":TRUTH_CLOSURE,
            "n_cells":N,"axis_noise_fractional_sd":NOISE,
            "calibration_field_ratios":[.8,1.0,1.2],"held_out_field_ratio":1.4,
            "target_frequencies_MHz":(TARGET_FREQ/1e6).tolist(),"protocols":{}}
    archive={}
    for pname,methods_data in result.items():
        report["protocols"][pname]={}
        for mname,record in methods_data.items():
            pe=np.asarray(record["parameter_error"]);te=np.asarray(record["target_lambda_error"])
            archive[pname+"__"+mname+"__parameter"]=pe
            archive[pname+"__"+mname+"__target"]=te
            report["protocols"][pname][mname]={
                "median_parameter_error":np.median(pe,axis=0).tolist(),
                "median_target_lambda_mae":float(np.median(te)),
                "p90_target_lambda_mae":float(np.quantile(te,.9))}
    np.savez_compressed(OUT/f"amplitude_per_cell{TAG}.npz",truth=truth,a0=a0,c0=c0,**archive)
    (OUT/f"amplitude_gate{TAG}.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    for pname in protocols:
        print(pname)
        for mname in methods:
            r=report["protocols"][pname][mname]
            print(mname,"param",np.round(r["median_parameter_error"],3),
                  "target",round(r["median_target_lambda_mae"],4),
                  "p90",round(r["p90_target_lambda_mae"],4))


if __name__=="__main__":main()
