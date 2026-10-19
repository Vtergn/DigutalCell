"""Paired synthetic cohort under measured-axis, field and mechanics uncertainty."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from full_model import forward, infer

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(20261012)
FREQ = np.geomspace(1e5, 15e6, 21)
N = 120
IMAGE_NOISE = 0.005  # fractional noise on a,b; assumed, not measured here


def dopt(k):
    theta = np.log([4.7,.65,2.5])
    a0,b0,c0 = 7.8e-6,7.8e-6,2.7e-6
    columns=[]
    for j in range(3):
        shift=np.zeros(3); shift[j]=1e-3
        plus=forward(FREQ,np.exp(theta+shift),a0,b0,c0)
        minus=forward(FREQ,np.exp(theta-shift),a0,b0,c0)
        columns.append((np.log(plus[:,:2])-np.log(minus[:,:2]))/(2e-3*IMAGE_NOISE))
    jac=np.stack(columns,axis=-1)  # frequency, measured axis, parameter
    best,score_best=None,-np.inf
    for subset in itertools.combinations(range(len(FREQ)),k):
        matrix=jac[list(subset)].reshape(-1,3)
        sign,score=np.linalg.slogdet(matrix.T@matrix+1e-12*np.eye(3))
        if sign>0 and score>score_best:
            best,score_best=subset,score
    return best


def observed_dims(clean: np.ndarray, volume: float, noise: np.ndarray) -> np.ndarray:
    # The source paper measures projected a,b and derives c from cell volume.
    a=clean[:,0]*np.exp(noise[:,0]); b=clean[:,1]*np.exp(noise[:,1])
    c=volume/(a*b)
    return np.column_stack([a,b,c])


def simulate_case(case, theta, a0,b0,c0, field_gain):
    kwargs={}
    if case=="field_gain_uncertain":kwargs["field_gain"]=field_gain
    if case=="nonlinear_membrane":kwargs["nonlinear_kappa"]=2.0
    if case=="membrane_conductivity":kwargs["sigma_mem"]=1e-5
    if case=="alternative_shape":kwargs["closure"]="fixed_thickness"
    return forward(FREQ,theta,a0,b0,c0,**kwargs)


def summaries(errors):
    by_cell=errors.mean(axis=1)
    cutoff=np.quantile(by_cell,.9)
    return {"median_mean_absolute_log_error":float(np.median(by_cell)),
            "p90_mean_absolute_log_error":float(cutoff),
            "cvar90_mean_absolute_log_error":float(by_cell[by_cell>=cutoff].mean()),
            "median_by_parameter":np.median(errors,axis=0).tolist()}


def main():
    designs={"full21":tuple(range(len(FREQ))),
             "log3":tuple(np.round(np.linspace(0,20,3)).astype(int)),
             "dopt3":dopt(3),
             "log5":tuple(np.round(np.linspace(0,20,5)).astype(int)),
             "dopt5":dopt(5)}
    theta=np.column_stack([RNG.uniform(2,8,N),RNG.uniform(.3,1.1,N),RNG.uniform(1.5,4,N)])
    a0=RNG.uniform(7.2,8.6,N)*1e-6
    b0=a0.copy()
    c0=RNG.uniform(2.3,3.1,N)*1e-6
    gains=np.exp(RNG.normal(0,.15,N))
    noise=RNG.normal(0,IMAGE_NOISE,(N,len(FREQ),2))
    cases=["nominal","alternative_shape","field_gain_uncertain",
           "nonlinear_membrane","membrane_conductivity"]
    all_errors={}; report={}
    for case in cases:
        measured=[]
        for i in range(N):
            clean=simulate_case(case,theta[i],a0[i],b0[i],c0[i],gains[i])
            measured.append(observed_dims(clean,np.prod([a0[i],b0[i],c0[i]]),noise[i]))
        measured=np.asarray(measured)
        report[case]={}
        for name,idx in designs.items():
            idx=list(idx)
            estimates=np.array([infer(FREQ[idx],measured[i,idx],a0[i]) for i in range(N)])
            errors=np.abs(np.log(estimates/theta))
            all_errors[case+"__"+name]=errors
            report[case][name]=summaries(errors)
    payload={"scope":"Du Eq4/5/7/11 + declared forward shape closure; synthetic only",
             "n_cells":N,"axis_noise_fractional_sd":IMAGE_NOISE,
             "true_parameter_ranges":{"epsilon_mem":[2,8],"sigma_cyto_S_m":[.3,1.1],
                                      "mu_microN_m":[1.5,4]},
             "model_shifts":{"alternative_shape":"constant c, b=b0/lambda; a,b observed",
                             "field_gain_uncertain":"cell gain lognormal SD 0.15; inverse fixes gain=1",
                             "nonlinear_membrane":"true restoring force multiplied by 1+2(lambda-1)",
                             "membrane_conductivity":"true sigma_mem=1e-5 S/m; inverse 1e-6"},
             "design_frequencies_MHz":{k:(FREQ[list(v)]/1e6).tolist() for k,v in designs.items()},
             "results":report}
    (OUT/"ellipsoid_gate.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    np.savez_compressed(OUT/"ellipsoid_per_cell.npz",theta=theta,a0=a0,b0=b0,c0=c0,
                        gains=gains,noise=noise,**all_errors)
    for case in cases:
        print(case)
        for design in designs:
            s=report[case][design]
            print(f"  {design:7s} median={s['median_mean_absolute_log_error']:.3f} "
                  f"p90={s['p90_mean_absolute_log_error']:.3f} "
                  f"CVaR90={s['cvar90_mean_absolute_log_error']:.3f} "
                  f"by_param={np.round(s['median_by_parameter'],3)}")


if __name__=="__main__":
    main()
