"""Source-equivalence, pathway, and solver audits for the hybrid closure."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from sarcomere_closed_loop import (ClosedLoop, HERE, initial_state, run,
                                   sarcomere_params, SimParams, Model)
from sarcomere_stochastic_cosim import mechanical_window


def source_equivalence():
    np.random.seed(20261018)
    n=20
    p=sarcomere_params(n)
    p.eta=[2.2,9.]
    y0=np.r_[np.zeros(n), np.random.default_rng(99).normal(0,.02,n)]
    s=SimParams(t1=.7,dt=.002,a_mode="sine",a=[1.,0.,1.],p=0.,y0=y0.copy())
    source=Model(p,s,verbose=False)
    fake=np.zeros((source.num_steps+1,50))
    fake[:-1,46]=1-.2*source.act
    fake[-1,46]=fake[-2,46]
    bridge=SimpleNamespace(n=n,sarcomeres=source,length_scale=.3,
                           activation_scale=.2,calcium_motor=True)
    extended=np.r_[np.zeros(50),y0]
    state,_=mechanical_window(bridge,0.,extended,fake,
                              (source.eta_thermal,source.eta_active),
                              (0.,250.,350.))
    source.integrate()
    return float(np.max(np.abs(state[1:]-source.y)))


def main():
    torch.set_num_threads(1)
    difference=source_equivalence()
    data=np.load(HERE/"sarcomere_closed_loop.npz")
    a=data["closed_pulse_state"]
    b=data["no_external_pulse_state"]
    no_sac=data["pulse_no_sac_state"]
    no_fb=data["pulse_no_mechanical_feedback_state"]
    time=np.arange(len(a))
    period=(time>=250)&(time<=500)
    strict=run(ClosedLoop(),initial_state(),pulse=(-5.,250.,350.),
               rtol=2e-7,atol=2e-9,max_step=1.)
    sc=strict["state"]
    open_without_pulse=run(ClosedLoop(electrical_feedback=False),initial_state(),
                           pulse=(0.,250.,350.),duration=460.)
    open_reference=no_fb[:461]
    report={
      "upstream_Euler_Maruyama_max_abs_difference":difference,
      "strict_solver_vs_default":{
        "voltage_max_abs_mV":float(np.max(np.abs(a[:,0]-sc[:,0]))),
        "calcium_max_abs_uM":float(1000*np.max(np.abs(a[:,5]-sc[:,5]))),
        "mean_lambda_max_abs":float(np.max(np.abs(data["closed_pulse_lambda"]-strict["lambda"])))},
      "mechanical_feedback_off_pulse_vs_no_pulse":{
        "voltage_max_abs_mV":float(np.max(np.abs(
            open_reference[:,0]-open_without_pulse["state"][:,0]))),
        "calcium_max_abs_uM":float(1000*np.max(np.abs(
            open_reference[:,5]-open_without_pulse["state"][:,5]))),
        "lambda_max_abs":float(np.max(np.abs(
            data["pulse_no_mechanical_feedback_lambda"][:461]-
            open_without_pulse["lambda"])))},
      "pulse_vs_no_pulse_after_250ms":{
        "V_max_difference_mV":float(np.max(np.abs(a[period,0]-b[period,0]))),
        "Ca_max_difference_uM":float(1000*np.max(np.abs(a[period,5]-b[period,5]))),
        "lambda_max_difference":float(np.max(np.abs(data["closed_pulse_lambda"][period]-data["no_external_pulse_lambda"][period])))},
      "SAC_ablation_under_same_pulse":{
        "V_max_difference_mV":float(np.max(np.abs(a[period,0]-no_sac[period,0]))),
        "Ca_max_difference_uM":float(1000*np.max(np.abs(a[period,5]-no_sac[period,5])))},
      "all_mechanical_feedback_ablation_under_same_pulse":{
        "V_max_difference_mV":float(np.max(np.abs(a[period,0]-no_fb[period,0]))),
        "Ca_max_difference_uM":float(1000*np.max(np.abs(a[period,5]-no_fb[period,5])))}}
    (HERE/"sarcomere_closed_loop_validation.json").write_text(
        json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    main()
