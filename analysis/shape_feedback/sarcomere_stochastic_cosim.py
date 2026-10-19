"""Partitioned closed-loop solver retaining the upstream Euler-Maruyama noise.

The electrical ODE is solved over short windows. Within each window, the
published sarcomere Euler-Maruyama step uses the same random draw at every
fixed-point iteration. This is a numerical co-simulation of two published
modules with an explicitly exploratory activation/length bridge.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import solve_ivp

from sarcomere_closed_loop import ClosedLoop, initial_state, ROOT, HERE

WINDOW_MS = 20.0


def source_noise(n, steps, amplitudes, seed, dt_ms):
    """The upstream code's iid normal / sqrt(dt_s) convention."""
    rng = np.random.default_rng(seed)
    dt_s = dt_ms/1000
    thermal = rng.normal(0, amplitudes[0], (steps, n))/np.sqrt(dt_s)
    active = rng.normal(0, amplitudes[1], (steps, n))/np.sqrt(dt_s)
    return thermal, active


def mechanical_window(model, start, y0, electrical, noise, pulse, dt_ms=2.0):
    nsteps = electrical.shape[0]-1
    n = model.n
    v = y0[50:50+n].copy()
    x = y0[50+n:50+2*n].copy()
    loc = np.zeros((nsteps+1, 2*n))
    loc[0] = np.r_[v,x]
    lam = np.zeros(nsteps+1)
    lam[0] = 1-model.length_scale*np.mean(x)
    amplitude, onset, offset = pulse
    for j in range(nsteps):
        t = start + j*dt_ms
        act = (float(np.clip((1-electrical[j,46])/model.activation_scale,0,1))
               if model.calcium_motor else 0.)
        p = model.sarcomeres.model_params
        prestrain = amplitude if onset <= t < offset else 0.0
        f_det = (model.sarcomeres.force_single(x,v,act) -
                 p.k_l/(p.mu*p.N)*(np.sum(x)-prestrain))
        eta = (noise[0][:,j] + noise[1][:,j]*
               model.sarcomeres.n_active_motors(v)*act)
        # Identical ordering to the original Model.integrate().
        v_new = v + (f_det+eta)*dt_ms/1000
        x_new = x + v*dt_ms/1000
        v,x = v_new,x_new
        loc[j+1] = np.r_[v,x]
        lam[j+1] = 1-model.length_scale*np.mean(x)
    return loc,lam


def run_stochastic(model, initial, pulse=(-5.,250.,350.), duration=460.,
                   noise_amplitudes=(2.2,9.), seed=20261018,
                   max_iter=4, convergence=1e-5, dt_ms=1.0):
    if duration % WINDOW_MS or WINDOW_MS % dt_ms:
        raise ValueError("duration and window must align with the mechanical grid")
    nsteps = int(duration/dt_ms)
    thermal,active = source_noise(model.n,nsteps,noise_amplitudes,seed,dt_ms)
    state = initial.copy()
    all_states=[state.copy()]
    all_lam=[1-model.length_scale*np.mean(state[50+model.n:])]
    convergence_records=[]
    for start in np.arange(0,duration,WINDOW_MS):
        grid = np.arange(start,start+WINDOW_MS+dt_ms/2,dt_ms)
        first = int(start/dt_ms)
        count=len(grid)-1
        local_noise = (thermal[first:first+count].T,active[first:first+count].T)
        _,_,lam0,dlam0=model.fields(state)
        guess = lam0+(grid-start)*dlam0
        for iteration in range(max_iter):
            dguess = np.gradient(guess, dt_ms)
            def e_rhs(t, y):
                lam=float(np.interp(t,grid,guess)) if model.electrical_feedback else 1.
                dlam=float(np.interp(t,grid,dguess)) if model.electrical_feedback else 0.
                with torch.no_grad():
                    return model.electrical(torch.from_numpy(y),torch.tensor(t),
                                            torch.tensor(lam),torch.tensor(dlam)).numpy()
            sol=solve_ivp(e_rhs,(grid[0],grid[-1]),state[:50],method="BDF",
                          t_eval=grid,max_step=dt_ms,rtol=1e-6,atol=1e-8)
            if not sol.success:
                raise RuntimeError(f"electrical solve failed at {start} ms: {sol.message}")
            local,lam=mechanical_window(model,start,state,sol.y.T,
                                        local_noise,pulse,dt_ms=dt_ms)
            mismatch=float(np.max(np.abs(lam-guess)))
            guess=lam
            if mismatch < convergence:
                break
        convergence_records.append({"start_ms":float(start),
                                    "iterations":iteration+1,
                                    "max_lambda_update":mismatch})
        seg=np.c_[sol.y.T,local]
        all_states.extend(seg[1:])
        all_lam.extend(lam[1:])
        state=seg[-1].copy()
    t=np.arange(0,duration+dt_ms/2,dt_ms)
    state=np.asarray(all_states)
    lam=np.asarray(all_lam)
    return {"time_ms":t,"state":state,"lambda":lam,
            "local_lambda":1-model.length_scale*state[:,50+model.n:],
            "convergence":convergence_records,
            "max_iteration_update":max(x["max_lambda_update"] for x in convergence_records)}


def main():
    torch.set_num_threads(1)
    y0=initial_state()
    deterministic=run_stochastic(ClosedLoop(),y0,noise_amplitudes=(0.,0.))
    reference=np.load(HERE/"sarcomere_closed_loop.npz")
    tt=deterministic["time_ms"].astype(int)
    native=reference["closed_pulse_state"][tt]
    ref_lam=reference["closed_pulse_lambda"][tt]
    errors={"voltage_mV":float(np.max(np.abs(deterministic["state"][:,0]-native[:,0]))),
            "calcium_uM":float(1000*np.max(np.abs(deterministic["state"][:,5]-native[:,5]))),
            "lambda":float(np.max(np.abs(deterministic["lambda"]-ref_lam)))}
    print("deterministic_split_vs_monolithic",errors,flush=True)
    stochastic=run_stochastic(ClosedLoop(),y0)
    report={"method":"partitioned 20-ms electrical BDF / 1-ms Euler-Maruyama refinement of published sarcomere SDE with fixed-point coupling",
            "sarcomere_original_quickstart_dt_ms":2.0,
            "numerical_refinement_dt_ms":1.0,
            "original_noise_amplitudes":[2.2,9.],
            "deterministic_split_vs_monolithic_max_abs":errors,
            "stochastic_max_iteration_update":stochastic["max_iteration_update"],
            "windows_at_iteration_limit":int(sum(x["iterations"]==4 and x["max_lambda_update"]>=1e-5 for x in stochastic["convergence"])),
            "stochastic_lambda_range":[float(stochastic["lambda"].min()),float(stochastic["lambda"].max())],
            "stochastic_local_lambda_std_peak":float(stochastic["local_lambda"].std(axis=1).max()),
            "stochastic_vs_deterministic_V_max_mV":float(np.max(np.abs(stochastic["state"][:,0]-deterministic["state"][:,0]))),
            "interpretation_limit":"Exploratory Land thin-filament activation and length scale; no force-unit matching or synchronized AP/Ca/sarcomere experimental validation"}
    (HERE/"sarcomere_stochastic_cosim.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    np.savez_compressed(HERE/"sarcomere_stochastic_cosim.npz",
                        time_ms=stochastic["time_ms"],state=stochastic["state"],
                        lambda_mean=stochastic["lambda"],local_lambda=stochastic["local_lambda"],
                        deterministic_state=deterministic["state"],
                        deterministic_lambda=deterministic["lambda"])
    fig,ax=plt.subplots(3,1,figsize=(9,8),sharex=True,layout="constrained")
    t=stochastic["time_ms"]
    for j,label in ((0,"V (mV)"),(5,"Ca_i (mM)")):
        row=0 if j==0 else 1
        ax[row].plot(t,deterministic["state"][:,j],color="#777",label="deterministic")
        ax[row].plot(t,stochastic["state"][:,j],color="#b43b4d",label="stochastic",lw=1)
        ax[row].set_ylabel(label)
    ax[2].plot(t,stochastic["local_lambda"][:,::2],lw=.6,alpha=.45)
    ax[2].plot(t,stochastic["lambda"],color="black",lw=1.7,label="mean")
    ax[2].set_ylabel("Local stretch λ")
    for a in ax:
        a.axvspan(250,350,color="#ddd",alpha=.3)
        a.grid(alpha=.2)
    ax[0].legend(frameon=False)
    ax[2].legend(frameon=False)
    ax[-1].set_xlabel("Time (ms)")
    fig.savefig(HERE/"sarcomere_stochastic_cosim.png",dpi=170)
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    main()
