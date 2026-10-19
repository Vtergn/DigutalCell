"""Mechanism-focused figure for the synthetic dual-amplitude gate."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from full_model import forward, infer, infer_nonlinear

OUT=Path(__file__).resolve().parent/"results"
Z=np.load(OUT/"amplitude_per_cell.npz")
F=np.geomspace(1e5,15e6,21)
A0=B0=7.8e-6;C0=2.7e-6
TRUTH=np.array([4.7,.65,2.5,2.0])


def predicted(freq,theta,field):
    return forward(freq,theta[:3],A0,B0,C0,field_gain=field,
                   nonlinear_kappa=theta[3] if len(theta)==4 else 0)[:,0]/A0


def main():
    one=forward(F,TRUTH[:3],A0,B0,C0,nonlinear_kappa=TRUTH[3])
    linear=np.r_[infer(F,one,A0),0]
    f3=np.array([.21204,2.02141,15.0])*1e6
    two_dims=np.vstack([forward(f3,TRUTH[:3],A0,B0,C0,field_gain=e**2,
                               nonlinear_kappa=TRUTH[3]) for e in [.8,1.2]])
    two=infer_nonlinear(np.tile(f3,2),two_dims,
                        np.repeat([.8**2,1.2**2],3),A0)
    fig,axs=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    for ax,field,label in [(axs[0,0],1.0,"Observed amplitude"),
                           (axs[0,1],1.4**2,"Sealed higher amplitude")]:
        ax.semilogx(F/1e6,predicted(F,TRUTH,field),color="#191970",lw=2,label="Nonlinear truth")
        ax.semilogx(F/1e6,predicted(F,linear,field),color="#b44d2d",ls="--",lw=2,label="One-amplitude linear fit")
        ax.semilogx(F/1e6,predicted(F,two,field),color="#147d64",ls=":",lw=2,label="Two-amplitude nonlinear fit")
        ax.set(xlabel="Frequency (MHz)",ylabel="Elongation ratio λ",title=label)
        ax.legend(fontsize=7)
    single=Z["six_high_E__nonlinear__target"]
    dual=Z["three_two_E__nonlinear__target"]
    ax=axs[1,0]
    ax.scatter(single,dual,s=22,alpha=.6,color="#147d64")
    lim=max(np.quantile(np.r_[single,dual],.99)*1.1,.08)
    ax.plot([0,lim],[0,lim],color="#555",ls="--",lw=1)
    ax.set(xlim=(0,lim),ylim=(0,lim),xlabel="Six frequencies, one amplitude: λ MAE",
           ylabel="Three frequencies, two amplitudes: λ MAE",
           title="Paired held-out prediction (100 cells)")
    ax.text(.03,.95,f"Dual amplitude better: {np.sum(dual<single)}/100",
            transform=ax.transAxes,va="top",fontsize=9)
    ax=axs[1,1]
    for key,label,color in [
        ("full21_high_E__nonlinear__target","21 frequencies, one amplitude","#b44d2d"),
        ("six_high_E__nonlinear__target","6 frequencies, one amplitude","#907227"),
        ("three_two_E__nonlinear__target","3 frequencies × 2 amplitudes","#147d64")]:
        vals=np.sort(Z[key]);ax.plot(vals,np.arange(1,len(vals)+1)/len(vals),label=label,color=color,lw=2)
    ax.set(xlabel="Held-out λ mean absolute error",ylabel="Fraction of cells ≤ error",
           title="Prediction error distribution")
    ax.legend(fontsize=7,loc="lower right")
    fig.savefig(OUT/"amplitude_gate.png",dpi=190)


if __name__=="__main__":main()
