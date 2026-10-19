"""Audit plot for published Qiang nonlinear RBC constitutive relations."""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from full_model import forward

OUT=Path(__file__).resolve().parent/"results"
D3=np.array(json.loads((OUT/"ellipsoid_gate.json").read_text())
            ["design_frequencies_MHz"]["dopt3"])*1e6


def crossers(z,form):
    vals=[]
    for i in range(len(z["truth"])):
        d=forward(D3,z["truth"][i,:3],z["a0"][i],z["a0"][i],z["c0"][i],
                  closure="fixed_thickness",field_gain=1.2**2,mechanics_form=form)
        vals.append(float(np.max(z["a0"][i]/d[:,1])))
    return np.array(vals)>=1.4


def scatter(ax,archive,form,title):
    z=np.load(OUT/archive)
    high=z["six_high_E__nonlinear__target"]
    dual=z["three_two_E__nonlinear__target"]
    crossed=crossers(z,form)
    ax.scatter(high[~crossed],dual[~crossed],s=25,alpha=.68,
               color="#7a97a3",label=f"Below transition (n={(~crossed).sum()})")
    ax.scatter(high[crossed],dual[crossed],s=25,alpha=.68,
               color="#c45b31",label=f"Crossed transition (n={crossed.sum()})")
    lim=max(.1,float(np.quantile(np.r_[high,dual],.99)*1.06))
    ax.plot([0,lim],[0,lim],ls="--",lw=1,color="#444")
    ax.set(xlim=(0,lim),ylim=(0,lim),xlabel="6 frequencies × 1 amplitude: λ MAE",
           ylabel="3 frequencies × 2 amplitudes: λ MAE",title=title)
    ax.legend(fontsize=7,loc="upper left")


def main():
    fig,axs=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    x=np.linspace(1,2.05,300)
    low_s=np.full_like(x,3.02)
    high_s=.21*np.exp(2.06*x)-.25
    low_m=np.full_like(x,6.05)
    high_m=.28*np.exp(1.98*x)+1.96
    for low,high,label,color in [(low_s,high_s,"Scientific Reports 2018","#253978"),
                                 (low_m,high_m,"Micromachines 2018","#a4412d")]:
        axs[0].plot(x[x<1.4],low[x<1.4],color=color,lw=2,label=label)
        axs[0].plot(x[x>=1.4],high[x>=1.4],color=color,lw=2)
    axs[0].axvline(1.4,color="black",ls="--",lw=1)
    axs[0].set(xlabel="Published short-axis extension ratio λQ=b₀/b",
               ylabel="Published shear modulus (µN/m)",
               title="Two measured population fits")
    axs[0].legend(fontsize=7)
    scatter(axs[1],"amplitude_per_cell_qiang2018.npz",
            "qiang2018_regularized","Scientific Reports truth")
    scatter(axs[2],"amplitude_per_cell_qiang_micromachines.npz",
            "qiang2018_micromachines_regularized","Micromachines truth")
    fig.savefig(OUT/"published_gate.png",dpi=190)


if __name__=="__main__":main()
