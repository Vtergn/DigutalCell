"""Show continuous calibration gradient and protocol-specific triggered branch."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
cohorts=(HERE/"embodied_cohort",HERE/"embodied_cohort_confirm")


def main():
    rows=[]
    for cohort_id,folder in enumerate(cohorts):
        for cell in range(1,13):
            with np.load(folder/f"cell_{cell:02d}.npz") as f:
                theta=f["theta"]
                wave=f["wave"]
            gain,beta1=map(float,theta[:2])
            rows.append((cohort_id,cell,gain,beta1,
                         float(wave[2,:,3].max()),
                         float(np.max(np.abs(wave[2,:,0]-wave[0,:,0]))),
                         bool(np.any(wave[2,450:,0]>0))))
    data=np.array(rows,dtype=float)
    audit=json.loads((HERE/"triggered_response_audit.json").read_text())
    scan=audit["gain_scan"]
    fig,axs=plt.subplots(1,2,figsize=(10,4.3),constrained_layout=True)
    x=data[:,2]*(data[:,4]-1.)
    for j,marker,label in ((0,"o","source"),(1,"s","independent cohort")):
        mask=data[:,0]==j
        axs[0].scatter(x[mask],data[mask,5],c=data[mask,3],cmap="viridis",
                       vmin=-4.1,vmax=-.9,s=48,marker=marker,
                       edgecolor="black",lw=.35,label=label)
    axs[0].annotate("extra AP in one virtual cell",xy=(x[1],data[1,5]),
                    xytext=(.064,65),arrowprops=dict(arrowstyle="->",lw=1),
                    fontsize=8)
    axs[0].set(xlabel="effective gain × peak (λ−1)",
               ylabel="max |loaded − free V| (mV)",ylim=(0,124),
               title="24-cell high-load response")
    axs[0].legend(fontsize=8)
    axs[0].grid(alpha=.18)
    gains=np.array([r["gain"] for r in scan])
    peaks=np.array([r["post_450_max_V_mV"] for r in scan])
    axs[1].plot(gains,peaks,color="#555555",lw=1)
    axs[1].scatter(gains,peaks,c=np.array([r["triggered_post_450_V_above_0"]
                                          for r in scan]),cmap="coolwarm",vmin=0,vmax=1,
                   edgecolor="black",s=50,zorder=3)
    axs[1].axhline(0,color="#999999",ls="--",lw=1)
    axs[1].axvspan(1.35,1.40,color="#d95f02",alpha=.12)
    axs[1].set(xlabel="effective stretch-electrical gain",
               ylabel="post-offset peak V (mV)",
               title="Same-cell gain scan; other parameters fixed")
    axs[1].grid(alpha=.18)
    fig.suptitle("Continuous response gradient with a protocol-specific event boundary")
    fig.savefig(HERE/"embodied_gradient_and_event_boundary.png",dpi=190)


if __name__=="__main__":
    main()
