"""Score first-beat functional trajectory geometry of the 51-state cohort.

The no-load control permits true within-cell stress-induced differences in all
four observed channels: voltage, Ca, tension, and endogenous length.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument("--out",type=Path,default=HERE/"embodied_cohort")
OUT=parser.parse_args().out
manifest=json.loads((OUT/"manifest.json").read_text(encoding="utf-8"))
cells=[];theta=[];wave=[]
for i in range(1,manifest["n_planned"]+1):
    path=OUT/f"cell_{i:02d}.npz"
    if not path.exists():
        continue
    with np.load(path) as f:
        theta.append(f["theta"])
        wave.append(f["wave"])
        cells.append(i)
theta=np.stack(theta)
w=np.stack(wave)  # cell, protocol, time, (V,Ca,T,lambda)
assert w.shape[1:]==(4,601,4) and len(cells)>=8
n=len(cells)
times=np.arange(0,601,5)
raw=w[:,:,times,:].transpose(0,1,3,2)
induced=raw[:,1:]-raw[:,0:1]

FLOOR=np.array([1.,.005,1.,.01])  # mV, microM, kPa, unit length
cal_std=induced[:,0].std(axis=0)
scale=np.maximum(np.quantile(cal_std,.75,axis=1),FLOOR)


def block(z,channels=slice(None)):
    a=z[:,channels]/scale[channels][None,:,None]
    return a.reshape(len(a),-1)/np.sqrt(a.shape[1]*a.shape[2])


base=block(raw[:,0])
cal=block(induced[:,0])
cal_e=block(induced[:,0],slice(0,3))
cal_l=block(induced[:,0],slice(3,4))
future=np.concatenate([block(induced[:,1]),block(induced[:,2])],axis=1)/np.sqrt(2)
future_by_channel={label:np.concatenate([
    block(induced[:,1],slice(j,j+1)),
    block(induced[:,2],slice(j,j+1))],axis=1)/np.sqrt(2)
    for j,label in enumerate(("V","Ca","T","lambda"))}


def neighbor_error(x,y,k=2):
    distance=cdist(x,x)
    np.fill_diagonal(distance,np.inf)
    near=np.argsort(distance,axis=1)[:,:k]
    return np.linalg.norm(y[:,None,:]-y[near],axis=2).mean(axis=1)


def loo_ridge(x,y,alpha=1.):
    result=np.zeros_like(y)
    for i in range(len(x)):
        train=np.arange(len(x))!=i
        result[i]=Ridge(alpha=alpha).fit(x[train],y[train]).predict(x[i:i+1])[0]
    return np.linalg.norm(result-y,axis=1)


def ci(d,seed=20261019):
    rng=np.random.default_rng(seed)
    means=d[rng.integers(len(d),size=(5000,len(d)))].mean(axis=1)
    return np.quantile(means,[.025,.975]).tolist()


conditions={"baseline":base,"baseline_plus_all":np.c_[base,cal],
            "baseline_plus_electrical_Ca_T":np.c_[base,cal_e],
            "baseline_plus_length_only":np.c_[base,cal_l],
            "baseline_duplicated":np.c_[base,base]}
neighbor={k:neighbor_error(x,future) for k,x in conditions.items()}
ridge={k:loo_ridge(x,future) for k,x in conditions.items()}
pairwise_rank={k:float(spearmanr(pdist(x),pdist(future)).statistic)
               for k,x in conditions.items()}
neighbor_by_channel={label:{k:float(neighbor_error(x,y).mean())
                    for k,x in conditions.items()}
                    for label,y in future_by_channel.items()}
ridge_by_channel={label:{k:float(loo_ridge(x,y).mean())
                 for k,x in conditions.items()}
                 for label,y in future_by_channel.items()}

rng=np.random.default_rng(4002)
null=[]
for _ in range(1000):
    sh=cal[rng.permutation(n)]
    null.append(float((neighbor["baseline"]-
                       neighbor_error(np.c_[base,sh],future)).mean()))

pc=PCA(n_components=min(4,n-1)).fit(cal)
z=pc.transform(cal)
rho=np.array([[spearmanr(z[:,j],theta[:,p]).statistic for p in range(4)]
              for j in range(z.shape[1])])

def magnitude(channel,protocol):
    a=induced[:,protocol,channel]
    return np.max(np.abs(a),axis=1)

summary={"valid_cells":n,"cell_ids":cells,"parameter_names":manifest["parameter_names"],
         "protocols":manifest["protocols"],"first_beat_not_prepaced_per_cell":True,
         "scale_V_Ca_T_lambda":scale.tolist(),
         "lambda_max_calibration":np.max(raw[:,1,3],axis=1).tolist(),
         "lambda_max_target_high":np.max(raw[:,2,3],axis=1).tolist(),
         "calibration_max_abs_change_median":{"V_mV":float(np.median(magnitude(0,0))),
              "Ca_uM":float(np.median(magnitude(1,0))),
              "T_kPa":float(np.median(magnitude(2,0))),
              "lambda":float(np.median(magnitude(3,0)))},
         "future_neighbor_distance":{k:float(v.mean()) for k,v in neighbor.items()},
         "pairwise_distance_rank_correlation":pairwise_rank,
         "future_neighbor_distance_median":{k:float(np.median(v)) for k,v in neighbor.items()},
         "neighbor_per_cell":{k:v.tolist() for k,v in neighbor.items()},
         "future_neighbor_distance_by_channel":neighbor_by_channel,
         "future_neighbor_gain_all_vs_base95":ci(neighbor["baseline"]-neighbor["baseline_plus_all"]),
         "n_neighbor_cells_improved":int(np.sum(neighbor["baseline"]>neighbor["baseline_plus_all"])),
         "neighbor_permutation_p":float((1+np.sum(np.array(null)>=
             (neighbor["baseline"]-neighbor["baseline_plus_all"]).mean()))/1001),
         "loo_ridge_distance":{k:float(v.mean()) for k,v in ridge.items()},
         "loo_ridge_distance_median":{k:float(np.median(v)) for k,v in ridge.items()},
         "ridge_per_cell":{k:v.tolist() for k,v in ridge.items()},
         "loo_ridge_distance_by_channel":ridge_by_channel,
         "loo_ridge_gain_all_vs_base95":ci(ridge["baseline"]-ridge["baseline_plus_all"]),
         "n_ridge_cells_improved":int(np.sum(ridge["baseline"]>ridge["baseline_plus_all"])),
         "pca_variance":pc.explained_variance_ratio_.tolist(),
         "spearman_pc_by_parameter":rho.tolist(),
         "caveat":"Exploratory 12-cell same-equation cohort, first beat from common nominal state; K and viscosity uncalibrated. No intrinsic-dimension or biological-state inference."}
(OUT/"score.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")

fig,axs=plt.subplots(4,1,figsize=(9,9),sharex=True,constrained_layout=True)
example=int(np.argsort(theta[:,0])[len(theta)//2])
labels=["V (mV)","Ca (µM)","T (kPa)","Length λ"]
for channel,ax in enumerate(axs):
    for j,color in enumerate(["#555555","#1b9e77","#d95f02","#7570b3"]):
        ax.plot(np.arange(601),w[example,j,:,channel],color=color,lw=1.1,
                label=manifest["protocols"][j][0])
    ax.set_ylabel(labels[channel])
axs[0].legend(fontsize=7,ncol=2)
axs[-1].set_xlabel("Time (ms)")
fig.suptitle(f"Stress-controlled 51-state cell {cells[example]}")
fig.savefig(OUT/"example_trajectories.png",dpi=170)

fig,axs=plt.subplots(5,1,figsize=(9,10),sharex=True,constrained_layout=True)
t=np.arange(601)
colors=["#1b9e77","#d95f02","#7570b3"]
for j,(_,pressure,onset,offset) in enumerate(manifest["protocols"][1:]):
    stress=np.where((t>=onset)&(t<offset),pressure,0.)
    axs[0].plot(t,stress,color=colors[j],lw=1.7,
                label=manifest["protocols"][j+1][0])
    for channel in range(4):
        axs[channel+1].plot(t,w[example,j+1,:,channel]-w[example,0,:,channel],
                            color=colors[j],lw=1.3)
for ax,lab in zip(axs,("External stress (kPa)","ΔV (mV)","ΔCa (µM)",
                        "ΔT (kPa)","Δλ")):
    ax.set_ylabel(lab)
    ax.grid(alpha=.17)
axs[0].legend(fontsize=7,ncol=3)
axs[-1].set(xlabel="Time (ms)",xlim=(200,600))
fig.suptitle(f"Paired intervention response; virtual cell {cells[example]}")
fig.savefig(OUT/"example_induced_response.png",dpi=170)

fig,axs=plt.subplots(1,2,figsize=(9,4),constrained_layout=True)
for ax,(p,label) in zip(axs,[(0,"Stretch-electrical gain"),(1,"Land beta1")]):
    sc=ax.scatter(z[:,0],z[:,1],c=theta[:,p],cmap="viridis",s=48)
    fig.colorbar(sc,ax=ax,shrink=.8)
    ax.set(xlabel="Calibration PC1",ylabel="Calibration PC2",title=label)
fig.savefig(OUT/"trajectory_parameter_space.png",dpi=170)

fig,axs=plt.subplots(1,2,figsize=(8.5,4),constrained_layout=True)
for ax,metric,title in zip(axs,(neighbor,ridge),
                           ("Two-neighbour future distance",
                            "Leave-one-cell-out future error")):
    x=metric["baseline"]
    y=metric["baseline_plus_all"]
    ax.scatter(x,y,c=theta[:,0],cmap="viridis",s=42,edgecolor="black",lw=.3)
    bound=max(float(x.max()),float(y.max()))*1.07
    ax.plot([0,bound],[0,bound],color="#777777",ls="--",lw=1)
    ax.set(xlabel="Baseline only",ylabel="Baseline + challenge",title=title,
           xlim=(0,bound),ylim=(0,bound))
fig.suptitle("Points below diagonal gain functional predictivity")
fig.savefig(OUT/"future_predictivity.png",dpi=180)
print(json.dumps({k:v for k,v in summary.items() if k in [
    "valid_cells","calibration_max_abs_change_median","future_neighbor_distance",
    "future_neighbor_gain_all_vs_base95","n_neighbor_cells_improved",
    "neighbor_permutation_p","loo_ridge_distance","loo_ridge_gain_all_vs_base95",
    "n_ridge_cells_improved","pca_variance","spearman_pc_by_parameter"]},indent=2))
