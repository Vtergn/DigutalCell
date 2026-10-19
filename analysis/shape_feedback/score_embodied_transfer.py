"""Freeze source-cohort scaling and predictors; evaluate a second LHS cohort."""
from __future__ import annotations

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
SOURCE=HERE/"embodied_cohort"
CONFIRM=HERE/"embodied_cohort_confirm"


def read_cohort(folder):
    manifest=json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    theta=[];wave=[];ids=[]
    for i in range(1,manifest["n_planned"]+1):
        path=folder/f"cell_{i:02d}.npz"
        if path.exists():
            with np.load(path) as f:
                theta.append(f["theta"])
                wave.append(f["wave"])
                ids.append(i)
    assert len(ids)>=10, f"Not enough completed cells in {folder}: {len(ids)}"
    return np.stack(theta),np.stack(wave),ids,manifest


def pairwise_future_error(x,y,k=2):
    dist=cdist(x,x)
    np.fill_diagonal(dist,np.inf)
    near=np.argsort(dist,axis=1)[:,:k]
    return np.linalg.norm(y[:,None]-y[near],axis=2).mean(axis=1)


def ci(values,rng):
    draw=values[rng.integers(len(values),size=(5000,len(values)))].mean(axis=1)
    return [float(q) for q in np.quantile(draw,[.025,.975])]


def main():
    ts,ws,source_ids,ms=read_cohort(SOURCE)
    tc,wc,confirm_ids,mc=read_cohort(CONFIRM)
    assert ms["protocols"]==mc["protocols"]
    assert ws.shape[1:]==wc.shape[1:]==(4,601,4)
    rs=ws[:,:,::5,:].transpose(0,1,3,2)
    rc=wc[:,:,::5,:].transpose(0,1,3,2)
    ds=rs[:,1:]-rs[:,0:1]
    dc=rc[:,1:]-rc[:,0:1]
    floors=np.array([1.,.005,1.,.01])
    scale=np.maximum(np.quantile(ds[:,0].std(axis=0),.75,axis=1),floors)

    def block(a,ch=slice(None)):
        b=a[:,ch]/scale[ch][None,:,None]
        return b.reshape(len(b),-1)/np.sqrt(b.shape[1]*b.shape[2])

    def design(raw,delta):
        base=block(raw[:,0])
        cal=block(delta[:,0])
        future=np.c_[block(delta[:,1]),block(delta[:,2])]/np.sqrt(2)
        return base,cal,future

    bs,cs,ys=design(rs,ds)
    bc,cc,yc=design(rc,dc)
    all_source=np.c_[bs,cs]
    all_confirm=np.c_[bc,cc]
    duplicate_source=np.c_[bs,bs]
    duplicate_confirm=np.c_[bc,bc]
    comparisons={"baseline":(bs,bc),"baseline_plus_challenge":
                 (all_source,all_confirm),"baseline_duplicated":
                 (duplicate_source,duplicate_confirm)}
    predictions={}
    for name,(train_x,test_x) in comparisons.items():
        pred=Ridge(alpha=1.).fit(train_x,ys).predict(test_x)
        predictions[name]=np.linalg.norm(pred-yc,axis=1)
    neigh_base=pairwise_future_error(bc,yc)
    neigh_cal=pairwise_future_error(all_confirm,yc)
    future_distance=pdist(yc)
    distance_rank_correlation={
        "baseline":float(spearmanr(pdist(bc),future_distance).statistic),
        "baseline_plus_challenge":float(spearmanr(pdist(all_confirm),
                                                  future_distance).statistic)}
    neighbor_k_sensitivity={str(k):{
        "baseline":float(pairwise_future_error(bc,yc,k).mean()),
        "challenge":float(pairwise_future_error(all_confirm,yc,k).mean())}
        for k in (1,2,3)}
    rng=np.random.default_rng(2228)
    observed=float(np.mean(neigh_base-neigh_cal))
    null=[]
    for _ in range(1000):
        shuffled=cc[rng.permutation(len(cc))]
        null.append(float(np.mean(neigh_base-
                       pairwise_future_error(np.c_[bc,shuffled],yc))))
    pca=PCA(n_components=4).fit(cs)
    source_pc=pca.transform(cs)
    confirm_pc=pca.transform(cc)
    rho=np.array([[spearmanr(confirm_pc[:,j],tc[:,p]).statistic
                   for p in range(4)] for j in range(4)])
    report={"source_seed":ms["seed"],"confirmation_seed":mc["seed"],
            "source_cells":source_ids,"confirmation_cells":confirm_ids,
            "source_fixed_scale_V_Ca_T_lambda":scale.tolist(),
            "confirmation_neighbor_future_distance_baseline":float(neigh_base.mean()),
            "confirmation_neighbor_future_distance_challenge":float(neigh_cal.mean()),
            "confirmation_pairwise_distance_rank_correlation":
                distance_rank_correlation,
            "confirmation_neighbor_k_sensitivity":neighbor_k_sensitivity,
            "confirmation_neighbor_future_distance_median":
                {"baseline":float(np.median(neigh_base)),
                 "challenge":float(np.median(neigh_cal))},
            "confirmation_neighbor_per_cell":
                {"baseline":neigh_base.tolist(),"challenge":neigh_cal.tolist()},
            "confirmation_neighbor_gain_CI95":ci(neigh_base-neigh_cal,rng),
            "confirmation_neighbor_cells_improved":int(np.sum(neigh_base>neigh_cal)),
            "confirmation_neighbor_permutation_p":float((1+np.sum(np.array(null)>=observed))/1001),
            "source_trained_ridge_confirmation_future_distance":
                 {k:float(v.mean()) for k,v in predictions.items()},
            "source_trained_ridge_confirmation_future_distance_median":
                 {k:float(np.median(v)) for k,v in predictions.items()},
            "source_trained_ridge_confirmation_error_per_cell":
                 {k:v.tolist() for k,v in predictions.items()},
            "source_trained_ridge_gain_CI95":ci(predictions["baseline"]-
                                               predictions["baseline_plus_challenge"],rng),
            "confirmation_ridge_cells_improved":int(np.sum(predictions["baseline"]>
                                      predictions["baseline_plus_challenge"])),
            "source_PCA_variance":pca.explained_variance_ratio_.tolist(),
            "confirmation_spearman_source_PC_by_parameter":rho.tolist(),
            "note":"Same-equation and same-protocol independent parameter cohort; not cross-model or protocol-family validation."}
    (HERE/"embodied_transfer_score.json").write_text(json.dumps(report,indent=2),encoding="utf-8")

    fig,axs=plt.subplots(1,2,figsize=(9,4),constrained_layout=True)
    for ax,baseline,challenge,name in zip(axs,
        (neigh_base,predictions["baseline"]),
        (neigh_cal,predictions["baseline_plus_challenge"]),
        ("Neighbour geometry","Frozen source-trained ridge")):
        ax.scatter(baseline,challenge,c=tc[:,0],cmap="viridis",s=48,
                   edgecolor="black",lw=.3)
        upper=max(np.max(baseline),np.max(challenge))*1.08
        ax.plot([0,upper],[0,upper],color="gray",ls="--",lw=1)
        ax.set(xlim=(0,upper),ylim=(0,upper),xlabel="Baseline only",
               ylabel="Baseline + challenge",title=name)
    fig.suptitle("Independent cohort: below diagonal means better future response")
    fig.savefig(HERE/"embodied_transfer_confirmation.png",dpi=180)
    fig,axs=plt.subplots(1,2,figsize=(9,4),constrained_layout=True)
    for ax,p,name in zip(axs,(0,1),("Stretch-electrical gain","Land beta1")):
        sc=ax.scatter(confirm_pc[:,0],confirm_pc[:,1],c=tc[:,p],
                      s=48,cmap="viridis",edgecolor="black",lw=.3)
        fig.colorbar(sc,ax=ax,shrink=.8)
        ax.set(xlabel="Frozen source PC1",ylabel="Frozen source PC2",title=name)
    fig.suptitle("Parameter gradients under frozen calibration coordinates")
    fig.savefig(HERE/"embodied_transfer_frozen_space.png",dpi=180)
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
