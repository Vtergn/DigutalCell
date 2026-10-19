"""Post-hoc audit of source-cell influence on frozen ridge transfer."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

HERE=Path(__file__).resolve().parent


def read(folder):
    wave=[]
    for i in range(1,13):
        with np.load(folder/f"cell_{i:02d}.npz") as f:
            wave.append(f["wave"])
    return np.stack(wave)[:,:,::5,:].transpose(0,1,3,2)


def main():
    src=read(HERE/"embodied_cohort")
    test=read(HERE/"embodied_cohort_confirm")
    ds=src[:,1:]-src[:,0:1]
    dt=test[:,1:]-test[:,0:1]
    scale=np.maximum(np.quantile(ds[:,0].std(axis=0),.75,axis=1),
                     np.array([1.,.005,1.,.01]))
    def block(a):
        x=a/scale[None,:,None]
        return x.reshape(len(x),-1)/np.sqrt(x.shape[1]*x.shape[2])
    def design(raw,delta):
        b=block(raw[:,0]);c=block(delta[:,0])
        f=np.c_[block(delta[:,1]),block(delta[:,2])]/np.sqrt(2)
        return np.c_[b,c],f
    xs,ys=design(src,ds)
    xt,yt=design(test,dt)
    full=np.linalg.norm(Ridge(alpha=1.).fit(xs,ys).predict(xt)-yt,axis=1)
    rows=[]
    for j in range(len(xs)):
        keep=np.arange(len(xs))!=j
        pred=Ridge(alpha=1.).fit(xs[keep],ys[keep]).predict(xt)
        err=np.linalg.norm(pred-yt,axis=1)
        rows.append({"removed_source_cell":j+1,"confirmation_mean_error":float(err.mean()),
                     "confirmation_median_error":float(np.median(err))})
    report={"all_source_mean_error":float(full.mean()),
            "leave_one_source_cell_out":rows,
            "warning":"Exploratory audit performed after inspecting confirmation errors; removal is not an unbiased model-selection result."}
    (HERE/"source_influence_audit.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
