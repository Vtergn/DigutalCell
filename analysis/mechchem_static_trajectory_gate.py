"""Independent ORd-MechChem full-waveform check under static preloads.

This is a separate within-model analysis. Static preloads here are not the
dynamic 300-ms pulses in the ToR-ORd/SAC-Land cohort; coordinate transfer is
therefore neither attempted nor claimed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

HERE=Path(__file__).resolve().parent
ROOT=HERE/"mechchem_cross_model"/"output"
cells=[]; wave=[]
for cell in range(1,21):
    paths=[ROOT/f"cell{cell:03d}_load{j:02d}.csv" for j in range(1,5)]
    if not all(p.exists() for p in paths):
        continue
    runs=[np.loadtxt(p,delimiter=",") for p in paths]
    assert all(a.shape==(1000,4) and np.array_equal(a[:,0],np.arange(1000))
               and np.isfinite(a).all() for a in runs)
    wave.append(np.stack([a[::5,1:].T for a in runs]))
    cells.append(cell)
w=np.stack(wave)  # cell, load, channel, time
assert w.shape==(19,4,3,200)

FLOOR=np.array([1.,.01,.5])
std=w[:,0].std(axis=0)
scale=np.maximum(np.quantile(std,.75,axis=1),FLOOR)


def vec(load):
    q=(w[:,load]-w[:,load].mean(axis=0,keepdims=True))/scale[None,:,None]
    return q.reshape(len(w),-1)/np.sqrt(3*q.shape[2])


b,c=vec(0),vec(1)
y=np.concatenate([vec(2),vec(3)],axis=1)/np.sqrt(2)


def near(x):
    d=cdist(x,x)
    np.fill_diagonal(d,np.inf)
    ids=np.argsort(d,axis=1)[:,:3]
    return np.linalg.norm(y[:,None,:]-y[ids],axis=2).mean(axis=1)


d0,d1=near(b),near(np.c_[b,c])
gain=d0-d1
rng=np.random.default_rng(20261018)
bootstrap=gain[rng.integers(len(gain),size=(5000,len(gain)))].mean(axis=1)
null=[]
for _ in range(1000):
    null.append(float((d0-near(np.c_[b,c[rng.permutation(len(c))]])).mean()))
out={"model":"ORd-MechChem, static-preload independent equation family",
     "valid_cells":len(cells),"cell_ids":cells,"neighbor_count":3,
     "loads":[1.,1.05,1.075,1.10],
     "truth":"Full 0-999 ms V/Ca/T trajectories at 5-ms sampling",
     "scale_V_Ca_T":scale.tolist(),
     "baseline_future_neighbor_distance":float(d0.mean()),
     "plus_105_future_neighbor_distance":float(d1.mean()),
     "improvement":float(gain.mean()),
     "improvement_bootstrap95":np.quantile(bootstrap,[.025,.975]).tolist(),
     "n_cells_improved":int(np.sum(gain>0)),
     "shuffle_p_one_sided":float((1+np.sum(np.array(null)>=gain.mean()))/1001),
     "caveat":"Exploratory 19 cells, static lengths with no direct SAC. Not the same dynamic protocol, so it does not validate ToR-ORd/SAC-Land trajectory coordinates."}
(HERE/"mechchem_static_trajectory_gate.json").write_text(
    json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(out,indent=2))
