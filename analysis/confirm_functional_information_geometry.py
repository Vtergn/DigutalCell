"""Frozen metrics and source-trained latent model on a new synthetic cohort.

This script intentionally has no hyperparameter selection on the new cohort.
The new cohort shares the *same* equations and parameter ranges but uses an
independent random seed and independently prepaced virtual cells.
"""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cross_decomposition import PLSRegression

ROOT=Path(__file__).resolve().parent/"human_bidirectional_output"
A=np.loadtxt(ROOT/"feedback_13param_cohort.csv",delimiter=",")
B=np.loadtxt(ROOT/"feedback_13param_confirm_cohort.csv",delimiter=",")
assert A.shape[1]==B.shape[1]==113
A=A[A[:,-1]==1]; B=B[B[:,-1]==1]
assert len(A)==64 and len(B)>=50
BF=np.array([.1,.001,.1])
CF=np.array([.1,.1,.1,.001,.001,.1,.1,.1,.1,.1,.0005,.0005])
YF=np.array([.1,.1,.001,.1,.1,.0005])


def unpack(a):
    f=a[:,16:-1].reshape(len(a),12,8).transpose(0,2,1)
    return a[:,13:16], f[:,:5],f[:,5:8,:][:,:,[0,2,3,6,8,10]]


ba,ca,ya=unpack(A)
bb,cb,yb=unpack(B)


def obs(base,challenge,protocols):
    means=ba.mean(axis=0); sd=ba.std(axis=0); active=sd>=BF
    x=(base[:,active]-means[active])/np.maximum(sd[active],BF[active])
    parts=[x/np.sqrt(x.shape[1])]
    for k in protocols:
        means=ca[:,k].mean(axis=0); sd=ca[:,k].std(axis=0); active=sd>=CF
        x=(challenge[:,k,active]-means[active])/np.maximum(sd[active],CF[active])
        parts.append(x/np.sqrt(x.shape[1]))
    return np.concatenate(parts,axis=1)


def response(y):
    parts=[]
    for k in range(3):
        means=ya[:,k].mean(axis=0); sd=ya[:,k].std(axis=0); active=sd>=YF
        x=(y[:,k,active]-means[active])/np.maximum(sd[active],YF[active])
        parts.append(x/np.sqrt(3*x.shape[1]))
    return np.concatenate(parts,axis=1)


za,zb=response(ya),response(yb)


def neighbor_distance(x,z):
    d=cdist(x,x)
    np.fill_diagonal(d,np.inf)
    nn=np.argsort(d,axis=1)[:,:5]
    return np.sqrt(((z[:,None,:]-z[nn])**2).sum(axis=2)).mean(axis=1)


def rbf_predict(xtrain,ytrain,xtest):
    mu=xtrain.mean(axis=0);sd=xtrain.std(axis=0);keep=sd>1e-10
    x=(xtrain[:,keep]-mu[keep])/sd[keep]
    q=(xtest[:,keep]-mu[keep])/sd[keep]
    sq=cdist(x,x,"sqeuclidean")
    nz=sq[sq>1e-12]
    scale=float(np.median(nz)) if len(nz) else 1.
    k=np.exp(-sq/scale)
    z=np.exp(-cdist(q,x,"sqeuclidean")/scale)
    ym=ytrain.mean(axis=0)
    return ym+z@np.linalg.solve(k+.02*np.eye(len(x)),ytrain-ym)


conditions={"baseline":[],"early150":[0],"mid300":[2],
            "late600":[1],"mid+late":[2,1]}
dist={label:neighbor_distance(obs(bb,cb,prot),zb)
      for label,prot in conditions.items()}
base=dist["baseline"]
gain=base-dist["mid300"]
rng=np.random.default_rng(90210)
bootstrap=[gain[rng.integers(len(gain),size=len(gain))].mean()
           for _ in range(5000)]
perm=[]
for _ in range(1000):
    sh=cb[rng.permutation(len(cb))]
    perm.append((base-neighbor_distance(obs(bb,sh,[2]),zb)).mean())

prediction={}
for label,prot in conditions.items():
    xa,xb=obs(ba,ca,prot),obs(bb,cb,prot)
    prediction[label]={}
    for dim in [1,2,3,5]:
        if dim>xa.shape[1]:continue
        m=PLSRegression(n_components=dim,scale=False).fit(xa,za)
        yp=m.predict(xb)
        prediction[label][f"PLS{dim}"]=float(np.sqrt(((yp-zb)**2).sum(axis=1)).mean())
    yp=rbf_predict(xa,za,xb)
    prediction[label]["full_RBF"]=float(np.sqrt(((yp-zb)**2).sum(axis=1)).mean())

pair_gain=np.minimum(dist["mid300"],dist["late600"])-dist["mid+late"]
pair_bootstrap=[pair_gain[rng.integers(len(pair_gain),size=len(pair_gain))].mean()
                for _ in range(5000)]
duplicate_mid=neighbor_distance(obs(bb,cb,[2,2]),zb)
duplicate_late=neighbor_distance(obs(bb,cb,[1,1]),zb)
unique_pair_gain=np.minimum(duplicate_mid,duplicate_late)-dist["mid+late"]
unique_pair_bootstrap=[unique_pair_gain[rng.integers(len(unique_pair_gain),
                                  size=len(unique_pair_gain))].mean()
                       for _ in range(5000)]
out={"source_cells":len(A),"new_cells":len(B),
     "new_seed":20261002,
     "new_cohort_type":"Independent parameter draw and pre-pacing under the same source equation family; not independent model truth or experimental data.",
     "neighbor_future_distance":{k:float(v.mean()) for k,v in dist.items()},
     "mid_improvement_vs_baseline":float(gain.mean()),
     "mid_improvement_bootstrap95":[float(np.quantile(bootstrap,.025)),
                                     float(np.quantile(bootstrap,.975))],
     "mid_challenge_permutation_p":float((1+np.sum(np.array(perm)>=gain.mean()))/1001),
     "pair_gain_over_better_single":float(pair_gain.mean()),
     "pair_gain_bootstrap95":[float(np.quantile(pair_bootstrap,.025)),
                              float(np.quantile(pair_bootstrap,.975))],
     "pair_gain_over_better_duplicated_single":float(unique_pair_gain.mean()),
     "pair_gain_over_duplicate_bootstrap95":[float(np.quantile(unique_pair_bootstrap,.025)),
                                                float(np.quantile(unique_pair_bootstrap,.975))],
     "source_trained_new_cohort_prediction":prediction}
path=ROOT/"feedback_functional_information_confirmation.json"
path.write_text(json.dumps(out,indent=2),encoding="utf-8")
print("n",len(B),"neighbor",out["neighbor_future_distance"])
print("gain",out["mid_improvement_vs_baseline"],out["mid_improvement_bootstrap95"],
      "shuffle p",out["mid_challenge_permutation_p"])
print("pair gain",out["pair_gain_over_better_single"])
print("prediction",prediction)
