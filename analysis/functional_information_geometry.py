"""Small, falsifiable feasibility test for intervention-conditioned function states.

All protocol choices, dimensions, and metrics are fixed in code. Outcomes from
the three future protocols are used only for scoring / supervised CV, never for
constructing observation-space neighbors or PCA features. The 64-cell cohort
has been explored previously; these results are hypothesis-generating.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent / "human_bidirectional_output"
SRC = np.loadtxt(ROOT / "feedback_13param_cohort.csv", delimiter=",")
ALT = np.loadtxt(ROOT / "feedback_13param_discrepancy.csv", delimiter=",")
assert SRC.shape == ALT.shape == (64, 113)
assert np.all(SRC[:, -1] == 1) and np.all(ALT[:, -1] == 1)
PARAM_RANGE = np.array([.2,.3,.3,.3,.2,.4,1.2,.11,.4,1,1,3.2,40])
BASE_FLOOR = np.array([.1,.001,.1])
CHALLENGE_FLOOR = np.array([.1,.1,.1,.001,.001,.1,.1,.1,.1,.1,.0005,.0005])
FUTURE_FLOOR = np.array([.1,.1,.001,.1,.1,.0005])
OUTCOLS = [0,2,3,6,8,10]
PROTOCOL = ["early150_110", "late600_110", "mid300_110",
            "early150_105", "late600_105"]


def unpack(a):
    f = a[:, 16:-1].reshape(len(a), 12, 8).transpose(0,2,1)
    return a[:, :13], a[:, 13:16], f[:, :5], f[:, 5:8, :][:, :, OUTCOLS]


THETA, BASE, CHALLENGE, FUTURE = unpack(SRC)
_, ALT_BASE, ALT_CHALLENGE, ALT_FUTURE = unpack(ALT)


def scales(base, challenge, future):
    bs = base.std(axis=0)
    cs = challenge.std(axis=0)
    fs = future.std(axis=0)
    return (bs, bs >= BASE_FLOOR,
            cs, cs >= CHALLENGE_FLOOR,
            fs, fs >= FUTURE_FLOOR)


SCALE = scales(BASE, CHALLENGE, FUTURE)


def obs_components(base, challenge, protocols, scale=SCALE):
    bs, ba, cs, ca, _, _ = scale
    b = (base[:, ba]-BASE[:, ba].mean(axis=0)) / np.maximum(bs[ba], BASE_FLOOR[ba])
    groups = [b/np.sqrt(b.shape[1])]
    for k in protocols:
        active = ca[k]
        part = (challenge[:, k, active]-CHALLENGE[:, k, active].mean(axis=0))
        part /= np.maximum(cs[k, active], CHALLENGE_FLOOR[active])
        groups.append(part/np.sqrt(part.shape[1]))
    return np.concatenate(groups, axis=1)


def future_vector(future, scale=SCALE):
    _, _, _, _, fs, fa = scale
    blocks = []
    for k in range(3):
        active = fa[k]
        part = (future[:, k, active]-FUTURE[:, k, active].mean(axis=0))
        part /= np.maximum(fs[k, active], FUTURE_FLOOR[active])
        blocks.append(part/np.sqrt(3*part.shape[1]))
    return np.concatenate(blocks, axis=1)


ZF = future_vector(FUTURE)
ALT_ZF = future_vector(ALT_FUTURE)


def fold_observation(train, base, challenge, protocols):
    """Source training cells alone set means, scales, and active endpoints."""
    bmean=BASE[train].mean(axis=0)
    bsd=BASE[train].std(axis=0)
    active=bsd>=BASE_FLOOR
    b=(base[:,active]-bmean[active])/np.maximum(bsd[active],BASE_FLOOR[active])
    groups=[b/np.sqrt(b.shape[1])]
    for k in protocols:
        mean=CHALLENGE[train,k].mean(axis=0)
        sd=CHALLENGE[train,k].std(axis=0)
        ok=sd>=CHALLENGE_FLOOR
        part=(challenge[:,k,ok]-mean[ok])/np.maximum(sd[ok],CHALLENGE_FLOOR[ok])
        groups.append(part/np.sqrt(part.shape[1]))
    return np.concatenate(groups,axis=1)


def fold_future(train, future):
    groups=[]
    for k in range(3):
        mean=FUTURE[train,k].mean(axis=0)
        sd=FUTURE[train,k].std(axis=0)
        ok=sd>=FUTURE_FLOOR
        part=(future[:,k,ok]-mean[ok])/np.maximum(sd[ok],FUTURE_FLOOR[ok])
        groups.append(part/np.sqrt(3*part.shape[1]))
    return np.concatenate(groups,axis=1)


def neighbors(x, k=5):
    d = cdist(x, x)
    np.fill_diagonal(d, np.inf)
    return np.argsort(d, axis=1)[:, :k]


def future_neighbor_distance(x, future_z=ZF, k=5):
    near = neighbors(x, k)
    delta = future_z[:, None, :] - future_z[near]
    return np.sqrt((delta**2).sum(axis=2)).mean(axis=1)


def bootstrap_mean_ci(diff, seed=1, repeats=5000):
    rng = np.random.default_rng(seed)
    vals = np.array([diff[rng.integers(len(diff), size=len(diff))].mean()
                     for _ in range(repeats)])
    return [float(np.quantile(vals,.025)), float(np.quantile(vals,.975))]


def neighbor_audit(base, challenge, future_z, scale=SCALE):
    x0 = obs_components(base, challenge, [])
    d0 = future_neighbor_distance(x0, future_z)
    out = {"baseline": float(d0.mean()), "random_reference": float(
        np.mean(pdist(future_z))) }
    detail = {}
    for name, protocols in [(PROTOCOL[k], [k]) for k in range(5)] + [
        ("150+300", [0,2]), ("300+600", [2,1]),
        ("150+600", [0,1]), ("150+300+600", [0,2,1])]:
        x = obs_components(base, challenge, protocols, scale)
        d = future_neighbor_distance(x, future_z)
        diff = d0-d
        detail[name] = {"future_neighbor_distance": float(d.mean()),
                        "improvement_vs_baseline": float(diff.mean()),
                        "paired_bootstrap95": bootstrap_mean_ci(diff),
                        "n_observation_features": int(x.shape[1])}
    out["conditions"] = detail
    # Permuting challenge rows destroys cell-specific linkage while retaining
    # each challenge's marginal feature distribution.
    observed = detail["mid300_110"]["improvement_vs_baseline"]
    rng = np.random.default_rng(7001)
    null = []
    for _ in range(1000):
        shuffled = challenge[rng.permutation(len(challenge))]
        x = obs_components(base, shuffled, [2], scale)
        null.append(float((d0-future_neighbor_distance(x, future_z)).mean()))
    out["mid_shuffle_test"] = {"observed_improvement": observed,
        "null_mean": float(np.mean(null)),
        "p_one_sided": float((1+np.sum(np.array(null)>=observed))/(1+len(null)))}
    # Explicit protocol complementarity, compared with the best constituent.
    for pair, a, b in [("150+300", "early150_110", "mid300_110"),
                       ("300+600", "mid300_110", "late600_110"),
                       ("150+600", "early150_110", "late600_110")]:
        dp = future_neighbor_distance(obs_components(base, challenge,
                         [PROTOCOL.index(a),PROTOCOL.index(b)],scale),future_z)
        da = future_neighbor_distance(obs_components(base,challenge,
                         [PROTOCOL.index(a)],scale),future_z)
        db = future_neighbor_distance(obs_components(base,challenge,
                         [PROTOCOL.index(b)],scale),future_z)
        gain = np.minimum(da,db)-dp
        detail[pair]["gain_over_better_single"] = float(gain.mean())
        detail[pair]["gain95"] = bootstrap_mean_ci(gain)
        # Duplicating A or B yields the same observation twice but reweights
        # mechanical data relative to baseline. This is the fair redundancy
        # control for claiming complementarity from the additional protocol.
        dda=future_neighbor_distance(obs_components(base,challenge,
                         [PROTOCOL.index(a),PROTOCOL.index(a)],scale),future_z)
        ddb=future_neighbor_distance(obs_components(base,challenge,
                         [PROTOCOL.index(b),PROTOCOL.index(b)],scale),future_z)
        unique_gain=np.minimum(dda,ddb)-dp
        detail[pair]["gain_over_better_duplicated_single"] = float(unique_gain.mean())
        detail[pair]["unique_gain95"] = bootstrap_mean_ci(unique_gain)
    return out


def pair_equivalence_audit(base, challenge, future_z):
    b = obs_components(base, challenge, [])
    c = obs_components(base, challenge, [2])
    bd = pdist(b)
    fd = pdist(future_z)
    cd = pdist(c)
    chosen = bd <= np.quantile(bd,.10)
    # Functional divergence threshold is based on observed future distances
    # and is descriptive, not a preclinical decision threshold.
    future_divergent = fd[chosen] > np.median(fd[chosen])
    rho_b = spearmanr(bd[chosen], fd[chosen]).statistic
    rho_c = spearmanr(cd[chosen], fd[chosen]).statistic
    return {"baseline_near_pairs": int(chosen.sum()),
            "future_divergent_fraction": float(future_divergent.mean()),
            "future_distance_median_near": float(np.median(fd[chosen])),
            "future_distance_median_all": float(np.median(fd)),
            "spearman_baseline_distance_to_future_in_near_pairs": float(rho_b),
            "spearman_challenge_distance_to_future_in_near_pairs": float(rho_c)}


def cv_predictive_state():
    protocol_sets = {"baseline": [], "mid300": [2],
                     "early150": [0], "late600": [1],
                     "early+mid": [0,2], "mid+late": [2,1]}
    outer = KFold(n_splits=5, shuffle=True, random_state=2401)
    output = {}
    for label, prot in protocol_sets.items():
        xdim=obs_components(BASE,CHALLENGE,prot).shape[1]
        output[label] = {}
        for dim in [1,2,3,5,8,13]:
            if dim > min(xdim, len(BASE)-1):
                continue
            errors = []
            for train, test in outer.split(BASE):
                x=fold_observation(train,BASE,CHALLENGE,prot)
                z=fold_future(train,FUTURE)
                # PCA is fitted to training cells only. PLS is supervised and
                # therefore also fitted only to training cells.
                pca = PCA(n_components=dim).fit(x[train])
                ztr, zte = pca.transform(x[train]), pca.transform(x[test])
                mean, std = z[train].mean(axis=0), z[train].std(axis=0)
                ys = np.maximum(std, .05)
                ridge = Ridge(alpha=10).fit(ztr, (z[train]-mean)/ys)
                yp = ridge.predict(zte)*ys+mean
                errors.extend(np.sqrt(((yp-z[test])**2).sum(axis=1)))
            output[label][f"PCA{dim}"] = float(np.mean(errors))
        for dim in [1,2,3,5]:
            if dim > xdim: continue
            errors=[]
            for train,test in outer.split(BASE):
                x=fold_observation(train,BASE,CHALLENGE,prot)
                z=fold_future(train,FUTURE)
                model=PLSRegression(n_components=dim,scale=False)
                model.fit(x[train],z[train])
                yp=model.predict(x[test])
                errors.extend(np.sqrt(((yp-z[test])**2).sum(axis=1)))
            output[label][f"PLS{dim}"]=float(np.mean(errors))
    # An actual 13-parameter oracle must use the true forward equations to
    # attain zero in-model error. A learned map from true theta is separately
    # reported, because with 64 cells it also has finite-sample error.
    theta_errors=[]
    for train,test in outer.split(THETA):
        ztheta=(THETA-THETA[train].mean(axis=0))/PARAM_RANGE
        z=fold_future(train,FUTURE)
        model=Ridge(alpha=10).fit(ztheta[train],z[train])
        yp=model.predict(ztheta[test])
        theta_errors.extend(np.sqrt(((yp-z[test])**2).sum(axis=1)))
    output["true_theta_linear_map"]={"13d_ridge":float(np.mean(theta_errors))}
    def rbf_predict(xtrain,ytrain,xtest):
        mu=xtrain.mean(axis=0); sd=xtrain.std(axis=0)
        keep=sd>1e-10
        a=(xtrain[:,keep]-mu[keep])/sd[keep]
        b=(xtest[:,keep]-mu[keep])/sd[keep]
        sq=cdist(a,a,"sqeuclidean")
        nonzero=sq[sq>1e-12]
        bandwidth=float(np.median(nonzero)) if len(nonzero) else 1.
        kernel=np.exp(-sq/bandwidth)
        cross=np.exp(-cdist(b,a,"sqeuclidean")/bandwidth)
        ym=ytrain.mean(axis=0)
        return ym+cross@np.linalg.solve(kernel+.02*np.eye(len(a)),ytrain-ym)
    for label,prot in protocol_sets.items():
        errors=[]
        for train,test in outer.split(BASE):
            x=fold_observation(train,BASE,CHALLENGE,prot)
            z=fold_future(train,FUTURE)
            yp=rbf_predict(x[train],z[train],x[test])
            errors.extend(np.sqrt(((yp-z[test])**2).sum(axis=1)))
        output[label]["full_RBF"]=float(np.mean(errors))
    errors=[]
    for train,test in outer.split(THETA):
        ztheta=(THETA-THETA[train].mean(axis=0))/PARAM_RANGE
        z=fold_future(train,FUTURE)
        yp=rbf_predict(ztheta[train],z[train],ztheta[test])
        errors.extend(np.sqrt(((yp-z[test])**2).sum(axis=1)))
    output["true_theta_linear_map"]["13d_RBF"]=float(np.mean(errors))
    return output


def cross_form_predictive_state():
    """One frozen 52/12 source-to-alternate functional-state transfer."""
    rng=np.random.default_rng(777)
    order=rng.permutation(64)
    test,train=order[:12],order[12:]
    protocol_sets={"baseline":[],"mid300":[2],"mid+late":[2,1]}
    result={}
    for label,prot in protocol_sets.items():
        xsrc=fold_observation(train,BASE,CHALLENGE,prot)
        xalt=fold_observation(train,ALT_BASE,ALT_CHALLENGE,prot)
        zsrc=fold_future(train,FUTURE)
        zalt=fold_future(train,ALT_FUTURE)
        dim=min(5,xsrc.shape[1])
        model=PLSRegression(n_components=dim,scale=False).fit(xsrc[train],zsrc[train])
        psource=model.predict(xsrc[test])
        palt=model.predict(xalt[test])
        result[label]={"latent_dim":dim,
            "source_test_L2":float(np.sqrt(((psource-zsrc[test])**2).sum(axis=1)).mean()),
            "gain_exponent_test_L2":float(np.sqrt(((palt-zalt[test])**2).sum(axis=1)).mean())}
    forms_path=ROOT/"feedback_13param_two_forms.csv"
    if forms_path.exists():
        forms=np.loadtxt(forms_path,delimiter=",")
        assert forms.shape==(24,66)
        for name,block in [("saturating_SAC",forms[:12]),
                           ("reduced_TnC_buffer",forms[12:])]:
            ids=block[:,0].astype(int)-1
            assert np.array_equal(ids,test)
            f=block[:,17:-1].reshape(12,4,12)
            b=ALT_BASE.copy(); ch=ALT_CHALLENGE.copy()
            b[ids]=block[:,14:17]
            ch[ids,2]=f[:,0]
            # These two alternate truth sets only contain the 300-ms
            # calibration protocol, so compare baseline and mid only.
            truth=np.zeros_like(ALT_FUTURE)
            truth[ids]=f[:,1:,:][:,:,[0,2,3,6,8,10]]
            ztruth=fold_future(train,truth)
            for label,prot in [("baseline",[]),("mid300",[2])]:
                xsrc=fold_observation(train,BASE,CHALLENGE,prot)
                xform=fold_observation(train,b,ch,prot)
                zsrc=fold_future(train,FUTURE)
                dim=min(5,xsrc.shape[1])
                model=PLSRegression(n_components=dim,scale=False).fit(xsrc[train],zsrc[train])
                yp=model.predict(xform[test])
                result[label][name+"_test_L2"]=float(np.sqrt(
                    ((yp-ztruth[test])**2).sum(axis=1)).mean())
    return result


def repeated_split_sensitivity():
    settings={"baseline":([],3),"mid300":([2],5),
              "mid+late":([2,1],5)}
    per_seed={key:[] for key in settings}
    for seed in range(20):
        cv=KFold(n_splits=5,shuffle=True,random_state=3400+seed)
        for label,(prot,dim) in settings.items():
            errs=[]
            for train,test in cv.split(BASE):
                x=fold_observation(train,BASE,CHALLENGE,prot)
                z=fold_future(train,FUTURE)
                model=PLSRegression(n_components=dim,scale=False).fit(x[train],z[train])
                yp=model.predict(x[test])
                errs.extend(np.sqrt(((yp-z[test])**2).sum(axis=1)))
            per_seed[label].append(float(np.mean(errs)))
    out={key:{"mean":float(np.mean(v)),
              "range":[float(np.min(v)),float(np.max(v))]} for key,v in per_seed.items()}
    improvement=np.array(per_seed["baseline"])-np.array(per_seed["mid300"])
    complement=np.array(per_seed["mid300"])-np.array(per_seed["mid+late"])
    out["paired_gain"]={
        "baseline_to_mid_mean":float(improvement.mean()),
        "baseline_to_mid_range":[float(improvement.min()),float(improvement.max())],
        "mid_to_midlate_mean":float(complement.mean()),
        "mid_to_midlate_range":[float(complement.min()),float(complement.max())],
    }
    return out


source_audit=neighbor_audit(BASE,CHALLENGE,ZF)
alternate_audit=neighbor_audit(ALT_BASE,ALT_CHALLENGE,ALT_ZF)
k_sensitivity={}
for k in [3,5,10]:
    b=future_neighbor_distance(obs_components(BASE,CHALLENGE,[]),ZF,k)
    c=future_neighbor_distance(obs_components(BASE,CHALLENGE,[2]),ZF,k)
    k_sensitivity[str(k)]={"baseline":float(b.mean()),
                           "mid300":float(c.mean()),
                           "improvement":float((b-c).mean())}
pair_audit=pair_equivalence_audit(BASE,CHALLENGE,ZF)
predictive=cv_predictive_state()
cross_predictive=cross_form_predictive_state()
split_sensitivity=repeated_split_sensitivity()
future_pca=PCA().fit(ZF)
out={"n_cells":64,"n_neighbors":5,
     "observation_spaces":"Baseline APD/Ca peak/T peak; candidate challenge adds twelve AP/Ca/T features. Each block is standardized by the source cohort and weighted equally in Euclidean distance.",
     "future_space":"Three protocols x six outputs; train-cohort SD and physical floors, equal protocol weighting.",
     "source_neighbors":source_audit,
     "gain_exponent_mismatch_neighbors":alternate_audit,
     "k_neighbour_sensitivity":k_sensitivity,
     "baseline_equivalence_pairs":pair_audit,
     "predictive_latent_5fold":predictive,
     "source_trained_latent_cross_form":cross_predictive,
     "repeated_5fold_split_sensitivity":split_sensitivity,
     "future_PCA_cumulative_variance":np.cumsum(future_pca.explained_variance_ratio_).tolist(),
     "limitations":[
         "The same 64-cell synthetic cohort has been inspected in earlier analyses; this is an exploratory feasibility test, not a new independent confirmation.",
         "Cross-form check covers the gain-exponent discrepancy with all 64 cells; saturating SAC and TnC datasets have only 12 cells.",
         "Low-dimensional PCA of smooth feature outputs does not by itself prove a biological state manifold.",
         "Pairwise complementarity is not evidence of irreducible higher-order information synergy or a need for a hypergraph.",
         "No real human-cell mechanical-challenge dataset is used.",
     ]}
path=ROOT/"feedback_functional_information_geometry.json"
path.write_text(json.dumps(out,indent=2),encoding="utf-8")
print("source baseline/mid/mid+late:",source_audit["baseline"],
      source_audit["conditions"]["mid300_110"]["future_neighbor_distance"],
      source_audit["conditions"]["300+600"]["future_neighbor_distance"])
print("mismatch baseline/mid:",alternate_audit["baseline"],
      alternate_audit["conditions"]["mid300_110"]["future_neighbor_distance"])
print("mid shuffle p",source_audit["mid_shuffle_test"]["p_one_sided"])
print("pair",pair_audit)
print("predictive",predictive)
print(path)
