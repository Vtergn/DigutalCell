"""Cell-held-out full AP/Ca/tension trajectory geometry, same-model pilot.

Source 64 cells set every scale and estimator. Independent-seed 64 cells are
queried with pre-challenge baseline or baseline + 300-ms challenge. Future
protocols 5-7 are used only to train source-cell predictions and score the
confirmation cells. Outputs are trajectory blocks, not summary endpoints.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parent / "human_bidirectional_output"
HERE = Path(__file__).resolve().parent
FLOOR = np.array([1., .01, .5])  # mV, microM, nominal tension units
ONSET = np.array([150, 600, 300, 150, 600, 225, 225, 750])


def load(stem):
    with h5py.File(ROOT / (stem + ".mat")) as f:
        wave = f["waveforms"][:].astype(np.float64)
        onsets = f["onsets"][:].ravel().astype(int)
    params = np.loadtxt(ROOT / (stem + ".csv"), delimiter=",")[:, :13]
    assert wave.shape == (64, 8, 3, 1001) and np.array_equal(onsets, ONSET)
    assert np.isfinite(wave).all()
    return wave, params


wa, ta = load("feedback_13param_cohort")
wb, tb = load("feedback_13param_confirm_cohort")
STEP = 5
BASE_TIME = np.arange(0, 300, STEP)
CAL_TIME = np.arange(300, 601, STEP)
FUTURE_TIME = [np.arange(int(o)-40, min(int(o)+251, 1001), STEP)
               for o in ONSET[5:8]]
assert all(len(t) >= 50 for t in FUTURE_TIME)

# All modality scales and means are frozen on source calibration data. The
# signal floor prevents near-constant intervals from being exaggerated.
cal_std = wa[:, 2][:, :, CAL_TIME].std(axis=0)
SCALE = np.maximum(np.quantile(cal_std, .75, axis=1), FLOOR)


def block(w, protocol, times, reference, pure_shape=False):
    # Two-step indexing keeps the cell axis first. NumPy's mixed advanced
    # index w[:, protocol, :, times] silently moves the time axis to front.
    x = w[:, protocol][:, :, times].copy()
    assert x.shape == (64, 3, len(times))
    if pure_shape:
        # Time-standardized waveforms isolate timing/morphology from peak size.
        x -= x.mean(axis=2, keepdims=True)
        denom = np.maximum(x.std(axis=2, keepdims=True), FLOOR[None,:,None])
        x /= denom
        return x / np.sqrt(x.shape[1]*x.shape[2])
    r = reference[:, protocol][:, :, times]
    x = (x - r.mean(axis=0, keepdims=True)) / SCALE[None,:,None]
    return x / np.sqrt(x.shape[1]*x.shape[2])


def obs(w, include_challenge):
    b = block(w, 2, BASE_TIME, wa).reshape(64, -1)
    if not include_challenge:
        return b
    c = block(w, 2, CAL_TIME, wa).reshape(64, -1)
    return np.concatenate([b, c], axis=1)


def future(w, pure_shape=False, channel=None):
    chunks = []
    for j,t in zip(range(5,8), FUTURE_TIME):
        z = block(w, j, t, wa, pure_shape=pure_shape)
        if channel is not None:
            z = z[:,channel:channel+1]
        chunks.append(z.reshape(64,-1) / np.sqrt(3))
    return np.concatenate(chunks, axis=1)


xa0, xb0 = obs(wa, False), obs(wb, False)
xa1, xb1 = obs(wa, True), obs(wb, True)
ya, yb = future(wa), future(wb)


def neighbors_same(x, y, k=5):
    d = cdist(x, x)
    np.fill_diagonal(d, np.inf)
    near = np.argsort(d, axis=1)[:,:k]
    delta = y[:,None,:] - y[near]
    return np.sqrt(np.square(delta).sum(axis=2)).mean(axis=1)


def source_knn(xa, xb, y, k=5):
    near = np.argsort(cdist(xb, xa), axis=1)[:,:k]
    return y[near].mean(axis=1)


def row_error(pred, truth):
    return np.linalg.norm(pred-truth, axis=1)


def paired_ci(values, seed=777):
    rng = np.random.default_rng(seed)
    samples = rng.integers(len(values), size=(5000,len(values)))
    means = values[samples].mean(axis=1)
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def eval_space(y_train, y_test):
    near_base = neighbors_same(xb0, y_test)
    near_challenge = neighbors_same(xb1, y_test)
    k0 = row_error(source_knn(xa0, xb0, y_train), y_test)
    k1 = row_error(source_knn(xa1, xb1, y_train), y_test)
    # Frozen regularization; fit source cells only, evaluate new cells once.
    r0 = row_error(Ridge(alpha=10).fit(xa0,y_train).predict(xb0), y_test)
    r1 = row_error(Ridge(alpha=10).fit(xa1,y_train).predict(xb1), y_test)
    return {"within_confirmation_neighbor": {"baseline": float(near_base.mean()),
              "challenge": float(near_challenge.mean()),
              "improvement95": paired_ci(near_base-near_challenge)},
            "source_to_confirmation_knn5": {"baseline": float(k0.mean()),
              "challenge": float(k1.mean()), "improvement95": paired_ci(k0-k1)},
            "source_to_confirmation_ridge10": {"baseline": float(r0.mean()),
              "challenge": float(r1.mean()), "improvement95": paired_ci(r0-r1)}}


primary = eval_space(ya,yb)
shape = eval_space(future(wa,True),future(wb,True))
channels = {name: eval_space(future(wa,channel=j),future(wb,channel=j))
            for j,name in enumerate(["voltage","calcium","tension"])}

rng = np.random.default_rng(45677)
base_near = neighbors_same(xb0,yb)
actual = float((base_near - neighbors_same(xb1,yb)).mean())
null = []
for _ in range(1000):
    shuffled = np.concatenate([xb1[:,:xb0.shape[1]],
                               xb1[rng.permutation(64),xb0.shape[1]:]], axis=1)
    null.append(float((base_near-neighbors_same(shuffled,yb)).mean()))

state_pca = PCA(n_components=5).fit(xa1)
z_a,z_b = state_pca.transform(xa1),state_pca.transform(xb1)
rho_a = [[float(spearmanr(z_a[:,i],ta[:,j]).statistic)
          for j in range(13)] for i in range(5)]
rho_b = [[float(spearmanr(z_b[:,i],tb[:,j]).statistic)
          for j in range(13)] for i in range(5)]
best = np.unravel_index(np.argmax(np.abs(rho_a)),(5,13))


def parameter_gradients(xsource, xconfirm):
    pcs=PCA(n_components=5).fit(xsource)
    a,b=pcs.transform(xsource),pcs.transform(xconfirm)
    ra=np.array([[spearmanr(a[:,i],ta[:,j]).statistic for j in range(13)]
                 for i in range(5)])
    rb=np.array([[spearmanr(b[:,i],tb[:,j]).statistic for j in range(13)]
                 for i in range(5)])
    pairs=[]
    for i,j in np.ndindex(ra.shape):
        if abs(ra[i,j]) >= .45:
            pairs.append({"pc":i+1,"parameter_index":j,
                          "source_rho":float(ra[i,j]),"confirm_rho":float(rb[i,j])})
    return pcs,a,b,sorted(pairs,key=lambda v:-abs(v["source_rho"]))


# Challenge innovation: remove source-predictable continuation of baseline.
# This preserves cell-held-out evaluation while exposing mechanical response
# directions that are small relative to the normal AP/Ca cycle.
c_a,c_b=xa1[:,xa0.shape[1]:],xb1[:,xb0.shape[1]:]
continuation=Ridge(alpha=10).fit(xa0,c_a)
innov_a,innov_b=c_a-continuation.predict(xa0),c_b-continuation.predict(xb0)
innov_pca,innov_za,innov_zb,innov_pairs=parameter_gradients(innov_a,innov_b)


def mixture_audit(a,b):
    return [{"components":k,
             "source_BIC":float((g:=GaussianMixture(n_components=k,reg_covar=.01,
                   n_init=10,random_state=42).fit(a)).bic(a)),
             "confirmation_log_density_per_cell":float(g.score(b))}
            for k in range(1,5)]


mixture_raw=mixture_audit(z_a,z_b)
mixture_innovation=mixture_audit(innov_za,innov_zb)

# Paired model-form interventions: preserve the source axes and compare each
# altered calibration trace with the exact same source-model cell.
forms=np.loadtxt(ROOT/"feedback_13param_two_forms.csv",delimiter=",")
with h5py.File(ROOT/"feedback_13param_two_forms_waveforms.mat") as f:
    altwave=f["calibration_waveforms"][:].transpose(2,1,0).astype(float)
assert forms.shape==(24,66) and altwave.shape==(24,3,1001)
ids=forms[:,0].astype(int)-1
assert np.array_equal(ids[:12],ids[12:])
source_cal=wa[:,2]
base_alt=(altwave[:,:,BASE_TIME]-source_cal[:,:,BASE_TIME].mean(axis=0))/SCALE[None,:,None]
base_alt=(base_alt/np.sqrt(3*len(BASE_TIME))).reshape(24,-1)
cal_alt=(altwave[:,:,CAL_TIME]-source_cal[:,:,CAL_TIME].mean(axis=0))/SCALE[None,:,None]
cal_alt=(cal_alt/np.sqrt(3*len(CAL_TIME))).reshape(24,-1)
z_alt=innov_pca.transform(cal_alt-continuation.predict(base_alt))
delta=z_alt-innov_za[ids]
form_shifts={}
for label,selection in [("saturating_SAC",slice(0,12)),
                         ("reduced_TnC_buffer",slice(12,24))]:
    shift=delta[selection]
    form_shifts[label]={"n_paired_cells":12,
      "median_delta_per_pc":np.median(shift,axis=0).tolist(),
      "median_abs_delta_per_pc":np.median(abs(shift),axis=0).tolist(),
      "median_vector_norm":float(np.median(np.linalg.norm(shift,axis=1)))}


def source_cv_ridge(x_train, x_test):
    folds = KFold(n_splits=5,shuffle=True,random_state=413)
    losses = {}
    for alpha in [.1,1.,10.,100.]:
        per = []
        for train,val in folds.split(x_train):
            p = Ridge(alpha=alpha).fit(x_train[train],ya[train]).predict(x_train[val])
            per.extend(row_error(p,ya[val]))
        losses[str(alpha)] = float(np.mean(per))
    selected = float(min(losses,key=losses.get))
    pred = Ridge(alpha=selected).fit(x_train,ya).predict(x_test)
    return {"source_cv_by_alpha":losses,"selected_alpha":selected,
            "confirmation_mean":float(row_error(pred,yb).mean())},pred


tuned = {}
predictions = {}
for label,atr,btr in [("baseline",xa0,xb0),
                       ("baseline_duplicated",np.c_[xa0,xa0],np.c_[xb0,xb0]),
                       ("baseline_plus_challenge",xa1,xb1)]:
    tuned[label],predictions[label] = source_cv_ridge(atr,btr)
tuned["population_mean"] = float(row_error(
    np.broadcast_to(ya.mean(0),yb.shape),yb).mean())
tuned["challenge_vs_baseline_95"] = paired_ci(
    row_error(predictions["baseline"],yb) -
    row_error(predictions["baseline_plus_challenge"],yb))
tuned["challenge_vs_duplicated_95"] = paired_ci(
    row_error(predictions["baseline_duplicated"],yb) -
    row_error(predictions["baseline_plus_challenge"],yb))


def physical_mae(z):
    out=[]; offset=0
    for j,t in zip(range(5,8),FUTURE_TIME):
        n = 3*len(t)
        part = z[:,offset:offset+n].reshape(64,3,len(t))
        ref = wa[:,j][:,:,t].mean(axis=0)
        raw = part*np.sqrt(3*n)*SCALE[None,:,None]+ref[None,:,:]
        truth = wb[:,j][:,:,t]
        out.append(np.abs(raw-truth).mean(axis=(0,2)))
        offset += n
    assert offset == z.shape[1]
    return dict(zip(["V_mV","Ca_uM","T_model_units"],np.mean(out,axis=0).tolist()))


tuned["baseline_physical_MAE"] = physical_mae(predictions["baseline"])
tuned["challenge_physical_MAE"] = physical_mae(predictions["baseline_plus_challenge"])
cell_gain = row_error(predictions["baseline"],yb)-row_error(
    predictions["baseline_plus_challenge"],yb)
tuned["confirmation_cells_improved"] = int(np.sum(cell_gain>0))
tuned["cell_gain_quantiles_10_50_90"] = np.quantile(cell_gain,[.1,.5,.9]).tolist()

# A physically absent input at lambda=1 should not acquire a reliable
# baseline gradient merely because the 13 parameters were jointly sampled.
baseline_pca=PCA(n_components=5).fit(xa0)
baseline_a,baseline_b=baseline_pca.transform(xa0),baseline_pca.transform(xb0)
baseline_param_control={
    "g_ionic_source":[float(spearmanr(baseline_a[:,i],ta[:,6]).statistic) for i in range(5)],
    "g_ionic_confirmation":[float(spearmanr(baseline_b[:,i],tb[:,6]).statistic) for i in range(5)],
    "beta1_source":[float(spearmanr(baseline_a[:,i],ta[:,11]).statistic) for i in range(5)],
    "beta1_confirmation":[float(spearmanr(baseline_b[:,i],tb[:,11]).statistic) for i in range(5)]}

out = {"source_cells":64,"confirmation_cells":64,
       "calibration_protocol":"lambda=1.10, onset 300 ms, hold 10 ms",
       "future_protocols":"indices 5-7; full V/Ca/tension windows 40 ms before to 250 ms after onset, 5 ms spacing",
       "distance":"Each time series normalized by source calibration modality scale; each modality and target protocol equal weight. Norm is dimensionless, not physical MAE.",
       "scale_V_Ca_T":SCALE.tolist(), "primary_full_waveform":primary,
       "source_cv_ridge_and_redundancy_control":tuned,
       "shape_normalized_waveform":shape,"per_modality":channels,
       "challenge_cell_permutation":{"observed_neighbor_gain":actual,
          "null_mean":float(np.mean(null)),
          "p_one_sided":float((1+np.sum(np.array(null)>=actual))/1001)},
       "observational_pca_source_variance":state_pca.explained_variance_ratio_.tolist(),
       "trajectory_density_components":{"raw":mixture_raw,
                                        "challenge_innovation":mixture_innovation},
       "paired_model_form_perturbation":form_shifts,
       "challenge_innovation":{"definition":"Challenge waveform minus source-fitted baseline-to-continuation ridge prediction; source alpha fixed at 10.",
             "pca_source_variance":innov_pca.explained_variance_ratio_.tolist(),
             "source_gradient_candidates":innov_pairs,
             "beta1_rho_per_pc_source":[float(spearmanr(innov_za[:,i],ta[:,11]).statistic) for i in range(5)],
             "beta1_rho_per_pc_confirm":[float(spearmanr(innov_zb[:,i],tb[:,11]).statistic) for i in range(5)]},
       "baseline_parameter_negative_control":baseline_param_control,
       "strongest_source_parameter_gradient":{"pc":int(best[0]+1),
           "parameter_index_zero_based":int(best[1]),
           "source_spearman":rho_a[best[0]][best[1]],
           "confirmation_spearman":rho_b[best[0]][best[1]]},
       "caveat":"Source and confirmation cells share the same equations and protocols. Target protocol waveforms are available for source-cell supervised regression; this is cell-held-out, not population-level leave-protocol-out."}
(HERE/"full_trajectory_space_gate.json").write_text(
    json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")

fig,ax=plt.subplots(1,3,figsize=(12,3.8),constrained_layout=True)
for a,kind in zip(ax,["within_confirmation_neighbor","source_to_confirmation_knn5",
                      "source_to_confirmation_ridge10"]):
    v=primary[kind]
    a.plot([0,1],[v["baseline"],v["challenge"]],"o-",lw=2)
    a.set(xticks=[0,1],xticklabels=["Baseline","+Challenge"],
          title=kind.replace("_"," "),ylabel="Future full-trajectory distance")
fig.savefig(HERE/"full_trajectory_space_gate.png",dpi=180)

fig,axs=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
for row,(j,label,lo,hi) in enumerate([(6,"SAC gain g_ionic",.4,1.6),
                                      (11,"Land beta1",-4.1,-.9)]):
    for col,(z,theta,cohort) in enumerate([(innov_za,ta,"Source"),
                                           (innov_zb,tb,"New seed")]):
        sc=axs[row,col].scatter(z[:,0],z[:,2],c=theta[:,j],cmap="viridis",
                                vmin=lo,vmax=hi,s=32,alpha=.85)
        axs[row,col].set(xlabel="Innovation PC1",ylabel="Innovation PC3",
                             title=f"{cohort}: {label}")
        fig.colorbar(sc,ax=axs[row,col],shrink=.78)
fig.savefig(HERE/"full_trajectory_innovation_space.png",dpi=180)

fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for ax,(label,selection,color) in zip(axs,[("Saturating SAC",slice(0,12),"#d95f02"),
                                        ("Reduced TnC buffer",slice(12,24),"#1b9e77")]):
    before=innov_za[ids[selection]][:,[0,2]]
    after=z_alt[selection][:,[0,2]]
    ax.scatter(before[:,0],before[:,1],color="black",s=18,label="Original")
    ax.scatter(after[:,0],after[:,1],color=color,s=22,label="Perturbed")
    for p,q in zip(before,after):
        ax.annotate("",xy=q,xytext=p,arrowprops={"arrowstyle":"->","color":color,
                                                "alpha":.6,"lw":1})
    ax.set(xlabel="Innovation PC1",ylabel="Innovation PC3",title=label)
    ax.legend(fontsize=8)
fig.savefig(HERE/"full_trajectory_perturbation_vectors.png",dpi=180)


def unscale_one(pred,cell,protocol_index):
    offset=0
    for j,t in zip(range(5,8),FUTURE_TIME):
        n=3*len(t)
        if j==protocol_index:
            ref=wa[:,j][:,:,t].mean(axis=0)
            raw=pred[cell,offset:offset+n].reshape(3,len(t))
            return t,raw*np.sqrt(3*n)*SCALE[:,None]+ref
        offset+=n
    raise ValueError(protocol_index)


gains=row_error(predictions["baseline"],yb)-row_error(
    predictions["baseline_plus_challenge"],yb)
example=int(np.argsort(gains)[len(gains)//2])
fig,axs=plt.subplots(3,1,figsize=(9,7),sharex=True,constrained_layout=True)
t,pred_b=unscale_one(predictions["baseline"],example,6)
_,pred_c=unscale_one(predictions["baseline_plus_challenge"],example,6)
truth=wb[example,6][:,t]
for ch,(axis,name,unit) in enumerate(zip(axs,["V","Ca","T"],["mV","µM","model units"])):
    axis.plot(t,truth[ch],color="black",lw=1.6,label="True simulator")
    axis.plot(t,pred_b[ch],color="#d95f02",lw=1.2,label="Baseline")
    axis.plot(t,pred_c[ch],color="#1b9e77",lw=1.2,label="+ mechanical probe")
    axis.axvspan(225,247,color="gray",alpha=.12)
    axis.set_ylabel(f"{name} ({unit})")
axs[0].legend(loc="best",fontsize=8)
axs[-1].set_xlabel("Time after pacing (ms)")
fig.suptitle(f"Unseen cell {example+1}, target λ=1.125 pulse; median gain example")
fig.savefig(HERE/"full_trajectory_prediction_example.png",dpi=180)
print(json.dumps({"primary":primary,"shape":shape,"per_modality":channels,
                  "permutation":out["challenge_cell_permutation"],
                  "gradient":out["strongest_source_parameter_gradient"]},indent=2))
