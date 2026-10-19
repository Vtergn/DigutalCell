"""Compare target-blind Land ensemble and point-twin predictive bands.

First 32 confirmation cells calibrate; last 32 evaluate. Each score covers
an entire waveform. Condition and candidate file may be selected by CLI.
"""
from pathlib import Path
import argparse
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt

base=Path(__file__).resolve().parent/'virtual_cardiomyocyte_5000'
parser=argparse.ArgumentParser()
parser.add_argument('--ensemble-file',default='land_eight_direction_ensemble_confirm64.mat')
parser.add_argument('--condition',default='low_noise')
parser.add_argument('--suffix',default='confirm64')
parser.add_argument('--manifest',default='')
parser.add_argument('--truth-115-file',default='land_confirm_115_truth.mat')
args=parser.parse_args()
with h5py.File(base/args.ensemble_file) as f:
    draws=f['prediction'][()].transpose(3,2,1,0)
    weights=f['weights'][()].T
    point=f['point'][()].transpose(2,1,0)
    cells=f['cell_index_zero_based'][()].ravel().astype(int)
with h5py.File(base/'merged_cohort.h5') as f:
    lam=f['lambda'][()]
    jj=int(np.argmin(abs(lam-1.125)))
    assert abs(lam[jj]-1.125)<1e-6
    truth_1125=f['Ta_kPa'][cells][:,jj,:601:2]
with h5py.File(base/args.truth_115_file) as f:
    truth_115=f['truth'][()].transpose(2,1,0)[:,0,::2]
    assert np.array_equal(cells,f['cell_index_zero_based'][()].ravel().astype(int))
    assert not np.any(f['failed'][()].ravel())
truth=np.stack([truth_1125,truth_115],axis=1)
assert draws.shape[:3]==truth.shape and len(cells)==64
assert np.allclose(weights.sum(axis=1),1)
if args.manifest:
    manifest=json.loads((base/args.manifest).read_text())
    assert np.array_equal(cells,manifest['cells'])
    cal=np.array(manifest['calibration_rows_zero_based'],dtype=int)
    test=np.array(manifest['evaluation_rows_zero_based'],dtype=int)
else:
    cal=np.arange(32);test=np.arange(32,64)
assert len(cal)==len(test)==32 and not set(cal)&set(test)
k=int(np.ceil((len(cal)+1)*.9))

def metrics(center,scale,score):
    q=float(np.partition(score,k-1)[k-1])
    test_scale=scale[test] if scale.ndim==2 else scale
    lower=np.maximum(center[test]-q*test_scale,0)
    upper=center[test]+q*test_scale
    y=truth[test,j]
    inside=(y>=lower)&(y<=upper)
    return {'q':q,'covered_cells':int(inside.all(axis=1).sum()),
            'simultaneous_coverage':float(inside.all(axis=1).mean()),
            'pointwise_coverage':float(inside.mean()),
            'mean_width_kPa':float(np.mean(upper-lower)),
            'interval_score_kPa':float(np.mean(upper-lower+20*np.maximum(lower-y,0)+20*np.maximum(y-upper,0)))}

def wquant(x,weight,p):
    order=np.argsort(x,axis=-1)
    xx=np.take_along_axis(x,order,axis=-1)
    ww=np.take_along_axis(np.broadcast_to(weight[:,None,:],x.shape),order,axis=-1)
    index=np.argmax(np.cumsum(ww,axis=-1)>=p,axis=-1)
    return np.take_along_axis(xx,index[...,None],axis=-1)[...,0]

def per_cell_band_scores(center,scale,q,y):
    sc=scale[test] if scale.ndim==2 else scale
    lo=np.maximum(center[test]-q*sc,0);hi=center[test]+q*sc
    width=np.mean(hi-lo,axis=1)
    interval=np.mean(hi-lo+20*np.maximum(lo-y,0)+20*np.maximum(y-hi,0),axis=1)
    return width,interval

rows=[]
for j,target in enumerate([1.125,1.15]):
    ycal=truth[cal,j]
    p=point[:,j]
    d=draws[:,j]
    mu=np.sum(d*weights[:,None,:],axis=-1)
    sd=np.sqrt(np.sum(weights[:,None,:]*(d-mu[:,:,None])**2,axis=-1))
    # Positive floor avoids numerical amplification around the resting tail.
    es=np.maximum(sd,.5)
    ps=np.maximum(np.sqrt(np.mean((ycal-p[cal])**2,axis=0)),.5)
    methods={}
    for name,center,scale in [
        ('point_uniform',p,np.ones(p.shape[1])),
        ('point_time_scaled',p,ps),
        ('mechanistic_scale_point_center',p,es),
        ('mechanistic_ensemble_scaled',mu,es)]:
        cal_scale=scale[cal] if scale.ndim==2 else scale
        residual=abs(ycal-center[cal])/cal_scale
        cell_score=np.max(residual,axis=1)
        methods[name]=metrics(center,scale,cell_score)
        methods[name]['center_MAE_kPa']=float(np.mean(abs(center[test]-truth[test,j])))
    dt=d[test]
    wt=weights[test]
    yy=truth[test,j]
    crps_first=np.sum(wt[:,None,:]*abs(dt-yy[:,:,None]),axis=-1)
    crps_pair=np.abs(dt[:,:,:,None]-dt[:,:,None,:])
    crps_second=.5*np.sum(wt[:,None,:,None]*wt[:,None,None,:]*crps_pair,axis=(-2,-1))
    raw_low=wquant(dt,wt,.05)
    raw_high=wquant(dt,wt,.95)
    raw_inside=(yy>=raw_low)&(yy<=raw_high)
    ref_w,ref_s=per_cell_band_scores(p,ps,methods['point_time_scaled']['q'],yy)
    new_w,new_s=per_cell_band_scores(mu,es,methods['mechanistic_ensemble_scaled']['q'],yy)
    rng=np.random.default_rng(20260929+j)
    boot=rng.integers(0,len(test),size=(3000,len(test)))
    gain_width=ref_w-new_w
    gain_score=ref_s-new_s
    baseline=methods['point_time_scaled']
    candidate=methods['mechanistic_ensemble_scaled']
    gate=(candidate['covered_cells']>=29 and
          candidate['mean_width_kPa']<baseline['mean_width_kPa'] and
          candidate['interval_score_kPa']<baseline['interval_score_kPa'])
    rows.append({'target_lambda':target,'methods':methods,
                 'pre_registered_gate_pass':bool(gate),
                 'mechanistic_vs_time_scaled_point_paired_width_gain_kPa':float(gain_width.mean()),
                 'paired_width_gain_bootstrap_95pct_CI_kPa':np.quantile(gain_width[boot].mean(axis=1),[.025,.975]).tolist(),
                 'mechanistic_vs_time_scaled_point_paired_interval_score_gain_kPa':float(gain_score.mean()),
                 'paired_interval_score_gain_bootstrap_95pct_CI_kPa':np.quantile(gain_score[boot].mean(axis=1),[.025,.975]).tolist(),
                 'raw_ensemble_CRPS_kPa':float(np.mean(crps_first-crps_second)),
                 'raw_ensemble_pointwise_coverage':float(raw_inside.mean()),
                 'raw_ensemble_simultaneous_coverage':float(raw_inside.all(axis=1).mean()),
                 'raw_ensemble_mean_width_kPa':float(np.mean(raw_high-raw_low)),
                 'ensemble_scale_median_kPa':float(np.median(es[test]))})

report={'calibration_cells':32,'evaluation_cells':32,'n_draws':draws.shape[-1],
        'observation_condition':args.condition,'ensemble_file':args.ensemble_file,
        'split_manifest':args.manifest or 'ordered first 32 / last 32',
        'calibration_rank':k,'score':'Per-cell maximum standardized absolute error over 301 time points',
        'results':rows,'all_targets_pre_registered_gate_pass':bool(all(r['pre_registered_gate_pass'] for r in rows)),
        'scope':'Fixed target-blind ensemble and point twin, split within 64 independent confirmation cells with matching synthetic observation condition. Coverage is across cells and simultaneous over the waveform. Interval bands are conformal predictive sets, not Bayesian credible regions. Small 32-cell test has substantial binomial uncertainty.'}
(base/f'land_conformal_ensemble_{args.suffix}_score.json').write_text(json.dumps(report,indent=2))
fig,axes=plt.subplots(1,2,figsize=(10,3.9),constrained_layout=True)
names=['point_uniform','point_time_scaled','mechanistic_scale_point_center','mechanistic_ensemble_scaled']
for j,ax in enumerate(axes):
    result=rows[j]
    vals=[result['methods'][name]['mean_width_kPa'] for name in names]
    cover=[result['methods'][name]['covered_cells'] for name in names]
    ax.bar(np.arange(4),vals,color=['#5076a6','#6a91ba','#398064','#76a98e'])
    for m,(v,c) in enumerate(zip(vals,cover)):
        ax.text(m,v+.15,f'{c}/32',ha='center',fontsize=8)
    ax.set_xticks(np.arange(4),['Point uniform','Point time scaled','Land scale + point','Land scale + mean'],rotation=16)
    ax.set(title=f'λ={result["target_lambda"]}',ylabel='Mean 90% band width (kPa)')
    ax.grid(axis='y',alpha=.2)
fig.savefig(base/f'land_conformal_ensemble_{args.suffix}.png',dpi=180)
print(json.dumps(report,indent=2))
