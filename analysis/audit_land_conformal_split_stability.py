"""Exploratory sensitivity of cell-level conformal bands to the 32/32 split.

No model fitting or method selection occurs here. Repeated random splits of
the same 64 cells only diagnose how much the small calibration set matters.
The original ordered split remains the primary held-out result.
"""
from pathlib import Path
import json
import h5py
import numpy as np

base=Path(__file__).resolve().parent/'virtual_cardiomyocyte_5000'
files={12:'land_eight_direction_ensemble_confirm64.mat',
       24:'land_eight_direction_ensemble_confirm64_24.mat'}
with h5py.File(base/'merged_cohort.h5') as f:
    cells=np.array(json.loads((base/'land_stress_confirmation_manifest.json').read_text())['selected_test_cells'])
    lam=f['lambda'][()]
    jj=int(np.argmin(abs(lam-1.125)))
    truth_1125=f['Ta_kPa'][cells][:,jj,:601:2]
with h5py.File(base/'land_confirm_115_truth.mat') as f:
    truth_115=f['truth'][()].transpose(2,1,0)[:,0,::2]
truth=np.stack([truth_1125,truth_115],axis=1)

pred={}
for count,file in files.items():
    with h5py.File(base/file) as f:
        d=f['prediction'][()].transpose(3,2,1,0)
        w=f['weights'][()].T
        point=f['point'][()].transpose(2,1,0)
    mean=np.sum(d*w[:,None,None,:],axis=-1)
    sd=np.maximum(np.sqrt(np.sum(w[:,None,None,:]*(d-mean[:,:,:,None])**2,axis=-1)),.5)
    pred[count]=(mean,sd,point)

def evaluate(cal,test,j,count,method):
    mean,scale,point=pred[count]
    if method=='land':
        center=mean[:,j];sc=scale[:,j]
    elif method=='point':
        center=point[:,j]
        rmse=np.maximum(np.sqrt(np.mean((truth[cal,j]-center[cal])**2,axis=0)),.5)
        sc=np.broadcast_to(rmse,center.shape)
    else:raise ValueError(method)
    scores=np.max(abs(truth[cal,j]-center[cal])/sc[cal],axis=1)
    q=float(np.partition(scores,29)[29])
    lo=np.maximum(center[test]-q*sc[test],0)
    hi=center[test]+q*sc[test]
    y=truth[test,j]
    coverage=float(np.mean(np.all((y>=lo)&(y<=hi),axis=1)))
    width=float(np.mean(hi-lo))
    interval=float(np.mean(hi-lo+20*np.maximum(lo-y,0)+20*np.maximum(y-hi,0)))
    return coverage,width,interval

rng=np.random.default_rng(20261003)
permutations=[np.arange(64),np.r_[np.arange(32,64),np.arange(32)]]
permutations.extend(rng.permutation(64) for _ in range(256))
results={}
for j,target in enumerate([1.125,1.15]):
    rows=[]
    for perm in permutations:
        cal=perm[:32];test=perm[32:]
        rows.append({f'{method}_{count}':evaluate(cal,test,j,count,method)
                     for count in [12,24] for method in ['point','land']})
    summary={}
    for key in rows[0]:
        arr=np.array([r[key] for r in rows[2:]])
        summary[key]={'original_ordered_split':rows[0][key],
                      'reversed_ordered_split':rows[1][key],
                      'random_split_median':np.median(arr,axis=0).tolist(),
                      'random_split_5_to_95pct':np.quantile(arr,[.05,.95],axis=0).tolist()}
    diff=np.array([np.array(r['land_12'])-np.array(r['land_24']) for r in rows[2:]])
    summary['land_12_minus_land_24']={'random_split_median_coverage_width_score':np.median(diff,axis=0).tolist(),
                                      'fraction_24_has_worse_coverage_by_at_least_0p1':float(np.mean(diff[:,0]>=.1)),
                                      'fraction_24_has_lower_interval_score':float(np.mean(diff[:,2]>0))}
    results[str(target)]=summary
report={'n_cells':64,'n_random_splits':256,'split_size':[32,32],
        'original_split_still_primary':True,'results':results,
        'scope':'Exploratory reuse of the same 64 cells to diagnose split variability. The random-split median is not an independent confirmation, and coverage near 90% on average is expected from the conformal rank rule.'}
(base/'land_conformal_split_stability_score.json').write_text(json.dumps(report,indent=2))
print(json.dumps({target:{k:v for k,v in group.items() if k in ['land_12','land_24','land_12_minus_land_24']} for target,group in results.items()},indent=2))
