"""Seal a fresh 64-cell low-noise UQ cohort before target scoring.

Calibration/evaluation allocation is balanced within quartiles of observed
T(1.1) peak. This script never reads hidden target tension or true theta.
"""
from pathlib import Path
import json
import hashlib
import h5py
import joblib
import numpy as np
from scipy.io import savemat
from scipy.signal import savgol_filter

base=Path(__file__).resolve().parent/'virtual_cardiomyocyte_5000'
original=set(json.loads((base/'land_blind_128_selection.json').read_text())['selected_test_cells'])
stress=set(json.loads((base/'land_observation_stress_manifest.json').read_text())['cells'])
confirm=set(json.loads((base/'land_stress_confirmation_manifest.json').read_text())['selected_test_cells'])
earlier=set(json.loads((base/'posterior_draws_16x8.json').read_text())['test_cell_indices_zero_based'])
excluded=original|stress|confirm|earlier
cfg=json.loads((base/'land_observation_stress_manifest.json').read_text())['conditions']['low_noise']
with h5py.File(base/'merged_cohort.h5') as f:
    split=f['split_code'][()]
    eligible=np.array(sorted(set(np.flatnonzero(split==2))-excluded))
    cells=np.sort(np.random.default_rng(20261004).choice(eligible,size=64,replace=False))
    lam=f['lambda'][()]
    ix=[int(np.argmin(abs(lam-v))) for v in [1.,1.1]]
    assert np.allclose(lam[ix],[1.,1.1])
    v=f['V'][cells]
    ca=f['Cai_uM'][cells]
    t=f['Ta_kPa'][cells][:,ix,:]
    lo=f['lo'][()];hi=f['hi'][()]

def observe(clean,sigma,step,rng):
    sample=np.arange(0,1001,step)
    if sample[-1]!=1000:sample=np.r_[sample,1000]
    coarse=clean[...,sample]+rng.normal(0,sigma,(*clean.shape[:-1],len(sample)))
    return np.stack([np.interp(np.arange(1001),sample,row)
                     for row in coarse.reshape(-1,len(sample))]).reshape(clean.shape)

rng=np.random.default_rng(20261005)
step=cfg['step_ms']
vo=observe(v,cfg['ap_sigma_mV'],step,rng)
co=observe(ca,cfg['ca_sigma_uM'],step,rng)
to=observe(t,cfg['t_sigma_kPa'],step,rng)
vo=savgol_filter(vo,5,3,axis=-1)
co=np.maximum(savgol_filter(co,31,3,axis=-1),.001)
to=savgol_filter(to,21,3,axis=-1)
feature=np.concatenate([vo[:,:601:5],co[:,:601:5],to[:,0,:601:5],to[:,1,:601:5]],axis=1)
ridge=joblib.load(base/'land_stress_robust_ridge.joblib')
start=lo+np.clip(ridge.predict(feature),0,1)*(hi-lo)

observed_peak=to[:,1,:601].max(axis=1)
rank_order=np.argsort(observed_peak)
groups=np.empty(64,dtype=int)
groups[rank_order]=np.repeat(np.arange(4),16)
split_rng=np.random.default_rng(20261006)
cal=[];test=[]
for q in range(4):
    members=split_rng.permutation(np.flatnonzero(groups==q))
    cal.extend(members[:8]);test.extend(members[8:])
cal=np.sort(cal);test=np.sort(test)
assert len(cal)==len(test)==32 and not set(cal)&set(test)
path=base/'land_uq_fresh64_low_noise_input.mat'
savemat(path,{'cell_index_zero_based':cells.astype(float),
              'split_code':split[cells].astype(float),
              'V_mV':vo,'Cai_uM':co,'Tcal_kPa':to,
              'theta_start':start,'lo':lo,'hi':hi},do_compression=True)
manifest={'cells':cells.tolist(),'calibration_rows_zero_based':cal.tolist(),
          'evaluation_rows_zero_based':test.tolist(),
          'selection_seed':20261004,'noise_seed':20261005,'allocation_seed':20261006,
          'observed_peak_quartile_counts_cal':[int(np.sum(groups[cal]==q)) for q in range(4)],
          'observed_peak_quartile_counts_test':[int(np.sum(groups[test]==q)) for q in range(4)],
          'observed_T11_peak_mean_cal_kPa':float(observed_peak[cal].mean()),
          'observed_T11_peak_mean_test_kPa':float(observed_peak[test].mean()),
          'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
          'excluded_prior_UQ_or_blind_cells':len(excluded),
          'target_truth_and_true_theta_absent_from_input':True,
          'scope':'Pre-target allocation using only observed T(1.1) peak; all rules fixed before target scoring.'}
(base/'land_uq_fresh64_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({k:manifest[k] for k in ['observed_peak_quartile_counts_cal',
    'observed_peak_quartile_counts_test','observed_T11_peak_mean_cal_kPa',
    'observed_T11_peak_mean_test_kPa','input_sha256']},indent=2))
