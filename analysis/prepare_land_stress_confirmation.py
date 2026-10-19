"""Seal independent 64-cell confirmation for the fixed noise-robust pipeline."""
from pathlib import Path
import json
import hashlib
import h5py
import joblib
import numpy as np
from scipy.io import savemat
from scipy.signal import savgol_filter

root=Path(__file__).resolve().parents[1]
base=root/'analysis/virtual_cardiomyocyte_5000'
first=json.loads((base/'land_blind_128_selection.json').read_text())['selected_test_cells']
previous=json.loads((base/'posterior_draws_16x8.json').read_text())['test_cell_indices_zero_based']
stress=json.loads((base/'land_observation_stress_manifest.json').read_text())
excluded=set(first)|set(previous)
with h5py.File(base/'merged_cohort.h5') as f:
    split=f['split_code'][()]
    eligible=np.array(sorted(set(np.flatnonzero(split==2))-excluded))
    cells=np.sort(np.random.default_rng(20260930).choice(eligible,size=64,replace=False))
    lam=f['lambda'][()]
    ix=[int(np.argmin(abs(lam-x))) for x in [1.,1.1]]
    assert np.allclose(lam[ix],[1.,1.1])
    v=f['V'][cells]
    ca=f['Cai_uM'][cells]
    t=f['Ta_kPa'][cells][:,ix,:]
    lo,hi=f['lo'][()],f['hi'][()]
assert len(excluded.intersection(cells))==0
ridge=joblib.load(base/'land_stress_robust_ridge.joblib')

def observe(clean,sigma,step,rng):
    sample=np.arange(0,1001,step)
    if sample[-1]!=1000:sample=np.r_[sample,1000]
    coarse=clean[...,sample]+rng.normal(0,sigma,(*clean.shape[:-1],len(sample)))
    return np.stack([np.interp(np.arange(1001),sample,row)
                     for row in coarse.reshape(-1,len(sample))]).reshape(clean.shape)

def filt(vv,cc,tt):
    return (savgol_filter(vv,5,3,axis=-1),
            np.maximum(savgol_filter(cc,31,3,axis=-1),.001),
            savgol_filter(tt,21,3,axis=-1))

manifest={'selected_test_cells':cells.tolist(),
          'n_cells':len(cells),'excludes_original_128_and_earlier_16':True,
          'target_lambda':1.125,'calibration_lambdas':[1.,1.1],
          'fixed_filter_windows_ms':{'AP':5,'Ca':31,'T':21},
          'fixed_ridge_checkpoint':'land_stress_robust_ridge.joblib',
          'fixed_direct_checkpoint':'direct_2load_noise_augmented/model.pt',
          'fixed_refinement_gamma':.1,
          'target_trace_and_true_parameters_excluded_from_MATLAB_inputs':True,
          'conditions':stress['conditions']}
for number,(name,cfg) in enumerate(stress['conditions'].items()):
    rng=np.random.default_rng(20261001+number)
    step=cfg['step_ms']
    vo=observe(v,cfg['ap_sigma_mV'],step,rng)
    co=observe(ca,cfg['ca_sigma_uM'],step,rng)
    to=observe(t,cfg['t_sigma_kPa'],step,rng)
    if cfg['ca_drift_uM']:
        size=cfg['ca_drift_uM']
        co+=rng.uniform(-size,size,(len(cells),1))+rng.uniform(-size,size,(len(cells),1))*np.linspace(0,1,1001)
    if cfg['t_drift_kPa']:
        size=cfg['t_drift_kPa']
        to+=rng.uniform(-size,size,(len(cells),2,1))+rng.uniform(-size,size,(len(cells),2,1))*np.linspace(0,1,1001)
    vo,co,to=filt(vo,co,to)
    x=np.concatenate([vo[:,:601:5],co[:,:601:5],to[:,0,:601:5],to[:,1,:601:5]],axis=1)
    start=lo+np.clip(ridge.predict(x),0,1)*(hi-lo)
    path=base/f'land_stress_confirm_{name}_input.mat'
    savemat(path,{'cell_index_zero_based':cells.astype(float),
        'split_code':split[cells].astype(float),
        'V_mV':vo,'Cai_uM':co,'Tcal_kPa':to,
        'theta_start':start,'lo':lo,'hi':hi},do_compression=True)
    manifest.setdefault('input_sha256',{})[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    manifest.setdefault('ridge_boundary_coordinates',{})[name]=int(np.sum(np.isclose(start,lo)|np.isclose(start,hi)))
(base/'land_stress_confirmation_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({'n_cells':len(cells),'range':[int(cells.min()),int(cells.max())],
                  'boundary_coordinates':manifest['ridge_boundary_coordinates']},indent=2))
