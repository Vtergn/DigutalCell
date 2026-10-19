"""Predeclare synthetic measurement stress tests with clean held-out targets.

All predictors receive the identical perturbed AP, Ca, T(1), T(1.1).
The MATLAB refinement input contains no hidden-load tension or true theta.
"""
from pathlib import Path
import json
import hashlib
import h5py
import numpy as np
from scipy.io import savemat
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

root=Path(__file__).resolve().parents[1]
base=root/'analysis/virtual_cardiomyocyte_5000'
cells=np.array(json.loads((base/'land_blind_128_selection.json').read_text())['selected_test_cells'][:64])
conditions={
    'low_noise':dict(ap_sigma_mV=1.,ca_sigma_uM=.005,t_sigma_kPa=.5,step_ms=1,
                     ca_drift_uM=0.,t_drift_kPa=0.),
    'high_noise':dict(ap_sigma_mV=3.,ca_sigma_uM=.015,t_sigma_kPa=1.5,step_ms=1,
                      ca_drift_uM=0.,t_drift_kPa=0.),
    'drift_sparse':dict(ap_sigma_mV=1.,ca_sigma_uM=.005,t_sigma_kPa=.5,step_ms=20,
                        ca_drift_uM=.004,t_drift_kPa=1.),
}
with h5py.File(base/'merged_cohort.h5') as f:
    lam=f['lambda'][()]
    ix=[int(np.argmin(abs(lam-x))) for x in [1.,1.1]]
    assert np.allclose(lam[ix],[1.,1.1])
    v=f['V'][()]
    ca=f['Cai_uM'][()]
    t=f['Ta_kPa'][:,ix,:]
    theta=f['theta'][()]
    lo,hi=f['lo'][()],f['hi'][()]
    split=f['split_code'][()]
train=np.flatnonzero(split!=2)
x=np.concatenate([v[:,:601:5],ca[:,:601:5],t[:,0,:601:5],t[:,1,:601:5]],axis=1)
ridge=make_pipeline(StandardScaler(),Ridge(alpha=1.))
ridge.fit(x[train],(theta[train]-lo)/(hi-lo))

def observe(clean,sigma,step,rng):
    sample=np.arange(0,1001,step)
    if sample[-1]!=1000: sample=np.r_[sample,1000]
    coarse=clean[...,sample]+rng.normal(0,sigma,(*clean.shape[:-1],len(sample)))
    return np.stack([np.interp(np.arange(1001),sample,row)
                     for row in coarse.reshape(-1,len(sample))]).reshape(clean.shape)

manifest={'cells':cells.tolist(),'target_lambda':1.125,
          'target_remains_clean_and_excluded_from_refinement':True,
          'input_model_ridge_trained_on_original_training_cells':True,
          'refinement_gamma':.1,'refinement_fit_parameters':['Tref','ca50','nperm','beta0','beta1'],
          'conditions':conditions,'noise_model':'independent zero-mean Gaussian at acquisition samples; linear interpolation for sparse sampling; cell/load specific uniform baseline offset and linear slope for drift_sparse',
          'prediction_target':'latent clean tension, not a noisy target measurement'}
for number,(name,cfg) in enumerate(conditions.items()):
    rng=np.random.default_rng(20260929+number)
    step=cfg['step_ms']
    vo=observe(v[cells],cfg['ap_sigma_mV'],step,rng)
    co=observe(ca[cells],cfg['ca_sigma_uM'],step,rng)
    to=observe(t[cells],cfg['t_sigma_kPa'],step,rng)
    if cfg['ca_drift_uM']:
        offset=rng.uniform(-cfg['ca_drift_uM'],cfg['ca_drift_uM'],(len(cells),1))
        slope=rng.uniform(-cfg['ca_drift_uM'],cfg['ca_drift_uM'],(len(cells),1))
        co+=offset+slope*np.linspace(0,1,1001)
    if cfg['t_drift_kPa']:
        offset=rng.uniform(-cfg['t_drift_kPa'],cfg['t_drift_kPa'],(len(cells),2,1))
        slope=rng.uniform(-cfg['t_drift_kPa'],cfg['t_drift_kPa'],(len(cells),2,1))
        to+=offset+slope*np.linspace(0,1,1001)
    co=np.maximum(co,.001)
    xx=np.concatenate([vo[:,:601:5],co[:,:601:5],to[:,0,:601:5],to[:,1,:601:5]],axis=1)
    start=lo+np.clip(ridge.predict(xx),0,1)*(hi-lo)
    path=base/f'land_stress_{name}_input.mat'
    savemat(path,{'cell_index_zero_based':cells.astype(float),
                  'split_code':split[cells].astype(float),
                  'V_mV':vo,'Cai_uM':co,'Tcal_kPa':to,
                  'theta_start':start,'lo':lo,'hi':hi},do_compression=True)
    manifest.setdefault('input_sha256',{})[name]=hashlib.sha256(path.read_bytes()).hexdigest()
(base/'land_observation_stress_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({'n_cells':len(cells),'conditions':conditions},indent=2))
