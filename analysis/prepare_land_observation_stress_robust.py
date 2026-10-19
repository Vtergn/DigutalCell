"""Training-only noise augmentation and fixed denoising for a robustness follow-up.

The 128-cell clean confirmation is untouched. This variant is exploratory:
it repairs the observed out-of-support Ridge initialization under noise.
"""
from pathlib import Path
import json
import hashlib
import h5py
import numpy as np
from scipy.io import loadmat,savemat
from scipy.signal import savgol_filter
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib

root=Path(__file__).resolve().parents[1]
base=root/'analysis/virtual_cardiomyocyte_5000'
manifest=json.loads((base/'land_observation_stress_manifest.json').read_text())
cells=np.array(manifest['cells'])
with h5py.File(base/'merged_cohort.h5') as f:
    lam=f['lambda'][()]
    ix=[int(np.argmin(abs(lam-x))) for x in [1.,1.1]]
    v=f['V'][()]
    ca=f['Cai_uM'][()]
    t=f['Ta_kPa'][:,ix,:]
    theta=f['theta'][()]
    lo,hi=f['lo'][()],f['hi'][()]
    split=f['split_code'][()]

def filt(vv,cc,tt):
    return (savgol_filter(vv,5,3,axis=-1),
            np.maximum(savgol_filter(cc,31,3,axis=-1),.001),
            savgol_filter(tt,21,3,axis=-1))

def features(vv,cc,tt):
    return np.concatenate([vv[:,:601:5],cc[:,:601:5],
                           tt[:,0,:601:5],tt[:,1,:601:5]],axis=1)

def perturb(clean,sigma,step,rng):
    sample=np.arange(0,1001,step)
    if sample[-1]!=1000:sample=np.r_[sample,1000]
    coarse=clean[...,sample]+rng.normal(0,sigma,(*clean.shape[:-1],len(sample)))
    return np.stack([np.interp(np.arange(1001),sample,row)
                     for row in coarse.reshape(-1,len(sample))]).reshape(clean.shape)

rng=np.random.default_rng(20260930)
train=np.flatnonzero(split!=2)
aug_x=[]
for cfg in [None,*manifest['conditions'].values()]:
    if cfg is None:
        vv,cc,tt=v[train],ca[train],t[train]
    else:
        step=cfg['step_ms']
        vv=perturb(v[train],cfg['ap_sigma_mV'],step,rng)
        cc=perturb(ca[train],cfg['ca_sigma_uM'],step,rng)
        tt=perturb(t[train],cfg['t_sigma_kPa'],step,rng)
        if cfg['ca_drift_uM']:
            size=cfg['ca_drift_uM']
            cc+=rng.uniform(-size,size,(len(train),1))+rng.uniform(-size,size,(len(train),1))*np.linspace(0,1,1001)
        if cfg['t_drift_kPa']:
            size=cfg['t_drift_kPa']
            tt+=rng.uniform(-size,size,(len(train),2,1))+rng.uniform(-size,size,(len(train),2,1))*np.linspace(0,1,1001)
    aug_x.append(features(*filt(vv,cc,tt)))
X=np.concatenate(aug_x,axis=0)
Y=np.tile((theta[train]-lo)/(hi-lo),(len(aug_x),1))
ridge=make_pipeline(StandardScaler(),Ridge(alpha=1.))
ridge.fit(X,Y)
joblib.dump(ridge,base/'land_stress_robust_ridge.joblib')
follow={'training_cell_count':len(train),'augmentation_variants':['clean',*manifest['conditions']],
        'ridge_alpha':1.,'fixed_filters':{'AP_ms':5,'Ca_ms':31,'T_ms':21},
        'test_cells':cells.tolist(),
        'target_waveform_absent_from_refinement_input':True,
        'purpose':'Exploratory follow-up to frozen noise stress; not part of sealed clean confirmation.'}
for name in manifest['conditions']:
    raw=loadmat(base/f'land_stress_{name}_input.mat')
    vv,cc,tt=filt(raw['V_mV'],raw['Cai_uM'],raw['Tcal_kPa'])
    unit=np.clip(ridge.predict(features(vv,cc,tt)),0,1)
    start=lo+unit*(hi-lo)
    output=base/f'land_stress_robust_{name}_input.mat'
    savemat(output,{'cell_index_zero_based':cells.astype(float),
                    'split_code':split[cells].astype(float),
                    'V_mV':vv,'Cai_uM':cc,'Tcal_kPa':tt,
                    'theta_start':start,'lo':lo,'hi':hi},do_compression=True)
    follow.setdefault('input_sha256',{})[name]=hashlib.sha256(output.read_bytes()).hexdigest()
    follow.setdefault('ridge_boundary_coordinates',{})[name]=int(np.sum(np.isclose(start,lo)|np.isclose(start,hi)))
(base/'land_observation_stress_robust_manifest.json').write_text(json.dumps(follow,indent=2))
print(json.dumps({'n_cells':len(cells),'ridge_boundary_coordinates':follow['ridge_boundary_coordinates']},indent=2))
