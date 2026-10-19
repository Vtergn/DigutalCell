"""Prepare train-fitted ridge starts and observed Ca/T for true-Land refinement."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from scipy.io import savemat
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

root=Path(__file__).resolve().parents[1]
base=root/'analysis/virtual_cardiomyocyte_5000'
with h5py.File(base/'merged_cohort.h5') as f:
    v=f['V'][()]
    ca=f['Cai_uM'][()]
    ta=f['Ta_kPa'][()]
    theta=f['theta'][()]
    split=f['split_code'][()]
    lam=f['lambda'][()]
    lo,hi=f['lo'][()],f['hi'][()]
ix=[int(np.argmin(abs(lam-x))) for x in [1.,1.1,1.125]]
assert np.allclose(lam[ix],[1.,1.1,1.125])
x=np.concatenate([v[:,:601:5],ca[:,:601:5],ta[:,ix[0],:601:5],
                  ta[:,ix[1],:601:5]],axis=1)
y=(theta-lo)/(hi-lo)
test=np.array(json.loads((base/'posterior_draws_16x8.json').read_text())['test_cell_indices_zero_based'])
rng=np.random.default_rng(20260929)
valid=np.sort(rng.choice(np.flatnonzero(split==1),size=4,replace=False))
start=np.zeros((20,12))
for target_ids,fit_ids in [(valid,np.flatnonzero(split==0)),
                           (test,np.flatnonzero(split!=2))]:
    fit=make_pipeline(StandardScaler(),Ridge(alpha=1.))
    fit.fit(x[fit_ids],y[fit_ids])
    z=np.clip(fit.predict(x[target_ids]),0,1)
    positions=range(0,4) if len(target_ids)==4 else range(4,20)
    start[list(positions)]=lo+z*(hi-lo)
ids=np.r_[valid,test]
out=base/'land_refinement_input.mat'
savemat(out,{'cell_index_zero_based':ids.astype(float),
    'split_code':split[ids].astype(float),
    'Cai_uM':ca[ids], 'Tcal_kPa':ta[ids][:,ix[:2],:],
    'Ttarget_kPa':ta[ids,ix[2],:],
    'theta_start':start,'theta_true':theta[ids],
    'lo':lo,'hi':hi},do_compression=True)
(base/'land_refinement_selection.json').write_text(json.dumps({
    'validation_cells':valid.tolist(),'test_cells':test.tolist(),
    'ridge_fit_validation':'training split only',
    'ridge_fit_test':'training plus validation splits',
    'calibration_loads':[1.,1.1],'target_load':1.125,
    'Ca_conditioning':'baseline measured Ca, which is load-invariant in this one-way simulator'},indent=2))
print('validation',valid.tolist(),'test',test.tolist(),'saved',out)
