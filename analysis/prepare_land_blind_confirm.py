"""Seal 128 fresh test cells for a target-blind true-Land confirmation."""
from __future__ import annotations

import hashlib
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
used=set(json.loads((base/'posterior_draws_16x8.json').read_text())['test_cell_indices_zero_based'])
with h5py.File(base/'merged_cohort.h5') as f:
    split=f['split_code'][()]
    eligible=np.array(sorted(set(np.flatnonzero(split==2))-used))
    rng=np.random.default_rng(20260929)
    chosen=np.sort(rng.choice(eligible,size=128,replace=False))
    lam=f['lambda'][()]
    ix=[int(np.argmin(abs(lam-x))) for x in [1.,1.1]]
    assert np.allclose(lam[ix],[1.,1.1])
    v=f['V'][:, :601:5]
    ca=f['Cai_uM'][()]
    ta=f['Ta_kPa'][:,ix,:601:5]
    theta=f['theta'][()]
    lo,hi=f['lo'][()],f['hi'][()]
    tcal=f['Ta_kPa'][chosen][:,ix,:]
x=np.concatenate([v,ca[:,:601:5],ta[:,0],ta[:,1]],axis=1)
y=(theta-lo)/(hi-lo)
train=np.flatnonzero(split!=2)
model=make_pipeline(StandardScaler(),Ridge(alpha=1.))
model.fit(x[train],y[train])
unit=np.clip(model.predict(x[chosen]),0,1)
start=lo+unit*(hi-lo)
blind=base/'land_blind_128_input.mat'
savemat(blind,{'cell_index_zero_based':chosen.astype(float),
    'split_code':split[chosen].astype(float),
    'Cai_uM':ca[chosen], 'Tcal_kPa':tcal,
    'theta_start':start,'lo':lo,'hi':hi},do_compression=True)
manifest={'selected_test_cells':chosen.tolist(),
    'excluded_previous_16_test_cells':True,
    'target_lambda':1.125,'calibration_lambdas':[1.,1.1],
    'target_waveform_absent_from_MATLAB_input':True,
    'fixed_gamma':.1,'fixed_fit_parameters':['Tref','ca50','nperm','beta0','beta1'],
    'frozen_setting_from_previous_4_validation_cells':True,
    'input_sha256':hashlib.sha256(blind.read_bytes()).hexdigest()}
(base/'land_blind_128_selection.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(f"sealed {len(chosen)} new test cells, sha256 {manifest['input_sha256'][:12]}")
