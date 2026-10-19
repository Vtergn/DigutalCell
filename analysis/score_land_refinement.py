"""Same-cell blind-load comparison of true-Land refinement and baselines."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


root=Path(__file__).resolve().parents[1]
base=root/'analysis/virtual_cardiomyocyte_5000'
posterior=json.loads((base/'posterior_true_forward_16x8_score.json').read_text())
comparators={row['cell']:row for row in posterior['modes']['3']['per_cell']}
with h5py.File(base/'merged_cohort.h5') as f:
    lam=f['lambda'][()]
    target_index=int(np.argmin(abs(lam-1.125)))
    assert abs(lam[target_index]-1.125)<1e-8
    truth=f['Ta_kPa'][()][:,target_index,:601:2]


def read_one(path):
    with h5py.File(path) as f:
        r=f['result']
        scalar=lambda key:float(r[key][()].item())
        return {
            'cell':int(scalar('cell_index_zero_based')),
            'split_code':int(scalar('split_code')),
            'cal_before_MAE_kPa':scalar('before_cal_MAE_kPa'),
            'cal_after_MAE_kPa':scalar('after_cal_MAE_kPa'),
            'before':r['before_target_waveform'][()].ravel()[:601:2],
            'after':r['after_target_waveform'][()].ravel()[:601:2],
            'function_evaluations':int(scalar('function_evaluations')),
            'exitflag':int(scalar('exitflag')),
        }


valid=[read_one(base/f'land_refine_val_{i}.mat') for i in range(1,5)]
test=[read_one(base/'land_refinement_results'/f'land_refine_test_{i:02d}.mat')
      for i in range(5,21)]
assert all(row['split_code']==1 for row in valid)
assert all(row['split_code']==2 for row in test)
for row in valid+test:
    target=truth[row['cell']]
    row['initial_waveform_MAE_kPa']=float(np.mean(abs(row['before']-target)))
    row['refined_waveform_MAE_kPa']=float(np.mean(abs(row['after']-target)))
    row['initial_peak_MAE_kPa']=float(abs(row['before'].max()-target.max()))
    row['refined_peak_MAE_kPa']=float(abs(row['after'].max()-target.max()))
    row.pop('before');row.pop('after')
for row in test:
    c=comparators[row['cell']]
    row['direct_waveform_MAE_kPa']=c['direct_waveform_MAE_kPa']
    row['linear_waveform_MAE_kPa']=c['static_waveform_MAE_kPa']
    row['posterior_waveform_MAE_kPa']=c['posterior_waveform_MAE_kPa']


def summarize(rows, include_baselines):
    names=['initial_waveform_MAE_kPa','refined_waveform_MAE_kPa',
           'initial_peak_MAE_kPa','refined_peak_MAE_kPa',
           'cal_before_MAE_kPa','cal_after_MAE_kPa']
    if include_baselines:
        names += ['direct_waveform_MAE_kPa','linear_waveform_MAE_kPa',
                  'posterior_waveform_MAE_kPa']
    return {key:float(np.mean([row[key] for row in rows])) for key in names}

result={'method':'Ridge initialization, observed Ca, true Land lsqnonlin 5-parameter calibration',
        'calibration_loads':[1.,1.1],'target_load':1.125,
        'regularization_gamma':.1,'fit_parameters':['Tref','ca50','nperm','beta0','beta1'],
        'validation_n':4,'test_n':16,
        'validation_mean':summarize(valid,False),
        'test_mean':summarize(test,True),
        'validation_cells':valid,'test_cells':test,
        'scope':'Exploratory 16-cell blind test; validation setting fixed before test; same simulator family and noiseless Ca; wider test and uncertainty not yet established.'}
(base/'land_refinement_score.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({'validation_mean':result['validation_mean'],
                  'test_mean':result['test_mean'],
                  'test_cells_improved':int(sum(r['refined_waveform_MAE_kPa']<r['initial_waveform_MAE_kPa'] for r in test))},indent=2))
