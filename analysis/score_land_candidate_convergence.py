"""Check whether 12 full-Land candidates approximate a 24-candidate pilot.

Uses the same first 12 candidates within each of 16 exploratory cells, so
the contrast isolates Monte Carlo count rather than a fresh random seed.
"""
from pathlib import Path
import json
import h5py
import numpy as np

base=Path(__file__).resolve().parent/'virtual_cardiomyocyte_5000'
with h5py.File(base/'land_eight_direction_ensemble_16.mat') as f:
    draws=f['prediction'][()].transpose(3,2,1,0)
    weight24=f['weights'][()].T
    theta_draw=f['theta_draw'][()].transpose(2,1,0)
    cells=f['cell_index_zero_based'][()].ravel().astype(int)
with h5py.File(base/'land_load_grid_truth.mat') as f:
    true=f['truth'][()].transpose(2,1,0)[:16,[4,5],::2]
    assert np.array_equal(cells,f['cell_index_zero_based'][()].ravel().astype(int)[:16])
assert draws.shape==(16,2,301,24)
with h5py.File(base/'merged_cohort.h5') as f:
    theta_true=f['theta'][cells,4:12]
within=(theta_true>=theta_draw.min(axis=2))&(theta_true<=theta_draw.max(axis=2))

def weighted_crps(d,w,y):
    a=np.sum(w[:,None,:]*abs(d-y[:,:,None]),axis=-1)
    b=.5*np.sum(w[:,None,:,None]*w[:,None,None,:]
                  *abs(d[:,:,:,None]-d[:,:,None,:]),axis=(-2,-1))
    return float(np.mean(a-b))

rows=[]
for j,lam in enumerate([1.125,1.15]):
    y=true[:,j]
    predictions={}
    for count in [12,24]:
        d=draws[:,j,:,:count]
        w=weight24[:,:count]
        w=w/w.sum(axis=1,keepdims=True)
        mean=np.sum(d*w[:,None,:],axis=-1)
        sd=np.sqrt(np.sum(w[:,None,:]*(d-mean[:,:,None])**2,axis=-1))
        predictions[count]={'mean':mean,'sd':sd,
            'ESS_median':float(np.median(1/np.sum(w*w,axis=1))),
            'mean_MAE_kPa':float(np.mean(abs(mean-y))),
            'mean_predictive_sd_kPa':float(sd.mean()),
            'CRPS_kPa':weighted_crps(d,w,y)}
    mean12=predictions[12]['mean'];mean24=predictions[24]['mean']
    sd12=predictions[12]['sd'];sd24=predictions[24]['sd']
    rows.append({'target_lambda':lam,
        'n12':{k:v for k,v in predictions[12].items() if k not in ['mean','sd']},
        'n24':{k:v for k,v in predictions[24].items() if k not in ['mean','sd']},
        'mean_absolute_change_in_predicted_mean_kPa':float(np.mean(abs(mean12-mean24))),
        'mean_absolute_change_in_predicted_sd_kPa':float(np.mean(abs(sd12-sd24))),
        'median_per_cell_relative_change_in_peak_sd':float(np.median(
            np.abs(sd12.max(axis=1)-sd24.max(axis=1))/np.maximum(sd24.max(axis=1),.1)))})
report={'n_cells':16,'nested_candidate_counts':[12,24],'results':rows,
        'true_parameters_in_candidate_marginal_range_per_parameter':within.mean(axis=0).tolist(),
        'cells_true_parameters_in_all_eight_marginal_ranges':int(within.all(axis=1).sum()),
        'scope':'Exploratory clean cells only. Candidate prefixes are nested within each cell. This is a convergence diagnostic, not a new independent prediction test.'}
(base/'land_candidate_convergence_16_score.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
