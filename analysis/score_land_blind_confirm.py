"""Unseal target only after all 128 target-blind refinements finish."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_direct_load_predictor import DirectLoadNetwork, get_data


root=Path(__file__).resolve().parents[1]
base=root/'analysis/virtual_cardiomyocyte_5000'
selection=json.loads((base/'land_blind_128_selection.json').read_text())
cells=np.array(selection['selected_test_cells'])
folder=base/'land_blind_128_results'
files=[folder/f'land_refine_test_{i:02d}.mat' for i in range(1,len(cells)+1)]
missing=[str(path) for path in files if not path.exists()]
if missing:
    raise SystemExit(f'{len(missing)} refinements incomplete')

before=[];after=[];cal_before=[];cal_after=[];exitflags=[];evaluations=[];estimated=[]
for cell,path in zip(cells,files):
    with h5py.File(path) as f:
        r=f['result']
        actual_cell=int(r['cell_index_zero_based'][()].item())
        if actual_cell!=cell: raise ValueError(f'Cell mismatch {path}')
        before.append(r['before_target_waveform'][()].ravel()[:601:2])
        after.append(r['after_target_waveform'][()].ravel()[:601:2])
        cal_before.append(float(r['before_cal_MAE_kPa'][()].item()))
        cal_after.append(float(r['after_cal_MAE_kPa'][()].item()))
        exitflags.append(int(r['exitflag'][()].item()))
        evaluations.append(int(r['function_evaluations'][()].item()))
        estimated.append(r['theta_after'][()].ravel())
before,after,estimated=map(np.stack,[before,after,estimated])
with h5py.File(base/'merged_cohort.h5') as f:
    lam=f['lambda'][()]
    ix=[int(np.argmin(abs(lam-x))) for x in [1.,1.1,1.125]]
    truth=f['Ta_kPa'][cells,ix[2],:601:2]
    t1=f['Ta_kPa'][cells,ix[0],:601:2]
    t11=f['Ta_kPa'][cells,ix[1],:601:2]
    true_theta=f['theta'][cells,4:12]
    lo,hi=f['lo'][4:12],f['hi'][4:12]
    split=f['split_code'][cells]
assert np.all(split==2)
linear=t1+1.25*(t11-t1)
torch.set_num_threads(4)
ap,ca,t,_,_=get_data(base/'merged_cohort.h5')
net=DirectLoadNetwork(2,ap.shape[1])
net.load_state_dict(torch.load(base/'direct_2load/model.pt',map_location='cpu',weights_only=True))
net.eval()
with torch.no_grad():
    direct=net(ap[cells],ca[cells],t[cells][:,ix[:2],:],
               torch.tensor(lam[ix[:2]],dtype=torch.float32).repeat(len(cells),1),
               torch.full((len(cells),),1.125)).numpy()
errors={
    'Ridge_Land_before':np.mean(abs(before-truth),axis=1),
    'Ridge_Land_refined':np.mean(abs(after-truth),axis=1),
    'direct_NN':np.mean(abs(direct-truth),axis=1),
    'linear_extrapolation':np.mean(abs(linear-truth),axis=1),
}
peaks={
    'Ridge_Land_before':abs(before.max(axis=1)-truth.max(axis=1)),
    'Ridge_Land_refined':abs(after.max(axis=1)-truth.max(axis=1)),
    'direct_NN':abs(direct.max(axis=1)-truth.max(axis=1)),
    'linear_extrapolation':abs(linear.max(axis=1)-truth.max(axis=1)),
}
rng=np.random.default_rng(20260929)
bootstrap=rng.integers(0,len(cells),(5000,len(cells)))
advantages={}
for reference in ['direct_NN','linear_extrapolation','Ridge_Land_before']:
    delta=errors[reference]-errors['Ridge_Land_refined']
    advantages[reference]={
        'mean_MAE_reduction_kPa':float(delta.mean()),
        'paired_bootstrap_95pct_CI_kPa':np.quantile(delta[bootstrap].mean(axis=1),[.025,.975]).tolist(),
        'cells_refined_better':int((delta>0).sum()),
    }
result={
    'n_fresh_test_cells':len(cells),'test_cell_indices':cells.tolist(),
    'mean_waveform_MAE_kPa':{key:float(val.mean()) for key,val in errors.items()},
    'median_waveform_MAE_kPa':{key:float(np.median(val)) for key,val in errors.items()},
    'mean_peak_MAE_kPa':{key:float(val.mean()) for key,val in peaks.items()},
    'comparison':advantages,
    'mean_calibration_MAE_before_after_kPa':[float(np.mean(cal_before)),float(np.mean(cal_after))],
    'mean_mechanical_parameter_MAE_prior_scaled':float(np.mean(abs((estimated-true_theta)/(hi-lo)))),
    'optimization_nonpositive_exitflag_n':int(sum(x<=0 for x in exitflags)),
    'function_evaluations_median_max':[float(np.median(evaluations)),int(max(evaluations))],
    'per_cell':[{'cell':int(cells[i]),
                 **{key+'_MAE_kPa':float(val[i]) for key,val in errors.items()},
                 'calibration_before_MAE_kPa':cal_before[i],
                 'calibration_after_MAE_kPa':cal_after[i],
                 'exitflag':exitflags[i],
                 'function_evaluations':evaluations[i]} for i in range(len(cells))],
    'scope':'Target waveform excluded from MATLAB input and optimizer. No measurement noise or simulator mismatch; this is a point-estimate twin, not a calibrated posterior.'
}
(base/'land_blind_128_score.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
axes[0].scatter(errors['direct_NN'],errors['Ridge_Land_refined'],s=18,alpha=.7)
maximum=float(max(np.max(errors['direct_NN']),np.max(errors['Ridge_Land_refined'])))
axes[0].plot([0,maximum],[0,maximum],'k--',lw=1)
axes[0].set(xlabel='Direct NN waveform MAE (kPa)',
            ylabel='Refined mechanistic twin MAE (kPa)')
axes[0].grid(alpha=.2)
delta=errors['direct_NN']-errors['Ridge_Land_refined']
axes[1].hist(delta,bins=20,color='#4e79a7',edgecolor='white')
axes[1].axvline(0,color='black',ls='--',lw=1)
axes[1].set(xlabel='Direct NN − twin MAE (kPa)',ylabel='Cells')
axes[1].grid(axis='y',alpha=.2)
fig.savefig(base/'land_blind_128_comparison.png',dpi=170)
plt.close(fig)
print(json.dumps({key:result[key] for key in ['n_fresh_test_cells','mean_waveform_MAE_kPa',
    'mean_peak_MAE_kPa','comparison','mean_calibration_MAE_before_after_kPa',
    'optimization_nonpositive_exitflag_n']},indent=2))
