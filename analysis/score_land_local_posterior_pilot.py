"""Evaluate exploratory conditional Laplace draws against withheld load-grid truth."""
from pathlib import Path
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]
base=root/'analysis/virtual_cardiomyocyte_5000'
with h5py.File(base/'land_local_posterior_pilot.mat') as f:
    draws=f['draws'][()].transpose(3,2,1,0) # cell, load, time, draw
    point=f['point'][()].transpose(2,1,0)
    sv=f['singular_values'][()].T
    sigma=f['sigma_effective'][()].ravel()
    clipped=f['clipped_fraction'][()].ravel()
    failed=f['failed_draws'][()].ravel()
    cells=f['cell_index_zero_based'][()].ravel().astype(int)
    target_lam=f['target_lambdas'][()].ravel()
    prior_sd=float(f['prior_sd'][()].item())
assert np.allclose(target_lam,[1.125,1.15])
with h5py.File(base/'land_load_grid_truth.mat') as f:
    truth=f['truth'][()].transpose(2,1,0)
    grid_cells=f['cell_index_zero_based'][()].ravel().astype(int)
assert np.array_equal(cells,grid_cells[:len(cells)])
truth=truth[:len(cells),[4,5],::2]
rows=[]
for j,lam in enumerate(target_lam):
    d=draws[:,j,:,:]
    y=truth[:,j,:]
    valid=np.isfinite(d).all(axis=(1,2))
    if not valid.any(): raise ValueError('No valid posterior draws')
    lo,hi=np.quantile(d[valid],[.05,.95],axis=-1)
    coverage=np.mean((y[valid]>=lo)&(y[valid]<=hi))
    mean_absolute=np.mean(abs(d[valid]-y[valid,:,None]),axis=-1)
    spread=.5*np.mean(abs(d[valid,:,:,None]-d[valid,:,None,:]),axis=(-2,-1))
    crps=np.mean(mean_absolute-spread)
    rows.append({'target_lambda':float(lam),'n_valid_cells':int(valid.sum()),
                 'n_total_cells':len(cells),'point_MAE_kPa':float(np.mean(abs(point[valid,j]-y[valid]))),
                 'posterior_mean_MAE_kPa':float(np.mean(abs(d[valid].mean(axis=-1)-y[valid]))),
                 'pointwise_90pct_coverage':float(coverage),
                 'mean_90pct_interval_width_kPa':float(np.mean(hi-lo)),
                 'CRPS_kPa':float(crps),
                 'per_cell_90pct_coverage':np.mean((y[valid]>=lo)&(y[valid]<=hi),axis=1).tolist()})
report={'n_cells':len(cells),'n_draws':draws.shape[-1],
        'conditioned_parameters':['Tref','ca50','nperm','beta0','beta1'],
        'fixed_parameters':['TRPN_n','phi','perm50','all_electrical'],
        'prior_sd_in_parameter_range_units':prior_sd,
        'sigma_effective_mean_kPa':float(sigma.mean()),
        'draws_clipped_cell_mean_fraction':float(clipped.mean()),
        'failed_draws_total':int(failed.sum()),
        'whitened_jacobian_singular_values_median':np.median(sv,axis=0).tolist(),
        'results':rows,
        'scope':'Exploratory local conditional approximation, not a calibrated full-parameter posterior. Sigma includes a fixed 0.5 kPa floor plus calibration residual RMS; no held-out-target tuning.'}
(base/'land_local_posterior_pilot_score.json').write_text(json.dumps(report,indent=2))
fig,axes=plt.subplots(1,2,figsize=(11,3.6),constrained_layout=True)
for j,ax in enumerate(axes):
    d=draws[0,j]
    low,high=np.quantile(d,[.05,.95],axis=-1)
    x=np.arange(0,601,2)
    ax.fill_between(x,low,high,alpha=.3,color='#2d7f5e',label='Local 90% interval')
    ax.plot(x,truth[0,j],color='black',lw=1.5,label='True simulator output')
    ax.plot(x,point[0,j],color='#2d7f5e',lw=1.1,label='Point twin')
    ax.set(xlabel='Time (ms)',ylabel='Active tension (kPa)',title=f'λ={target_lam[j]:.3f}, cell {cells[0]}')
    ax.grid(alpha=.2)
axes[0].legend(fontsize=8)
fig.savefig(base/'land_local_posterior_pilot.png',dpi=180)
print(json.dumps({k:report[k] for k in ['n_cells','failed_draws_total','draws_clipped_cell_mean_fraction','results']},indent=2))
