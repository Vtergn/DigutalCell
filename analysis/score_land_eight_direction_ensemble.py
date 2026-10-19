"""Score exploratory eight-parameter Land ensembles on withheld loads."""
from pathlib import Path
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt

base = Path(__file__).resolve().parent / 'virtual_cardiomyocyte_5000'
with h5py.File(base/'land_eight_direction_ensemble_16.mat') as f:
    d = f['prediction'][()].transpose(3,2,1,0)  # cell, load, time, draw
    point = f['point'][()].transpose(2,1,0)
    w = f['weights'][()].T
    cells = f['cell_index_zero_based'][()].ravel().astype(int)
    failures = f['candidate_failure'][()].ravel()
    reflected = f['reflected_fraction'][()].ravel()
    sigma = None
with h5py.File(base/'land_load_grid_truth.mat') as f:
    truth = f['truth'][()].transpose(2,1,0)
    grid_cells = f['cell_index_zero_based'][()].ravel().astype(int)
assert np.array_equal(cells,grid_cells[:len(cells)])
truth = truth[:len(cells),[4,5],::2]
assert d.shape[:3] == truth.shape

def wquant(x,weights,p):
    order=np.argsort(x,axis=-1)
    sorted_x=np.take_along_axis(x,order,axis=-1)
    sorted_w=np.take_along_axis(np.broadcast_to(weights,x.shape),order,axis=-1)
    return np.take_along_axis(sorted_x,np.argmax(np.cumsum(sorted_w,axis=-1)>=p,axis=-1)[...,None],axis=-1)[...,0]

rows=[]
for j,lam in enumerate([1.125,1.15]):
    x=d[:,j,:,:]
    y=truth[:,j,:]
    low=wquant(x,w[:,None,:],.05)
    high=wquant(x,w[:,None,:],.95)
    covered=(y>=low)&(y<=high)
    predictive_mean=np.sum(w[:,None,:]*x,axis=-1)
    first=np.sum(w[:,None,:]*abs(x-y[:,:,None]),axis=-1)
    pair=abs(x[:,:,:,None]-x[:,:,None,:])
    second=.5*np.sum(w[:,None,:,None]*w[:,None,None,:]*pair,axis=(-2,-1))
    rows.append({'target_lambda':lam,'point_MAE_kPa':float(np.mean(abs(point[:,j]-y))),
                 'ensemble_mean_MAE_kPa':float(np.mean(abs(predictive_mean-y))),
                 'pointwise_90pct_coverage':float(covered.mean()),
                 'simultaneous_cell_90pct_coverage':float(covered.all(axis=1).mean()),
                 'mean_90pct_interval_width_kPa':float(np.mean(high-low)),
                 'CRPS_kPa':float(np.mean(first-second)),
                 'per_cell_90pct_coverage':covered.mean(axis=1).tolist()})

ess=1/np.sum(w*w,axis=1)
report={'n_cells':len(cells),'n_candidates_per_cell':d.shape[-1],
        'n_candidate_failures':int(failures.sum()),
        'mean_reflected_candidate_fraction':float(reflected.mean()),
        'ESS_median':float(np.median(ess)),
        'ESS_min':float(ess.min()),
        'results':rows,
        'scope':'Exploratory local eight-parameter compatible-twin ensemble. Gaussian Jacobian proposals and ad hoc observed-fit weights are not an exact Bayesian posterior. No target outcome used to generate or weight candidates. The 16 cells were previously used for the five-parameter pilot; they are not an independent confirmation set.'}
(base/'land_eight_direction_ensemble_16_score.json').write_text(json.dumps(report,indent=2))

fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
for j,ax in enumerate(axes):
    x=d[0,j]
    lo=wquant(x,w[0],.05);hi=wquant(x,w[0],.95)
    t=np.arange(0,601,2)
    ax.fill_between(t,lo,hi,color='#619f85',alpha=.32,label='Eight-parameter ensemble 90%')
    ax.plot(t,point[0,j],color='#276b50',lw=1.2,label='Point twin')
    ax.plot(t,truth[0,j],color='#333333',lw=1.1,label='Withheld truth')
    ax.set(xlabel='Time (ms)',ylabel='Active tension (kPa)',title=f'λ={ [1.125,1.15][j] }')
    ax.grid(alpha=.2)
axes[0].legend(fontsize=7)
fig.savefig(base/'land_eight_direction_ensemble_16.png',dpi=180)
print(json.dumps({k:report[k] for k in ['n_cells','n_candidates_per_cell','n_candidate_failures','ESS_median','results']},indent=2))
