"""Split-conformal predictive bands for a frozen point twin.

Calibration: 64 exploratory stress cells. Evaluation: 64 disjoint fresh
stress cells. The same synthetic observation condition is used on both.
One score per cell (maximum over time) gives simultaneous waveform bands.
"""
from pathlib import Path
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]
base=root/'analysis/virtual_cardiomyocyte_5000'
cal_cells=np.array(json.loads((base/'land_observation_stress_manifest.json').read_text())['cells'])
test_cells=np.array(json.loads((base/'land_stress_confirmation_manifest.json').read_text())['selected_test_cells'])
assert len(cal_cells)==len(test_cells)==64
assert not set(cal_cells)&set(test_cells)
with h5py.File(base/'merged_cohort.h5') as f:
    lam=f['lambda'][()]
    j=int(np.argmin(abs(lam-1.125)))
    assert abs(lam[j]-1.125)<1e-6
    cal_true=f['Ta_kPa'][cal_cells,j,:601:2]
    test_true=f['Ta_kPa'][test_cells,j,:601:2]

def read_point(kind,group,cells):
    rows=[]
    folder=base/f'land_stress_{group}_{kind}_results'
    for i,cell in enumerate(cells,1):
        with h5py.File(folder/f'land_refine_test_{i:02d}.mat') as f:
            r=f['result']
            assert int(r['cell_index_zero_based'][()].item())==cell
            rows.append(r['after_target_waveform'][()].ravel()[:601:2])
    return np.stack(rows)

def summary(lo,hi,truth):
    covered=(truth>=lo)&(truth<=hi)
    miss_lo=np.maximum(lo-truth,0)
    miss_hi=np.maximum(truth-hi,0)
    return {'simultaneous_cell_coverage':float(covered.all(axis=1).mean()),
            'covered_cells':int(covered.all(axis=1).sum()),
            'pointwise_coverage':float(covered.mean()),
            'mean_interval_width_kPa':float(np.mean(hi-lo)),
            'mean_interval_score_alpha_0p1_kPa':float(np.mean(hi-lo+20*(miss_lo+miss_hi)))}

alpha=.1
k=int(np.ceil((len(cal_cells)+1)*(1-alpha)))
rows=[]
for kind in ['low_noise','high_noise','drift_sparse']:
    cal_point=read_point(kind,'robust',cal_cells)
    test_point=read_point(kind,'confirm',test_cells)
    residual=cal_true-cal_point
    scale=np.maximum(np.sqrt(np.mean(residual**2,axis=0)),.5)
    entries={}
    for method,cal_score,test_scale in [
        ('uniform',np.max(abs(residual),axis=1),np.ones(test_point.shape[1])),
        ('time_scaled',np.max(abs(residual)/scale,axis=1),scale)]:
        q=float(np.partition(cal_score,k-1)[k-1])
        width=q*test_scale
        lower=np.maximum(test_point-width,0)
        upper=test_point+width
        entries[method]={'calibration_rank_k':k,'conformal_quantile':q,
                         **summary(lower,upper,test_true)}
    rows.append({'condition':kind,'point_MAE_kPa':float(np.mean(abs(test_point-test_true))),
                 'methods':entries})
report={'n_calibration_cells':len(cal_cells),'n_independent_test_cells':len(test_cells),
        'target_lambda':1.125,'time_grid_ms':'0:2:600',
        'nominal_simultaneous_coverage':.9,'score_unit':'cell, max over 301 time samples',
        'calibration_and_test_cell_disjoint':True,
        'results':rows,
        'scope':'Distribution-matched synthetic observation conditions only. This is a conformal predictive band around the frozen point twin, not a Bayesian credible interval. Pipeline settings were developed using the exploratory calibration cohort, so formal exchangeability guarantees are not asserted.'}
(base/'land_conformal_point_score.json').write_text(json.dumps(report,indent=2))
fig,ax=plt.subplots(figsize=(8,4.5),constrained_layout=True)
x=np.arange(len(rows));width=.32
for j,method in enumerate(['uniform','time_scaled']):
    ax.bar(x+(j-.5)*width,
           [r['methods'][method]['mean_interval_width_kPa'] for r in rows],
           width,label=method.replace('_',' '),color=['#5076a6','#2d7f5e'][j])
    for i,r in enumerate(rows):
        v=r['methods'][method]
        ax.text(x[i]+(j-.5)*width,v['mean_interval_width_kPa']+.1,
                f"{v['covered_cells']}/64",ha='center',fontsize=8)
ax.set_xticks(x,[r['condition'].replace('_',' ') for r in rows])
ax.set(ylabel='Mean 90% band width (kPa)',title='Conformal point-twin bands (labels: cells covered)')
ax.legend();ax.grid(axis='y',alpha=.2)
fig.savefig(base/'land_conformal_point_comparison.png',dpi=180)
print(json.dumps(report,indent=2))
