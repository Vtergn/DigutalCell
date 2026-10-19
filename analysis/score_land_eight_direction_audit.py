"""Describe target-relevant observable and null mechanical directions."""
from pathlib import Path
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]
base=root/'analysis/virtual_cardiomyocyte_5000'
with h5py.File(base/'land_eight_direction_audit_16.mat') as f:
    jo=f['J_observed'][()].transpose(2,1,0)
    jt=f['J_target'][()].transpose(2,1,0)
    sigma=f['sigma_effective'][()].ravel()
    cells=f['cell_index_zero_based'][()].ravel().astype(int)
assert jo.shape==(16,122,8) and jt.shape==jo.shape
prior_sd=1/np.sqrt(12) # true synthetic uniform benchmark prior in range units
rows=[]
for i,cell in enumerate(cells):
    _,singular,vh=np.linalg.svd(jo[i]/sigma[i],full_matrices=False)
    v=vh.T
    rank=int(np.sum(singular>1/prior_sd))
    local_sd=1/np.sqrt(singular**2+1/prior_sd**2)
    targets=[]
    for j,lam in enumerate([1.125,1.15]):
        target=jt[i,j::2,:]
        total=np.linalg.norm(target,'fro')**2
        observed=np.linalg.norm(target@v[:,:rank],'fro')**2
        energy=np.sum((target@v)**2,axis=0)
        variance_by_direction=energy*local_sd**2
        targets.append({'lambda':lam,
                        'target_derivative_energy_in_observable_subspace':float(observed/total),
                        'target_derivative_energy_in_null_subspace':float(1-observed/total),
                        'target_linearized_variance_from_null_subspace':float(variance_by_direction[rank:].sum()/variance_by_direction.sum()),
                        'target_variance_relevant_directions_1pct':int(np.sum(variance_by_direction/variance_by_direction.sum()>=.01)),
                        'target_relevant_singular_directions_1pct':int(np.sum(energy/total>=.01)),
                        'target_energy_by_observation_direction':(energy/total).tolist()})
    rows.append({'cell':int(cell),'noise_whitened_singular_values':singular.tolist(),
                 'observable_rank_relative_to_uniform_prior':rank,
                 'targets':targets})
report={'n_cells':len(cells),'prior_sd_uniform_unit_range':prior_sd,
        'decision_rule':'SVD of observed-load Jacobian divided by cell-specific sigma; singular > 1/prior_sd is data-informed relative to prior.',
        'median_observable_rank':float(np.median([x['observable_rank_relative_to_uniform_prior'] for x in rows])),
        'median_target_null_energy':{str(lam):float(np.median([r['targets'][j]['target_derivative_energy_in_null_subspace'] for r in rows])) for j,lam in enumerate([1.125,1.15])},
        'median_target_variance_from_null':{str(lam):float(np.median([r['targets'][j]['target_linearized_variance_from_null_subspace'] for r in rows])) for j,lam in enumerate([1.125,1.15])},
        'cells':rows,
        'scope':'Local derivative geometry under assumed sigma and synthetic prior; not proof of globally identifiable parameters.'}
(base/'land_eight_direction_audit_16_score.json').write_text(json.dumps(report,indent=2))
fig,axes=plt.subplots(1,2,figsize=(10,4.1),constrained_layout=True)
for j,lam in enumerate([1.125,1.15]):
    for ax,key in zip(axes,['target_derivative_energy_in_null_subspace','target_linearized_variance_from_null_subspace']):
        vals=[r['targets'][j][key] for r in rows]
        ax.scatter(np.full(len(vals),lam),vals,alpha=.7)
        ax.plot(lam,np.median(vals),'k_',markersize=18)
for ax,title in zip(axes,['Raw target sensitivity','Local predictive variance']):
    ax.set(xlabel='Target stretch',ylabel='Fraction from weak observed directions',title=title)
    ax.grid(alpha=.2)
fig.savefig(base/'land_eight_direction_audit_16.png',dpi=180)
print(json.dumps({'n_cells':len(cells),'median_observable_rank':report['median_observable_rank'],
                  'median_target_null_energy':report['median_target_null_energy'],
                  'median_target_variance_from_null':report['median_target_variance_from_null']},indent=2))
