"""True-equation local observability versus unseen-load target relevance.

SVD thresholds here are numerical diagnostics under explicit trace scaling,
not formal measurement-noise-calibrated Fisher confidence intervals.
"""
import json
from pathlib import Path

import h5py
import numpy as np

root=Path(__file__).resolve().parents[1]
folder=root/'analysis/virtual_cardiomyocyte_5000'
names=['ICaL','Jup','INaCa','IKr','Tref','ca50','TRPN_n',
       'nperm','beta0','beta1','phi','perm50']
rows=[]
for i in range(1,5):
    with h5py.File(folder/f'jacobian_{i}.mat') as f:
        jv=f['Jv'][()].T
        jca=f['Jca'][()].T
        jt=f['Jt'][()].transpose(2,1,0)
        cell=int(f['cell_index_zero_based'][()].item())
    assert jv.shape==(61,12) and jt.shape==(3,61,12)
    # Per-trace scale, then divide by sqrt(samples) to avoid sample-count bias.
    ap=jv/(100*np.sqrt(61)); ca=jca/(.5*np.sqrt(61))
    ta=jt/(50*np.sqrt(61))
    modalities={
        'AP+Ca':np.concatenate([ap,ca]),
        'AP+Ca+T1.0':np.concatenate([ap,ca,ta[0]]),
        'AP+Ca+T1.0+T1.1':np.concatenate([ap,ca,ta[0],ta[1]]),
    }
    target=ta[2]
    cell_result={'cell_index_zero_based':cell, 'modalities':{},
                 'parameter_target_derivative_RMS':dict(zip(names,np.sqrt((target**2).sum(axis=0)).tolist()))}
    for mode, job in modalities.items():
        _, singular, vt=np.linalg.svd(job,full_matrices=False)
        sensitivity={name:float(np.linalg.norm(job[:,j])) for j,name in enumerate(names)}
        proj={}
        for cutoff in [.01,.05,.1]:
            keep=singular>=cutoff*singular[0]
            basis=vt[keep].T
            captured=np.linalg.norm(target@basis@basis.T)**2 / np.linalg.norm(target)**2
            proj[str(cutoff)]={
                'rank':int(keep.sum()),'target_sensitivity_energy_captured':float(captured),
                'target_sensitivity_energy_unseen':float(1-captured)}
        cell_result['modalities'][mode]={
            'singular_values':singular.tolist(),
            'parameter_observed_derivative_RMS':sensitivity,
            'target_projection_by_relative_singular_cutoff':proj,
        }
    rows.append(cell_result)
summary={
    'cells':rows,
    'mean_target_energy_captured':{
        mode:{cut:float(np.mean([row['modalities'][mode]['target_projection_by_relative_singular_cutoff'][cut]['target_sensitivity_energy_captured'] for row in rows]))
              for cut in ['0.01','0.05','0.1']}
        for mode in rows[0]['modalities']},
    'scope':'Four training cells only; local linearization; equalized trace scales AP100mV, Ca0.5µM, T50kPa; relative singular cutoffs are exploratory.',
}
(folder/'local_jacobian_audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary['mean_target_energy_captured'],indent=2))
print('beta sensitivities at one/two loads:')
for row in rows:
    print(row['cell_index_zero_based'],[(mode,
        row['modalities'][mode]['parameter_observed_derivative_RMS']['beta0'],
        row['modalities'][mode]['parameter_observed_derivative_RMS']['beta1'])
        for mode in ['AP+Ca+T1.0','AP+Ca+T1.0+T1.1']])
