"""Compare ridge-initialized true-ODE twin with same-cell strong baselines."""
import json
from pathlib import Path

import h5py
import numpy as np

root=Path(__file__).resolve().parents[1]
base=root/'analysis/virtual_cardiomyocyte_5000'
with h5py.File(base/'ridge_twin_forward_16.mat') as f:
    y=f['Ta_kPa'][()].T[:, :601:2]
    cells=f['cell_index_zero_based'][()].ravel().astype(int)
    success=f['ok'][()].ravel().astype(bool)
assert success.all() and len(np.unique(cells))==16
with h5py.File(base/'merged_cohort.h5') as f:
    lam=f['lambda'][()]
    j=int(np.argmin(abs(lam-1.125)))
    assert abs(lam[j]-1.125)<1e-8
    truth=f['Ta_kPa'][cells,j,:601:2]
posterior=json.loads((base/'posterior_true_forward_16x8_score.json').read_text())
comparators={row['cell']:row for row in posterior['modes']['3']['per_cell']}
assert set(cells)==set(comparators)
rows=[]
for c,yp,yt in zip(cells,y,truth):
    rows.append({'cell':int(c),
        'ridge_twin_waveform_MAE_kPa':float(abs(yp-yt).mean()),
        'ridge_twin_peak_MAE_kPa':float(abs(yp.max()-yt.max())),
        'posterior_twin_waveform_MAE_kPa':comparators[int(c)]['posterior_waveform_MAE_kPa'],
        'direct_waveform_MAE_kPa':comparators[int(c)]['direct_waveform_MAE_kPa'],
        'linear_waveform_MAE_kPa':comparators[int(c)]['static_waveform_MAE_kPa']})
metrics=['ridge_twin_waveform_MAE_kPa','posterior_twin_waveform_MAE_kPa',
         'direct_waveform_MAE_kPa','linear_waveform_MAE_kPa']
result={'n_cells':len(cells),'mean':{k:float(np.mean([r[k] for r in rows])) for k in metrics},
        'per_cell':rows,'note':'All methods compared on identical 16 frozen-test cells; ridge was fit on other cells only.'}
(base/'ridge_true_forward_16_score.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result['mean'],indent=2))
