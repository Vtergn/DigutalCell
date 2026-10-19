"""Exploratory cell-level conformal bands on the frozen clean load grid."""
from pathlib import Path
import json
import h5py
import numpy as np

base=Path(__file__).resolve().parent/'virtual_cardiomyocyte_5000'
with h5py.File(base/'land_load_grid_predictions.mat') as f:
    pred=f['prediction'][()].transpose(2,1,0)
    cells=f['cell_index_zero_based'][()].ravel().astype(int)
with h5py.File(base/'land_load_grid_truth.mat') as f:
    true=f['truth'][()].transpose(2,1,0)
    assert np.array_equal(cells,f['cell_index_zero_based'][()].ravel().astype(int))
assert pred.shape==true.shape and pred.shape[0]==128
cal=np.arange(64);test=np.arange(64,128)
k=int(np.ceil((len(cal)+1)*.9))
rows=[]
for j,lam in [(4,1.125),(5,1.15)]:
    score=np.max(abs(pred[cal,j,::2]-true[cal,j,::2]),axis=1)
    q=float(np.partition(score,k-1)[k-1])
    p=pred[test,j,::2];y=true[test,j,::2]
    lo=np.maximum(0,p-q);hi=p+q
    cover=(y>=lo)&(y<=hi)
    rows.append({'target_lambda':lam,'n_calibration_cells':64,'n_test_cells':64,
                 'k':k,'q_kPa':q,'point_MAE_kPa':float(np.mean(abs(p-y))),
                 'simultaneous_coverage':float(cover.all(axis=1).mean()),
                 'covered_cells':int(cover.all(axis=1).sum()),
                 'pointwise_coverage':float(cover.mean()),
                 'mean_width_kPa':float(np.mean(hi-lo)),
                 'interval_score_kPa':float(np.mean(hi-lo+20*np.maximum(lo-y,0)+20*np.maximum(y-hi,0)))})
report={'results':rows,'split':'First 64 vs last 64 clean held-out cells in frozen order',
        'score':'Maximum absolute error over 301 target waveform samples per cell',
        'scope':'Exploratory only: the same 128-cell cohort informed earlier model choices. These two target loads do not constitute a newly sealed independent UQ test.'}
(base/'land_conformal_clean_grid_score.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
