"""Target-blind rules already fixed; diagnose the sealed UQ confirmation."""
from pathlib import Path
import json
import h5py
import numpy as np
from scipy.io import loadmat

base=Path(__file__).resolve().parent/'virtual_cardiomyocyte_5000'
manifest=json.loads((base/'land_uq_fresh64_manifest.json').read_text())
cells=np.array(manifest['cells'])
cal=np.array(manifest['calibration_rows_zero_based'])
test=np.array(manifest['evaluation_rows_zero_based'])
obs=loadmat(base/'land_uq_fresh64_low_noise_input.mat')['Tcal_kPa']
observed_peak=obs[:,1,:601].max(axis=1)
with h5py.File(base/'land_uq_fresh64_ensemble24.mat') as f:
    d=f['prediction'][()].transpose(3,2,1,0)
    w=f['weights'][()].T
    assert np.array_equal(cells,f['cell_index_zero_based'][()].ravel().astype(int))
with h5py.File(base/'merged_cohort.h5') as f:
    lambdas=f['lambda'][()]
    j=int(np.argmin(abs(lambdas-1.125)))
    truth_1125=f['Ta_kPa'][cells][:,j,:601:2]
with h5py.File(base/'land_uq_fresh64_115_truth.mat') as f:
    truth_115=f['truth'][()].transpose(2,1,0)[:,0,::2]
truth=np.stack([truth_1125,truth_115],axis=1)
result=[]
for j,target in enumerate([1.125,1.15]):
    x=d[:,j]
    mean=np.sum(w[:,None,:]*x,axis=-1)
    scale=np.maximum(np.sqrt(np.sum(w[:,None,:]*(x-mean[:,:,None])**2,axis=-1)),.5)
    score=np.max(abs(truth[:,j]-mean)/scale,axis=1)
    ordered=np.argsort(score[cal])
    cutoff=float(score[cal][ordered[29]])
    missed=test[score[test]>cutoff]
    result.append({'target_lambda':target,
        'calibration_top_5_scores':np.sort(score[cal])[-5:].tolist(),
        'calibration_top_5_cell_ids':cells[cal[np.argsort(score[cal])[-5:]]].tolist(),
        'calibration_q_rank30':cutoff,
        'test_score_median':float(np.median(score[test])),
        'test_score_max':float(np.max(score[test])),
        'missed_test_cell_ids':cells[missed].tolist(),
        'missed_test_observed_T11_peaks_kPa':observed_peak[missed].tolist(),
        'observed_T11_peak_mean_cal_kPa':float(observed_peak[cal].mean()),
        'observed_T11_peak_mean_test_kPa':float(observed_peak[test].mean()),
        'true_target_peak_mean_cal_kPa':float(truth[cal,j].max(axis=1).mean()),
        'true_target_peak_mean_test_kPa':float(truth[test,j].max(axis=1).mean()),
        'ensemble_ESS_median_cal':float(np.median(1/np.sum(w[cal]**2,axis=1))),
        'ensemble_ESS_median_test':float(np.median(1/np.sum(w[test]**2,axis=1)))})
report={'n_calibration':32,'n_evaluation':32,'results':result,
        'scope':'Post-score descriptive audit only; no changes to the pre-registered prediction or interval method.'}
(base/'land_uq_fresh64_diagnostic.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
