"""Why did 12->24 candidates change sealed 32/32 waveform coverage?"""
from pathlib import Path
import json
import h5py
import numpy as np
from scipy.stats import ks_2samp

base=Path(__file__).resolve().parent/'virtual_cardiomyocyte_5000'
files={12:'land_eight_direction_ensemble_confirm64.mat',
       24:'land_eight_direction_ensemble_confirm64_24.mat'}
with h5py.File(base/'merged_cohort.h5') as f:
    lam=f['lambda'][()]
    jj=int(np.argmin(abs(lam-1.125)))
    assert abs(lam[jj]-1.125)<1e-6
    cells=json.loads((base/'land_stress_confirmation_manifest.json').read_text())['selected_test_cells']
    truth_1125=f['Ta_kPa'][cells][:,jj,:601:2]
    theta=f['theta'][cells]
    lo=f['lo'][()];hi=f['hi'][()]
with h5py.File(base/'land_confirm_115_truth.mat') as f:
    truth_115=f['truth'][()].transpose(2,1,0)[:,0,::2]
truth=np.stack([truth_1125,truth_115],axis=1)
cal=np.arange(32);test=np.arange(32,64)
rank=30
report={'calibration_cells_first32':cells[:32],
        'evaluation_cells_last32':cells[32:],
        'theta_unit_mean_cal':((theta[cal]-lo)/(hi-lo)).mean(axis=0).tolist(),
        'theta_unit_mean_test':((theta[test]-lo)/(hi-lo)).mean(axis=0).tolist(),
        'theta_KS_pvalues':[float(ks_2samp(theta[cal,j],theta[test,j]).pvalue) for j in range(theta.shape[1])],
        'target_peak_mean_cal':truth[cal].max(axis=-1).mean(axis=0).tolist(),
        'target_peak_mean_test':truth[test].max(axis=-1).mean(axis=0).tolist(),
        'counts':{}}
for count,path in files.items():
    with h5py.File(base/path) as f:
        d=f['prediction'][()].transpose(3,2,1,0)
        w=f['weights'][()].T
        assert np.array_equal(f['cell_index_zero_based'][()].ravel().astype(int),cells)
    entries=[]
    for j,target in enumerate([1.125,1.15]):
        dd=d[:,j]
        mu=np.sum(dd*w[:,None,:],axis=-1)
        scale=np.maximum(np.sqrt(np.sum(w[:,None,:]*(dd-mu[:,:,None])**2,axis=-1)),.5)
        score=np.max(abs(truth[:,j]-mu)/scale,axis=1)
        q=float(np.partition(score[cal],rank-1)[rank-1])
        newly_bad=np.flatnonzero(score[test]>q)
        entries.append({'target_lambda':target,'q':q,
            'calibration_score_median':float(np.median(score[cal])),
            'test_score_median':float(np.median(score[test])),
            'calibration_score_90pct':float(np.quantile(score[cal],.9)),
            'test_score_90pct':float(np.quantile(score[test],.9)),
            'mean_scale_cal_kPa':float(scale[cal].mean()),
            'mean_scale_test_kPa':float(scale[test].mean()),
            'mean_center_MAE_cal_kPa':float(np.mean(abs(truth[cal,j]-mu[cal]))),
            'mean_center_MAE_test_kPa':float(np.mean(abs(truth[test,j]-mu[test]))),
            'missed_test_cell_ids':[cells[32+int(i)] for i in newly_bad],
            'missed_test_cell_count':int(len(newly_bad))})
    report['counts'][str(count)]=entries
(base/'land_ensemble_count_shift_diagnostic.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'target_peak_cal':report['target_peak_mean_cal'],
                  'target_peak_test':report['target_peak_mean_test'],
                  'theta_KS_min_p':min(report['theta_KS_pvalues']),
                  'counts':report['counts']},indent=2))
