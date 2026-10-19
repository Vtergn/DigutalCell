"""Score fixed true-Land refinement against fixed direct NN and linear load baseline."""
from pathlib import Path
import json
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from train_direct_load_predictor import DirectLoadNetwork, get_data

root=Path(__file__).resolve().parents[1]
base=root/'analysis/virtual_cardiomyocyte_5000'
manifest=json.loads((base/'land_load_grid_manifest.json').read_text())
cells=np.array(json.loads((base/'land_blind_128_selection.json').read_text())['selected_test_cells'])
lam=np.array(manifest['target_lambdas'])
with h5py.File(base/'land_load_grid_predictions.mat') as f:
    predicted=f['prediction'][()].transpose(2,1,0)
    failed_prediction=f['failed'][()].ravel().astype(bool)
    assert np.array_equal(f['cell_index_zero_based'][()].ravel(),cells)
with h5py.File(base/'land_load_grid_truth.mat') as f:
    truth=f['truth'][()].transpose(2,1,0)
    failed_truth=f['failed'][()].ravel().astype(bool)
    assert np.array_equal(f['cell_index_zero_based'][()].ravel(),cells)
valid=~(failed_prediction|failed_truth)
assert predicted.shape==(len(cells),len(lam),601)
assert truth.shape==predicted.shape
if not valid.any(): raise ValueError('No valid cells on load grid')
data=get_data(base/'merged_cohort.h5')
ap,ca,t,_,cohort_lam=data
ix=[int(np.argmin(abs(cohort_lam-x))) for x in [1.,1.1,1.125]]
assert np.allclose(cohort_lam[ix],[1.,1.1,1.125])
torch.set_num_threads(4)
net=DirectLoadNetwork(2,ap.shape[1])
net.load_state_dict(torch.load(base/'direct_2load/model.pt',map_location='cpu',weights_only=True))
net.eval()
aug_net=DirectLoadNetwork(2,ap.shape[1])
aug_net.load_state_dict(torch.load(base/'direct_2load_noise_augmented/model.pt',map_location='cpu',weights_only=True))
aug_net.eval()
linear=[]; direct=[]; direct_aug=[]
with torch.no_grad():
    for target in lam:
        linear.append((t[cells,ix[0],:]+float((target-1.)/.1)*
                      (t[cells,ix[1],:]-t[cells,ix[0],:])).numpy())
        direct.append(net(ap[cells],ca[cells],t[cells][:,ix[:2],:],
            torch.tensor([1.,1.1],dtype=torch.float32).repeat(len(cells),1),
            torch.full((len(cells),),float(target))).numpy())
        direct_aug.append(aug_net(ap[cells],ca[cells],t[cells][:,ix[:2],:],
            torch.tensor([1.,1.1],dtype=torch.float32).repeat(len(cells),1),
            torch.full((len(cells),),float(target))).numpy())
linear=np.stack(linear,axis=1)
direct=np.stack(direct,axis=1)
direct_aug=np.stack(direct_aug,axis=1)
truth2=truth[:,:,::2]
error={
    'refined_twin':np.abs(predicted[:,:,::2]-truth2).mean(axis=2),
    'direct_NN':np.abs(direct-truth2).mean(axis=2),
    'direct_NN_noise_augmented':np.abs(direct_aug-truth2).mean(axis=2),
    'linear':np.abs(linear-truth2).mean(axis=2),
}
peak={
    'refined_twin':np.abs(predicted.max(axis=2)-truth.max(axis=2)),
    'direct_NN':np.abs(direct.max(axis=2)-truth2.max(axis=2)),
    'direct_NN_noise_augmented':np.abs(direct_aug.max(axis=2)-truth2.max(axis=2)),
    'linear':np.abs(linear.max(axis=2)-truth2.max(axis=2)),
}
rng=np.random.default_rng(20260929)
rows=[]
for j,target in enumerate(lam):
    e={k:v[valid,j] for k,v in error.items()}
    boot=rng.integers(0,valid.sum(),size=(3000,valid.sum()))
    diff=e['direct_NN']-e['refined_twin']
    rows.append({'target_lambda':float(target),
        'distance_from_calibration_interval':float(max(1.-target,target-1.1,0.)),
        'side':'below' if target<1 else 'above',
        'n_valid_cells':int(valid.sum()),
        'true_mean_peak_tension_kPa':float(truth[valid,j].max(axis=1).mean()),
        'waveform_MAE_kPa':{k:float(v.mean()) for k,v in e.items()},
        'waveform_MAE_divided_by_mean_true_peak':{k:float(v.mean()/truth[valid,j].max(axis=1).mean()) for k,v in e.items()},
        'peak_MAE_kPa':{k:float(v[valid,j].mean()) for k,v in peak.items()},
        'twin_minus_direct_mean_MAE_kPa':float(-diff.mean()),
        'direct_minus_twin_mean_MAE_95pct_CI_kPa':np.quantile(diff[boot].mean(axis=1),[.025,.975]).tolist(),
        'cells_twin_better_than_direct':int((diff>0).sum())})
    rows[-1]['cells_twin_better_than_augmented_direct']=int(np.sum(e['refined_twin']<e['direct_NN_noise_augmented']))
result={'n_cells_total':len(cells),'n_cells_valid':int(valid.sum()),
        'prediction_failures':int(failed_prediction.sum()),
        'truth_failures':int(failed_truth.sum()),
        'calibration_lambdas':[1.,1.1],
        'targets':rows,
        'scope':'Same Land generator supplies both virtual truth and refinement; fixed, clean calibration and no retraining across target loads.'}
(base/'land_load_grid_score.json').write_text(json.dumps(result,indent=2))
fig,ax=plt.subplots(figsize=(8,4.7),constrained_layout=True)
for key,label,color in [('refined_twin','Refined Land twin','#2d7f5e'),
                        ('direct_NN_noise_augmented','Noise-augmented direct NN','#d5782d'),
                        ('linear','Linear','#5377b8')]:
    for side,ls in [('below','--'),('above','-')]:
        rr=[r for r in rows if r['side']==side]
        ax.plot([r['distance_from_calibration_interval'] for r in rr],
                [r['waveform_MAE_kPa'][key] for r in rr],
                marker='o',ls=ls,color=color,label=label+(' (low)' if side=='below' else ' (high)'))
for r in rows:
    ax.annotate(str(r['target_lambda']),
                (r['distance_from_calibration_interval'],r['waveform_MAE_kPa']['refined_twin']),
                xytext=(2,5),textcoords='offset points',fontsize=8)
ax.set(xlabel='Distance outside calibrated stretch range',ylabel='Mean waveform MAE (kPa)',
       title='Fixed-method prediction across unseen loads')
ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2)
fig.savefig(base/'land_load_grid_comparison.png',dpi=180)
print(json.dumps({'n_valid':result['n_cells_valid'],'targets':rows},indent=2))
