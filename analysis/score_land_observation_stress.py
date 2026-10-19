"""Compare identical noisy calibration observations across three frozen methods."""
from pathlib import Path
import argparse
import json
import h5py
import numpy as np
from scipy.io import loadmat
import torch
import matplotlib.pyplot as plt
from train_direct_load_predictor import DirectLoadNetwork

parser=argparse.ArgumentParser()
parser.add_argument('--robust',action='store_true')
parser.add_argument('--conditions',default='low_noise,high_noise,drift_sparse')
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
base=root/'analysis/virtual_cardiomyocyte_5000'
manifest=json.loads((base/'land_observation_stress_manifest.json').read_text())
prefix='land_stress_robust' if args.robust else 'land_stress'
cells=np.array(manifest['cells'])
with h5py.File(base/'merged_cohort.h5') as f:
    lam=f['lambda'][()]
    ix=int(np.argmin(abs(lam-1.125)))
    assert abs(lam[ix]-1.125)<1e-6
    truth=f['Ta_kPa'][cells,ix,:601:2]
    assert np.all(f['split_code'][cells]==2)

torch.set_num_threads(4)
net=DirectLoadNetwork(2,301)
net.load_state_dict(torch.load(base/'direct_2load/model.pt',map_location='cpu',weights_only=True))
net.eval()
augmented_net=None
augmented_path=base/'direct_2load_noise_augmented/model.pt'
if augmented_path.exists():
    augmented_net=DirectLoadNetwork(2,301)
    augmented_net.load_state_dict(torch.load(augmented_path,map_location='cpu',weights_only=True))
    augmented_net.eval()
rng=np.random.default_rng(20260929)
results=[]
requested=args.conditions.split(',')
assert all(x in manifest['conditions'] for x in requested)
for condition in ['clean',*requested]:
    if condition=='clean':
        selection=json.loads((base/'land_blind_128_selection.json').read_text())
        assert cells.tolist()==selection['selected_test_cells'][:len(cells)]
        with h5py.File(base/'merged_cohort.h5') as f:
            l=f['lambda'][()]
            calix=[int(np.argmin(abs(l-x))) for x in [1.,1.1]]
            ap=f['V'][cells]
            ca=f['Cai_uM'][cells]
            t=f['Ta_kPa'][cells][:,calix,:]
        resultfiles=[base/'land_blind_128_results'/f'land_refine_test_{i:02d}.mat'
                     for i in range(1,len(cells)+1)]
    else:
        obs=loadmat(base/f'{prefix}_{condition}_input.mat')
        assert np.array_equal(obs['cell_index_zero_based'].ravel(),cells)
        ap=obs['V_mV'];ca=obs['Cai_uM'];t=obs['Tcal_kPa']
        resultfiles=[base/f'{prefix}_{condition}_results'/f'land_refine_test_{i:02d}.mat'
                     for i in range(1,len(cells)+1)]
    if not all(p.exists() for p in resultfiles):
        raise SystemExit(f'{condition}: missing {sum(not p.exists() for p in resultfiles)} refinements')
    twin=[]; exitflags=[]
    for cell,path in zip(cells,resultfiles):
        with h5py.File(path) as f:
            r=f['result']
            assert int(r['cell_index_zero_based'][()].item())==cell
            twin.append(r['after_target_waveform'][()].ravel()[:601:2])
            exitflags.append(int(r['exitflag'][()].item()))
    twin=np.stack(twin)
    linear=t[:,0,:601:2]+1.25*(t[:,1,:601:2]-t[:,0,:601:2])
    with torch.no_grad():
        ai=torch.tensor(ap[:,:601:2],dtype=torch.float32)
        ci=torch.tensor(ca[:,:601:2],dtype=torch.float32)
        ti=torch.tensor(t[:,:,:601:2],dtype=torch.float32)
        li=torch.tensor([1.,1.1],dtype=torch.float32).repeat(len(cells),1)
        target=torch.full((len(cells),),1.125)
        direct=net(ai,ci,ti,li,target).numpy()
        direct_aug=augmented_net(ai,ci,ti,li,target).numpy() if augmented_net else None
    err={key:np.mean(abs(pred-truth),axis=1) for key,pred in
         [('refined_twin',twin),('direct_NN',direct),('linear',linear)]}
    if direct_aug is not None:
        err['direct_NN_noise_augmented']=np.mean(abs(direct_aug-truth),axis=1)
    diff=err['direct_NN']-err['refined_twin']
    boot=rng.integers(0,len(cells),size=(3000,len(cells)))
    aug_diff=err['direct_NN_noise_augmented']-err['refined_twin'] if direct_aug is not None else None
    results.append({'condition':condition,'n_cells':len(cells),
        'waveform_MAE_kPa':{key:float(e.mean()) for key,e in err.items()},
        'median_waveform_MAE_kPa':{key:float(np.median(e)) for key,e in err.items()},
        'direct_minus_twin_MAE_95pct_CI_kPa':np.quantile(diff[boot].mean(axis=1),[.025,.975]).tolist(),
        'augmented_direct_minus_twin_MAE_95pct_CI_kPa':np.quantile(aug_diff[boot].mean(axis=1),[.025,.975]).tolist() if aug_diff is not None else None,
        'cells_twin_better_than_direct':int((diff>0).sum()),
        'cells_twin_better_than_augmented_direct':int(np.sum(err['refined_twin']<err['direct_NN_noise_augmented'])) if direct_aug is not None else None,
        'nonpositive_refinement_exitflags':int(np.sum(np.array(exitflags)<=0))})
report={'n_cells':len(cells),'target_lambda':1.125,'calibration_lambdas':[1.,1.1],
    'preprocessing':'Training-only noise-augmented Ridge plus fixed Savitzky-Golay filters' if args.robust else 'None; original clean-trained Ridge and direct network',
    'results':results,
    'caveats':'Synthetic measurement perturbations only; levels are stress assumptions rather than estimates from real instruments. Clean-trained and noise-augmented direct networks are both scored. The robust variant uses training-cell noise augmentation for Ridge. Ground truth target is clean.'}
suffix='' if len(requested)==len(manifest['conditions']) else '_partial'
(base/f'{prefix}_score{suffix}.json').write_text(json.dumps(report,indent=2))
fig,ax=plt.subplots(figsize=(8,4.5),constrained_layout=True)
labels=[r['condition'].replace('_',' ') for r in results]
x=np.arange(len(labels));width=.25
bars=[('refined_twin','#2d7f5e'),('direct_NN','#d5782d'),('linear','#5377b8')]
if augmented_net: bars.insert(2,('direct_NN_noise_augmented','#9e4f30'))
width=.75/len(bars)
for k,(name,color) in enumerate(bars):
    ax.bar(x+(k-(len(bars)-1)/2)*width,[r['waveform_MAE_kPa'][name] for r in results],
           width,label=name.replace('_',' '),color=color)
ax.set_xticks(x,labels)
ax.set(ylabel='Mean clean-target waveform MAE (kPa)',title='Synthetic observation stress')
ax.legend();ax.grid(axis='y',alpha=.2)
fig.savefig(base/f'{prefix}_comparison{suffix}.png',dpi=180)
print(json.dumps(report,indent=2))
