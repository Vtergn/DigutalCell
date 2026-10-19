"""Score independent 64-cell robustness confirmation after all target-blind fits."""
from pathlib import Path
import json
import h5py
import numpy as np
from scipy.io import loadmat
import torch
import matplotlib.pyplot as plt
from train_direct_load_predictor import DirectLoadNetwork

root=Path(__file__).resolve().parents[1]
base=root/'analysis/virtual_cardiomyocyte_5000'
manifest=json.loads((base/'land_stress_confirmation_manifest.json').read_text())
cells=np.array(manifest['selected_test_cells'])
with h5py.File(base/'merged_cohort.h5') as f:
    lam=f['lambda'][()]
    ti=int(np.argmin(abs(lam-1.125)))
    assert abs(lam[ti]-1.125)<1e-6
    truth=f['Ta_kPa'][cells,ti,:601:2]
    assert np.all(f['split_code'][cells]==2)
torch.set_num_threads(4)
models={}
for key,path in [('direct_NN',base/'direct_2load/model.pt'),
                 ('direct_NN_noise_augmented',base/'direct_2load_noise_augmented/model.pt')]:
    net=DirectLoadNetwork(2,301)
    net.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
    net.eval();models[key]=net
rng=np.random.default_rng(20261001)
rows=[]
for condition in manifest['conditions']:
    obs=loadmat(base/f'land_stress_confirm_{condition}_input.mat')
    assert np.array_equal(obs['cell_index_zero_based'].ravel(),cells)
    ap=torch.tensor(obs['V_mV'][:,:601:2],dtype=torch.float32)
    ca=torch.tensor(obs['Cai_uM'][:,:601:2],dtype=torch.float32)
    t=torch.tensor(obs['Tcal_kPa'][:,:,:601:2],dtype=torch.float32)
    li=torch.tensor([1.,1.1]).repeat(len(cells),1)
    target=torch.full((len(cells),),1.125)
    prediction={}
    with torch.no_grad():
        for key,net in models.items():
            prediction[key]=net(ap,ca,t,li,target).numpy()
    prediction['linear']=t[:,0,:].numpy()+1.25*(t[:,1,:].numpy()-t[:,0,:].numpy())
    twin=[];exitflags=[];evaluations=[]
    for i,cell in enumerate(cells,1):
        path=base/f'land_stress_confirm_{condition}_results'/f'land_refine_test_{i:02d}.mat'
        if not path.exists():raise SystemExit(f'Missing {path}')
        with h5py.File(path) as f:
            r=f['result']
            assert int(r['cell_index_zero_based'][()].item())==cell
            twin.append(r['after_target_waveform'][()].ravel()[:601:2])
            exitflags.append(int(r['exitflag'][()].item()))
            evaluations.append(int(r['function_evaluations'][()].item()))
    prediction['refined_twin']=np.stack(twin)
    error={k:np.mean(abs(p-truth),axis=1) for k,p in prediction.items()}
    peak={k:abs(p.max(axis=1)-truth.max(axis=1)) for k,p in prediction.items()}
    boot=rng.integers(0,len(cells),size=(5000,len(cells)))
    delta=error['direct_NN_noise_augmented']-error['refined_twin']
    rows.append({'condition':condition,'n_cells':len(cells),
        'waveform_MAE_kPa':{k:float(v.mean()) for k,v in error.items()},
        'median_waveform_MAE_kPa':{k:float(np.median(v)) for k,v in error.items()},
        'peak_MAE_kPa':{k:float(v.mean()) for k,v in peak.items()},
        'augmented_direct_minus_twin_mean_MAE_kPa':float(delta.mean()),
        'augmented_direct_minus_twin_95pct_paired_bootstrap_CI_kPa':np.quantile(delta[boot].mean(axis=1),[.025,.975]).tolist(),
        'cells_twin_better_than_augmented_direct':int((delta>0).sum()),
        'nonpositive_exitflag_count':int((np.array(exitflags)<=0).sum()),
        'median_function_evaluations':float(np.median(evaluations))})
result={'n_fresh_test_cells':len(cells),'cells_excluded_from_prior_confirmation_and_pilots':True,
        'target_lambda':1.125,'calibration_lambdas':[1.,1.1],
        'pipeline_fixed_before_fresh_cell_selection':True,
        'results':rows,
        'scope':'Same forward Land model as virtual truth; synthetic noise assumptions, not real measurement noise. Target withheld from MATLAB inputs.'}
(base/'land_stress_confirmation_score.json').write_text(json.dumps(result,indent=2))
fig,ax=plt.subplots(figsize=(8.2,4.9),constrained_layout=True)
x=np.arange(len(rows));width=.19
for k,(name,color) in enumerate([('refined_twin','#2d7f5e'),('direct_NN_noise_augmented','#9e4f30'),
                                  ('direct_NN','#d5782d'),('linear','#5377b8')]):
    ax.bar(x+(k-1.5)*width,[r['waveform_MAE_kPa'][name] for r in rows],width,
           label=name.replace('_',' '),color=color)
ax.set_xticks(x,[r['condition'].replace('_',' ') for r in rows])
ax.set(ylabel='Clean-target waveform MAE (kPa)',title='Independent robustness confirmation (64 cells)')
ax.legend(fontsize=8,ncol=4,loc='upper center',bbox_to_anchor=(.5,-.15))
ax.grid(axis='y',alpha=.2)
fig.savefig(base/'land_stress_confirmation_comparison.png',dpi=180)
print(json.dumps(result,indent=2))
