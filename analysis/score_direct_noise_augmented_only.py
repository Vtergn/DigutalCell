"""Score trained direct network before Land stress jobs finish."""
from pathlib import Path
import json
import h5py
import numpy as np
from scipy.io import loadmat
import torch
from train_direct_load_predictor import DirectLoadNetwork

root=Path(__file__).resolve().parents[1]
base=root/'analysis/virtual_cardiomyocyte_5000'
cells=json.loads((base/'land_observation_stress_manifest.json').read_text())['cells']
with h5py.File(base/'merged_cohort.h5') as f:
    lam=f['lambda'][()]
    target=int(np.argmin(abs(lam-1.125)))
    cal=[int(np.argmin(abs(lam-x))) for x in [1.,1.1]]
    truth=f['Ta_kPa'][cells,target,:601:2]
    clean=(f['V'][cells],f['Cai_uM'][cells],f['Ta_kPa'][cells][:,cal,:])
torch.set_num_threads(4)
models={}
for name,path in [('clean_trained',base/'direct_2load/model.pt'),
                  ('noise_augmented',base/'direct_2load_noise_augmented/model.pt')]:
    model=DirectLoadNetwork(2,301)
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
    model.eval();models[name]=model
rows=[]
for condition in ['clean','low_noise','high_noise','drift_sparse']:
    if condition=='clean': v,ca,t=clean
    else:
        data=loadmat(base/f'land_stress_robust_{condition}_input.mat')
        v,ca,t=data['V_mV'],data['Cai_uM'],data['Tcal_kPa']
    v=torch.tensor(v[:,:601:2],dtype=torch.float32)
    ca=torch.tensor(ca[:,:601:2],dtype=torch.float32)
    t=torch.tensor(t[:,:,:601:2],dtype=torch.float32)
    cal_lam=torch.tensor([1.,1.1]).repeat(len(cells),1)
    query=torch.full((len(cells),),1.125)
    values={}
    with torch.no_grad():
        for name,model in models.items():
            pred=model(v,ca,t,cal_lam,query).numpy()
            values[name]=float(np.mean(abs(pred-truth)))
    linear=t[:,0,:].numpy()+1.25*(t[:,1,:].numpy()-t[:,0,:].numpy())
    values['linear']=float(np.mean(abs(linear-truth)))
    rows.append({'condition':condition,'mean_waveform_MAE_kPa':values})
(base/'direct_noise_augmented_score.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
